import os
import re
import json
import hashlib
import base64
import logging
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import wikipedia

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fitness AI Trainer",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

INSTRUCTIONS_FILE = BASE_DIR / "custom_instructions.txt"
USER_PROFILE_FILE = BASE_DIR / "user_profile.json"
CHAT_HISTORY_DIR = BASE_DIR / "chat_history"

CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("fitness_ai_trainer")


# ============================================================
# API CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = None

# Streamlit Cloud
try:
    API_KEY = st.secrets.get("GROQ_API_KEY")
except Exception:
    API_KEY = None

# Local .env fallback
if not API_KEY:
    API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    st.error(
        "GROQ_API_KEY is missing. "
        "Add it to Streamlit Secrets or your local .env file."
    )
    st.stop()

client = Groq(api_key=API_KEY)

MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "qwen/qwen3.6-27b"


# ============================================================
# CONFIGURATION
# ============================================================

KB_THRESHOLD = 0.08
MIN_SIMILARITY_SCORE = 0.10

WIKI_RESULTS = 2
WIKI_SENTENCES = 5
WEB_RESULTS = 3

MAX_CONTEXT_MESSAGES = 10

MAX_IMAGE_SIZE = 20 * 1024 * 1024

SUPPORTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}

WIKI_LANG_MAP = {
    "English": "en",
    "العربية": "ar"
}


# ============================================================
# LANGUAGE
# ============================================================

LANGUAGE = {
    "English": {
        "welcome": (
            "Hello! I'm your Fitness AI Trainer. "
            "Ask me anything about fitness, nutrition, or workout plans."
        ),
        "outside": (
            "Sorry, I couldn't find an answer in the Knowledge Base, "
            "Wikipedia, or the web."
        ),
        "empty": (
            "Your Knowledge Base is empty. "
            "Please put a PDF inside the knowledge_base folder."
        ),
        "sensitive": (
            "For security reasons, please do not send passwords, API keys, "
            "OTP codes, or other sensitive information."
        ),
        "new": "🆕 New Chat",
        "delete": "🗑️ Delete Current Chat",
        "history": "💬 Chat History",
        "knowledge": "📚 Knowledge Base",
        "language": "🌍 Language",
        "human": "👨‍💼 Human Support",
        "summary": "📝 Conversation Summary",
        "reload": "🔄 Reload Knowledge Base",
        "instructions": "🧭 Custom Instructions",
        "instructions_placeholder": (
            "e.g. Always be motivating. Focus on safe exercise practices."
        ),
        "save_instructions": "💾 Save Instructions",
        "instructions_saved": "Instructions saved.",
        "onboard_title": "👋 Before we start...",
        "onboard_name": "What's your name?",
        "onboard_age": "What's your age?",
        "onboard_button": "Start Chat",
        "onboard_greeting": (
            "Hello {name}! Since you're {age}, I'll keep that in mind. "
            "How can I help with your fitness journey today?"
        ),
        "onboard_greeting_returning": (
            "Welcome back, {name}! Ready for another fitness session?"
        ),
        "profile_signed_in": "👤 Signed in as {name} ({age})",
        "profile_reset": "🔄 Not you? Reset profile",
        "thinking": "🤖 Thinking...",
        "searching_kb": "🔍 Searching the Knowledge Base...",
        "searching_wiki": "🌐 Searching Wikipedia...",
        "searching_web": "🌍 Searching the web...",
        "generating_answer": "✍️ Generating answer...",
        "sources_found": "✅ Found {count} relevant source(s).",
        "no_sources": "❌ No relevant sources found.",
        "error_occurred": "⚠️ An error occurred: {error}",
        "summary_generating": "Creating conversation summary...",
        "summary_generated": "Conversation Summary",
        "feedback_thanks": "Thank you for your feedback!",
        "escalation_requested": "👨‍💼 Human support has been requested.",
        "export_chat": "📤 Export Chat"
    },

    "العربية": {
        "welcome": (
            "مرحباً! أنا مدرب اللياقة البدنية الذكي. "
            "يمكنك سؤالي عن اللياقة أو التغذية أو خطط التمارين."
        ),
        "outside": (
            "عذراً، لم أتمكن من إيجاد إجابة في قاعدة المعرفة "
            "أو ويكيبيديا أو الويب."
        ),
        "empty": (
            "قاعدة المعرفة فارغة. يرجى وضع ملف PDF داخل مجلد "
            "knowledge_base."
        ),
        "sensitive": (
            "لأسباب أمنية، يرجى عدم إرسال كلمات المرور أو مفاتيح API "
            "أو رموز OTP أو أي معلومات سرية."
        ),
        "new": "🆕 محادثة جديدة",
        "delete": "🗑️ حذف المحادثة الحالية",
        "history": "💬 سجل المحادثات",
        "knowledge": "📚 قاعدة المعرفة",
        "language": "🌍 اللغة",
        "human": "👨‍💼 الدعم البشري",
        "summary": "📝 ملخص المحادثة",
        "reload": "🔄 إعادة تحميل قاعدة المعرفة",
        "instructions": "🧭 تعليمات مخصصة",
        "instructions_placeholder": (
            "مثال: كن محفزاً دائماً. ركز على ممارسات التمرين الآمنة."
        ),
        "save_instructions": "💾 حفظ التعليمات",
        "instructions_saved": "تم حفظ التعليمات.",
        "onboard_title": "👋 قبل أن نبدأ...",
        "onboard_name": "ما اسمك؟",
        "onboard_age": "كم عمرك؟",
        "onboard_button": "ابدأ المحادثة",
        "onboard_greeting": (
            "مرحباً {name}! بما أن عمرك {age}، سأضع ذلك في اعتباري. "
            "كيف يمكنني مساعدتك اليوم؟"
        ),
        "onboard_greeting_returning": (
            "مرحباً بعودتك، {name}! هل أنت مستعد لجلسة لياقة أخرى؟"
        ),
        "profile_signed_in": "👤 مسجل الدخول باسم {name} ({age})",
        "profile_reset": "🔄 لست أنت؟ إعادة تعيين الملف الشخصي",
        "thinking": "🤖 أفكر...",
        "searching_kb": "🔍 أبحث في قاعدة المعرفة...",
        "searching_wiki": "🌐 أبحث في ويكيبيديا...",
        "searching_web": "🌍 أبحث في الويب...",
        "generating_answer": "✍️ أنشئ الإجابة...",
        "sources_found": "✅ وجدت {count} مصدر متعلق.",
        "no_sources": "❌ لم أجد مصادر متعلقة.",
        "error_occurred": "⚠️ حدث خطأ: {error}",
        "summary_generating": "إنشاء ملخص المحادثة...",
        "summary_generated": "ملخص المحادثة",
        "feedback_thanks": "شكراً لك على التقييم!",
        "escalation_requested": "👨‍💼 تم طلب الدعم البشري.",
        "export_chat": "📤 تصدير المحادثة"
    }
}


# ============================================================
# CSS
# ============================================================

def load_css():
    css = """
    .agent-header {
        background: linear-gradient(
            135deg,
            #667eea 0%,
            #764ba2 100%
        );
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.20);
    }

    .agent-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .source-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 2px;
    }

    .user-message {
        background-color: #e3f2fd;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 15px;
    }

    .assistant-message {
        background-color: #f5f5f5;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 15px;
    }

    [data-testid="stChatInput"] {
        background-color: #000000 !important;
        border: 1px solid #333333 !important;
        border-radius: 10px !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #000000 !important;
        color: #ffffff !important;
        caret-color: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #bbbbbb !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] button svg {
        fill: #ffffff !important;
    }
    """

    st.markdown(
        "<style>{}</style>".format(css),
        unsafe_allow_html=True
    )


load_css()


# ============================================================
# UTILITY
# ============================================================

def get_timestamp():
    return datetime.now().isoformat()


def contains_sensitive_data(text):
    patterns = [
        r"sk-[A-Za-z0-9_-]{20,}",
        r"gsk_[A-Za-z0-9_-]{20,}",
        r"password\s*[:=]\s*\S+",
        r"Bearer\s+[A-Za-z0-9._~+/=-]+",
        r"\b(?:otp|verification code)\s*[:=]?\s*\d{4,8}\b",
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        r"(?:\d{1,3}\.){3}\d{1,3}",
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{12}"
    ]

    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


# ============================================================
# USER PROFILE
# ============================================================

def load_user_profile():
    if not USER_PROFILE_FILE.exists():
        return {
            "name": None,
            "age": None,
            "preferences": {}
        }

    try:
        with open(USER_PROFILE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data

    except Exception as e:
        logger.error("Profile loading error: %s", e)

    return {
        "name": None,
        "age": None,
        "preferences": {}
    }


def save_user_profile(name, age, preferences=None):
    if preferences is None:
        preferences = {}

    data = {
        "name": name,
        "age": age,
        "preferences": preferences,
        "last_updated": get_timestamp()
    }

    try:
        with open(USER_PROFILE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )
    except Exception as e:
        logger.error("Profile save error: %s", e)


def clear_user_profile():
    try:
        if USER_PROFILE_FILE.exists():
            USER_PROFILE_FILE.unlink()
    except Exception as e:
        logger.error("Profile delete error: %s", e)


# ============================================================
# SESSION STATE
# ============================================================

if "language" not in st.session_state:
    st.session_state.language = "English"

if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "documents" not in st.session_state:
    st.session_state.documents = []

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "matrix" not in st.session_state:
    st.session_state.matrix = None

if "custom_instructions" not in st.session_state:

    if INSTRUCTIONS_FILE.exists():
        try:
            with open(
                INSTRUCTIONS_FILE,
                "r",
                encoding="utf-8"
            ) as f:
                st.session_state.custom_instructions = f.read()

        except Exception:
            st.session_state.custom_instructions = ""

    else:
        st.session_state.custom_instructions = ""


T = LANGUAGE[st.session_state.language]


# ============================================================
# CHAT CREATION
# ============================================================

def make_new_chat_dict():

    profile = load_user_profile()

    has_profile = bool(profile.get("name"))

    chat_id = hashlib.md5(
        os.urandom(20)
    ).hexdigest()[:8]

    chat = {
        "id": chat_id,
        "title": "New Chat",
        "messages": [],
        "escalated": False,
        "user_name": profile.get("name"),
        "user_age": profile.get("age"),
        "onboarded": has_profile,
        "created_at": get_timestamp(),
        "last_updated": get_timestamp(),
        "tags": [],
        "summary": ""
    }

    if has_profile:

        greeting = T[
            "onboard_greeting_returning"
        ].format(
            name=chat["user_name"]
        )

        chat["messages"].append({
            "role": "assistant",
            "content": greeting,
            "timestamp": get_timestamp()
        })

    return chat


if "current_chat" not in st.session_state:

    chat = make_new_chat_dict()

    st.session_state.current_chat = chat["id"]

    st.session_state.conversations[
        chat["id"]
    ] = chat


# ============================================================
# PDF FUNCTIONS
# ============================================================

def read_pdf(pdf_file):

    text = ""

    try:

        reader = PdfReader(str(pdf_file))

        for page_number, page in enumerate(reader.pages):

            try:

                page_text = page.extract_text()

                if page_text:
                    text += (
                        "\n[PAGE_{}]\n{}".format(
                            page_number + 1,
                            page_text
                        )
                    )

            except Exception as e:
                logger.warning(
                    "PDF page error: %s",
                    e
                )

    except Exception as e:

        logger.error(
            "PDF reading error: %s",
            e
        )

    return text


def split_text(text, chunk_size=800, overlap=100):

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    chunks = []
    current = ""

    for sentence in sentences:

        if (
            len(current) + len(sentence)
            > chunk_size
            and current
        ):

            chunks.append(
                current.strip()
            )

            words = current.split()

            overlap_words = words[
                -(overlap // 5):
            ]

            current = (
                " ".join(overlap_words)
                + " "
                + sentence
            )

        else:

            current += " " + sentence

    if current.strip():
        chunks.append(
            current.strip()
        )

    return chunks


def load_knowledge():

    documents = []

    pdf_files = list(
        KNOWLEDGE_DIR.glob("*.pdf")
    )

    for pdf_file in pdf_files:

        text = read_pdf(pdf_file)

        chunks = split_text(text)

        for index, chunk in enumerate(chunks):

            if chunk.strip():

                documents.append({
                    "text": chunk.strip(),
                    "source": pdf_file.name,
                    "chunk_id": index,
                    "timestamp": get_timestamp()
                })

    return documents


def build_index():

    documents = load_knowledge()

    if not documents:
        return [], None, None

    texts = [
        document["text"]
        for document in documents
    ]

    try:

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=50000,
            stop_words=(
                "english"
                if st.session_state.language == "English"
                else None
            )
        )

        matrix = vectorizer.fit_transform(
            texts
        )

        return (
            documents,
            vectorizer,
            matrix
        )

    except Exception as e:

        logger.error(
            "Index error: %s",
            e
        )

        return [], None, None


if not st.session_state.documents:

    (
        st.session_state.documents,
        st.session_state.vectorizer,
        st.session_state.matrix
    ) = build_index()


# ============================================================
# SEARCH
# ============================================================

def search_knowledge(question):

    if (
        not st.session_state.documents
        or st.session_state.vectorizer is None
        or st.session_state.matrix is None
    ):
        return []

    try:

        question_vector = (
            st.session_state.vectorizer.transform(
                [question]
            )
        )

        scores = cosine_similarity(
            question_vector,
            st.session_state.matrix
        )[0]

        ranked = scores.argsort()[::-1]

        results = []

        minimum = max(
            KB_THRESHOLD,
            MIN_SIMILARITY_SCORE
        )

        for index in ranked[:10]:

            score = float(
                scores[index]
            )

            if score >= minimum:

                document = (
                    st.session_state.documents[index]
                )

                results.append({
                    "text": document["text"],
                    "source": (
                        "Knowledge Base: {}".format(
                            document["source"]
                        )
                    ),
                    "score": score
                })

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:5]

    except Exception as e:

        logger.error(
            "Knowledge search error: %s",
            e
        )

        return []


def search_wikipedia(question):

    language = WIKI_LANG_MAP.get(
        st.session_state.language,
        "en"
    )

    wikipedia.set_lang(language)

    results = []

    try:

        titles = wikipedia.search(
            question,
            results=WIKI_RESULTS
        )

    except Exception as e:

        logger.error(
            "Wikipedia search error: %s",
            e
        )

        return results

    for title in titles:

        try:

            summary = wikipedia.summary(
                title,
                sentences=WIKI_SENTENCES,
                auto_suggest=False
            )

            if summary.strip():

                results.append({
                    "text": summary,
                    "source": "Wikipedia: {}".format(
                        title
                    ),
                    "score": None,
                    "url": (
                        "https://{}.wikipedia.org/wiki/{}"
                        .format(
                            language,
                            title.replace(" ", "_")
                        )
                    )
                })

        except wikipedia.exceptions.DisambiguationError as e:

            if not e.options:
                continue

            try:

                option = e.options[0]

                summary = wikipedia.summary(
                    option,
                    sentences=WIKI_SENTENCES,
                    auto_suggest=False
                )

                if summary.strip():

                    results.append({
                        "text": summary,
                        "source": "Wikipedia: {}".format(
                            option
                        ),
                        "score": None
                    })

            except Exception:
                continue

        except Exception:
            continue

    return results


def search_web(question):

    results = []

    if DDGS is None:
        logger.warning(
            "ddgs package is not installed."
        )
        return results

    try:

        with DDGS() as ddgs:

            hits = ddgs.text(
                question,
                max_results=WEB_RESULTS
            )

            for hit in hits:

                title = (
                    hit.get("title", "")
                    .strip()
                )

                body = (
                    hit.get("body", "")
                    .strip()
                )

                url = (
                    hit.get("href")
                    or hit.get("link")
                    or ""
                )

                if not body or len(body) < 20:
                    continue

                body = re.sub(
                    r"\s+",
                    " ",
                    body
                ).strip()

                results.append({
                    "text": body,
                    "source": (
                        "Web: {} ({})".format(
                            title,
                            url
                        )
                    ),
                    "score": None,
                    "url": url
                })

    except Exception as e:

        logger.error(
            "Web search error: %s",
            e
        )

    return results


# ============================================================
# SOURCE BADGES
# ============================================================

BADGE_COLORS = {
    "Knowledge Base": "#0f9d58",
    "Wikipedia": "#4285f4",
    "Web": "#f4b400",
    "Image Analysis": "#9c27b0"
}


def render_source_badges(
    sources,
    placeholder=None
):

    if not sources:
        return

    html = (
        '<div style="margin-top:10px;">'
    )

    for source in sources:

        color = BADGE_COLORS.get(
            source,
            "#757575"
        )

        html += (
            '<span class="source-badge" '
            'style="background:{0}22;'
            'color:{0};'
            'border:1px solid {0}55;">'
            '{1}</span>'
        ).format(
            color,
            source
        )

    html += "</div>"

    if placeholder:
        placeholder.markdown(
            html,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            html,
            unsafe_allow_html=True
        )


# ============================================================
# NORMAL AI MESSAGES
# ============================================================

def build_answer_messages(
    question,
    search_results,
    history,
    user_name=None,
    user_age=None
):

    recent_history = history[
        -MAX_CONTEXT_MESSAGES:
    ]

    context_parts = []

    for item in search_results:

        context_parts.append(
            "Source: {}\n{}".format(
                item["source"],
                item["text"]
            )
        )

    context = "\n\n".join(
        context_parts
    )

    if st.session_state.language == "العربية":
        language_instruction = (
            "Answer in Arabic."
        )
    else:
        language_instruction = (
            "Answer in English."
        )

    user_info = ""

    if user_name:

        user_info = (
            "You are talking to {}, age {}."
            .format(
                user_name,
                user_age
            )
        )

    custom = (
        st.session_state
        .custom_instructions
        .strip()
    )

    system_prompt = """
You are a Professional Fitness AI Trainer.

{language}

{user_info}

Answer using the CONTEXT below.

Prefer Knowledge Base information when available.

Do not invent information that is not supported
by the context.

If the answer cannot be found in the context,
say that you could not find the answer.

Be professional, concise, helpful, and clear.

Focus on fitness, nutrition, exercise, and general
wellness guidance.

CUSTOM ADMIN INSTRUCTIONS:
{custom}

CONTEXT:
{context}

END CONTEXT.
""".format(
        language=language_instruction,
        user_info=user_info,
        custom=custom,
        context=context
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt.strip()
        }
    ]

    for message in recent_history:

        content = message.get(
            "content"
        )

        if isinstance(content, str):

            messages.append({
                "role": message["role"],
                "content": content
            })

    messages.append({
        "role": "user",
        "content": question
    })

    return messages


# ============================================================
# VISION MESSAGES
# ============================================================

def build_vision_messages(
    question,
    image_b64,
    mime_type,
    history,
    user_name=None,
    user_age=None
):

    recent_history = history[
        -MAX_CONTEXT_MESSAGES:
    ]

    if st.session_state.language == "العربية":
        language_instruction = (
            "Answer in Arabic."
        )
    else:
        language_instruction = (
            "Answer in English."
        )

    user_info = ""

    if user_name:

        user_info = (
            "You are talking to {}, age {}."
            .format(
                user_name,
                user_age
            )
        )

    custom = (
        st.session_state
        .custom_instructions
        .strip()
    )

    system_prompt = """
You are a Professional Fitness AI Trainer.

{language}

{user_info}

The user has attached an image.

Carefully analyze the image and answer the
user's question about it.

Only describe what can reasonably be observed
from the image. Do not claim certainty about
things that cannot be determined visually.

Be professional, concise, and helpful.

CUSTOM ADMIN INSTRUCTIONS:
{custom}
""".format(
        language=language_instruction,
        user_info=user_info,
        custom=custom
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt.strip()
        }
    ]

    for message in recent_history:

        content = message.get(
            "content"
        )

        if isinstance(content, str):

            messages.append({
                "role": message["role"],
                "content": content
            })

    image_data_url = (
        "data:{};base64,{}"
        .format(
            mime_type,
            image_b64
        )
    )

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": question or "Analyze this image."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data_url
                }
            }
        ]
    })

    return messages


# ============================================================
# GROQ STREAM
# ============================================================

def stream_groq(messages, model):

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1500,
        stream=True
    )

    for chunk in stream:

        if not chunk.choices:
            continue

        delta = (
            chunk.choices[0]
            .delta
            .content
        )

        if delta:
            yield delta


# ============================================================
# GENERATE RESPONSE
# ============================================================

def generate_response(
    chat,
    question,
    image_b64=None,
    mime_type=None
):

    history = chat["messages"][:-1]

    with st.chat_message("assistant"):

        answer_placeholder = st.empty()
        sources_placeholder = st.empty()

        # ====================================================
        # IMAGE RESPONSE
        # ====================================================

        if image_b64:

            with st.status(
                T["thinking"],
                expanded=False
            ):

                messages = build_vision_messages(
                    question,
                    image_b64,
                    mime_type,
                    history,
                    user_name=chat.get(
                        "user_name"
                    ),
                    user_age=chat.get(
                        "user_age"
                    )
                )

            try:

                answer = ""

                for chunk in stream_groq(
                    messages,
                    VISION_MODEL
                ):

                    answer += chunk

                    answer_placeholder.markdown(
                        answer
                    )

                sources = [
                    "Image Analysis"
                ]

            except Exception as e:

                logger.exception(
                    "Vision error"
                )

                answer = T[
                    "error_occurred"
                ].format(
                    error=str(e)
                )

                answer_placeholder.error(
                    answer
                )

                sources = []

        # ====================================================
        # TEXT RESPONSE
        # ====================================================

        else:

            with st.status(
                T["thinking"],
                expanded=False
            ):

                kb_results = search_knowledge(
                    question
                )

                wiki_results = search_wikipedia(
                    question
                )

                web_results = search_web(
                    question
                )

            combined_results = (
                kb_results
                + wiki_results
                + web_results
            )

            if not combined_results:

                answer = T["outside"]

                answer_placeholder.write(
                    answer
                )

                sources = []

            else:

                messages = build_answer_messages(
                    question,
                    combined_results,
                    history,
                    user_name=chat.get(
                        "user_name"
                    ),
                    user_age=chat.get(
                        "user_age"
                    )
                )

                try:

                    answer = ""

                    for chunk in stream_groq(
                        messages,
                        MODEL
                    ):

                        answer += chunk

                        answer_placeholder.markdown(
                            answer
                        )

                    sources = sorted(
                        set(
                            item["source"]
                            .split(":")[0]
                            .strip()
                            for item in combined_results
                        )
                    )

                except Exception as e:

                    logger.exception(
                        "Text generation error"
                    )

                    answer = T[
                        "error_occurred"
                    ].format(
                        error=str(e)
                    )

                    answer_placeholder.error(
                        answer
                    )

                    sources = []

        if sources:
            render_source_badges(
                sources,
                sources_placeholder
            )

    chat["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "rating": None,
        "timestamp": get_timestamp()
    })

    chat["last_updated"] = get_timestamp()


# ============================================================
# CHAT MANAGEMENT
# ============================================================

def new_chat():

    chat = make_new_chat_dict()

    st.session_state.conversations[
        chat["id"]
    ] = chat

    st.session_state.current_chat = (
        chat["id"]
    )


def delete_chat():

    current = (
        st.session_state.current_chat
    )

    if current in st.session_state.conversations:

        chat = st.session_state.conversations.pop(
            current
        )

        try:

            chat_file = (
                CHAT_HISTORY_DIR
                / "chat_{}.json".format(
                    current
                )
            )

            with open(
                chat_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    chat,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

        except Exception as e:

            logger.error(
                "Chat history error: %s",
                e
            )

    if not st.session_state.conversations:

        new_chat()

    else:

        st.session_state.current_chat = (
            list(
                st.session_state.conversations.keys()
            )[0]
        )


def export_chat(chat_id):

    chat = st.session_state.conversations.get(
        chat_id
    )

    if chat:

        return json.dumps(
            chat,
            ensure_ascii=False,
            indent=2
        )

    return None


# ============================================================
# SUMMARY
# ============================================================

def create_summary(messages):

    if not messages:
        return "No conversation history."

    recent = messages[-20:]

    conversation = "\n".join(
        "{}: {}".format(
            m["role"],
            m.get("content", "")
        )
        for m in recent
    )

    prompt = """
Create a professional fitness support conversation summary.

Use this format:

FITNESS CONVERSATION SUMMARY

User Goal:
Key Information:
Advice Given:
Current Status:
Next Steps:
Human Support Needed:

Conversation:
{}
""".format(
        conversation
    )

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional "
                        "fitness conversation summarizer."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=800
        )

        return (
            response.choices[0]
            .message
            .content
        )

    except Exception as e:

        logger.error(
            "Summary error: %s",
            e
        )

        return "Error generating summary."


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="agent-header">
            <h1>💪 Fitness AI Trainer</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    selected_language = st.selectbox(
        T["language"],
        ["English", "العربية"],
        index=(
            0
            if st.session_state.language == "English"
            else 1
        )
    )

    if (
        selected_language
        != st.session_state.language
    ):

        st.session_state.language = (
            selected_language
        )

        st.rerun()

    st.divider()

    if st.button(
        T["new"],
        use_container_width=True
    ):

        new_chat()
        st.rerun()

    if st.button(
        T["delete"],
        use_container_width=True
    ):

        delete_chat()
        st.rerun()

    st.divider()

    st.subheader(
        T["history"]
    )

    sorted_chats = sorted(
        st.session_state.conversations.items(),
        key=lambda item:
            item[1].get(
                "last_updated",
                item[1].get(
                    "created_at",
                    ""
                )
            ),
        reverse=True
    )

    for chat_id, chat_item in sorted_chats:

        title = chat_item.get(
            "title",
            "Chat"
        )

        if title == "New Chat":
            title = "Chat {}".format(
                chat_id
            )

        if st.button(
            "💬 {}".format(
                title[:25]
            ),
            key="history_{}".format(
                chat_id
            ),
            use_container_width=True
        ):

            st.session_state.current_chat = (
                chat_id
            )

            st.rerun()

    st.divider()

    st.subheader(
        T["knowledge"]
    )

    pdf_files = list(
        KNOWLEDGE_DIR.glob("*.pdf")
    )

    if pdf_files:

        st.success(
            "{} PDF(s) found".format(
                len(pdf_files)
            )
        )

        for pdf in pdf_files:
            st.write(
                "📄 {}".format(
                    pdf.name
                )
            )

        st.write(
            "Knowledge chunks: {}".format(
                len(
                    st.session_state.documents
                )
            )
        )

    else:

        st.warning(
            T["empty"]
        )

    if st.button(
        T["reload"],
        use_container_width=True
    ):

        (
            st.session_state.documents,
            st.session_state.vectorizer,
            st.session_state.matrix
        ) = build_index()

        st.success(
            "Knowledge Base reloaded."
        )

        st.rerun()

    st.divider()

    st.subheader(
        "🌐 Internet Search"
    )

    st.success(
        "Wikipedia search: enabled"
    )

    if DDGS:
        st.success(
            "Web search: enabled"
        )
    else:
        st.warning(
            "Web search package unavailable"
        )

    st.divider()

    st.subheader(
        T["instructions"]
    )

    instructions_input = st.text_area(
        "Instructions",
        value=st.session_state.custom_instructions,
        placeholder=T["instructions_placeholder"],
        height=150,
        label_visibility="collapsed"
    )

    if st.button(
        T["save_instructions"],
        use_container_width=True
    ):

        st.session_state.custom_instructions = (
            instructions_input
        )

        try:

            with open(
                INSTRUCTIONS_FILE,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    instructions_input
                )

            st.success(
                T["instructions_saved"]
            )

        except Exception as e:

            st.error(
                str(e)
            )

    st.divider()

    current_chat = st.session_state.conversations[
        st.session_state.current_chat
    ]

    if current_chat.get("user_name"):

        st.caption(
            T["profile_signed_in"].format(
                name=current_chat["user_name"],
                age=current_chat["user_age"]
            )
        )

        if st.button(
            T["profile_reset"],
            use_container_width=True
        ):

            clear_user_profile()

            current_chat["user_name"] = None
            current_chat["user_age"] = None
            current_chat["onboarded"] = False

            st.rerun()

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            T["human"],
            use_container_width=True
        ):

            current_chat["escalated"] = True

            st.info(
                T["escalation_requested"]
            )

    with col2:

        if st.button(
            T["export_chat"],
            use_container_width=True
        ):

            exported = export_chat(
                st.session_state.current_chat
            )

            if exported:

                st.download_button(
                    "📥 Download",
                    data=exported,
                    file_name="chat.json",
                    mime="application/json"
                )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="agent-header">
        <h1>💪 Fitness AI Trainer</h1>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CURRENT CHAT
# ============================================================

chat = st.session_state.conversations[
    st.session_state.current_chat
]


# ============================================================
# ONBOARDING
# ============================================================

if not chat["onboarded"]:

    st.subheader(
        T["onboard_title"]
    )

    onboard_name = st.text_input(
        T["onboard_name"]
    )

    onboard_age = st.number_input(
        T["onboard_age"],
        min_value=1,
        max_value=120,
        step=1
    )

    if st.button(
        T["onboard_button"],
        use_container_width=True
    ):

        if onboard_name.strip():

            chat["user_name"] = (
                onboard_name.strip()
            )

            chat["user_age"] = int(
                onboard_age
            )

            chat["onboarded"] = True

            save_user_profile(
                chat["user_name"],
                chat["user_age"]
            )

            greeting = T[
                "onboard_greeting"
            ].format(
                name=chat["user_name"],
                age=chat["user_age"]
            )

            chat["messages"].append({
                "role": "assistant",
                "content": greeting,
                "timestamp": get_timestamp()
            })

            st.rerun()

    st.stop()


# ============================================================
# MESSAGE DISPLAY
# ============================================================

if not chat["messages"]:

    st.info(
        T["welcome"]
    )


for msg_index, message in enumerate(
    chat["messages"]
):

    edit_key = (
        "editing_{}_{}".format(
            st.session_state.current_chat,
            msg_index
        )
    )

    with st.chat_message(
        message["role"]
    ):

        # Image
        if message.get("image_b64"):

            try:

                image_bytes = base64.b64decode(
                    message["image_b64"]
                )

                st.image(
                    image_bytes,
                    width=300
                )

            except Exception:
                pass

        # Message
        if message["role"] == "user":

            st.markdown(
                "<div class='user-message'>{}</div>"
                .format(
                    message.get(
                        "content",
                        ""
                    )
                ),
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                "<div class='assistant-message'>{}</div>"
                .format(
                    message.get(
                        "content",
                        ""
                    )
                ),
                unsafe_allow_html=True
            )

        # Sources
        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            render_source_badges(
                message["sources"]
            )

        # Actions
        if not st.session_state.get(
            edit_key,
            False
        ):

            action_cols = st.columns(
                [1, 1, 1, 1, 1, 6]
            )

            with action_cols[0]:

                if message["role"] == "user":

                    if st.button(
                        "✏️",
                        key="edit_{}_{}".format(
                            st.session_state.current_chat,
                            msg_index
                        )
                    ):

                        st.session_state[
                            edit_key
                        ] = True

                        st.rerun()

            with action_cols[1]:

                if st.button(
                    "🗑️",
                    key="del_{}_{}".format(
                        st.session_state.current_chat,
                        msg_index
                    )
                ):

                    chat["messages"].pop(
                        msg_index
                    )

                    st.rerun()

            if message["role"] == "assistant":

                rating = message.get(
                    "rating"
                )

                with action_cols[2]:

                    if st.button(
                        "👍"
                        if rating != "up"
                        else "✅👍",
                        key="up_{}_{}".format(
                            st.session_state.current_chat,
                            msg_index
                        )
                    ):

                        message["rating"] = "up"

                        st.rerun()

                with action_cols[3]:

                    if st.button(
                        "👎"
                        if rating != "down"
                        else "✅👎",
                        key="down_{}_{}".format(
                            st.session_state.current_chat,
                            msg_index
                        )
                    ):

                        message["rating"] = "down"

                        st.rerun()


# ============================================================
# CHAT INPUT + IMAGE UPLOAD
# ============================================================

prompt = st.chat_input(
    "Ask your fitness question or attach an image...",
    accept_file=True,
    file_type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],
    max_upload_size=20
)


if prompt:

    question = (
        prompt.text.strip()
        if prompt.text
        else ""
    )

    uploaded_files = (
        prompt.files
        if prompt.files
        else []
    )

    uploaded_file = (
        uploaded_files[0]
        if uploaded_files
        else None
    )

    # --------------------------------------------------------
    # Image-only message
    # --------------------------------------------------------

    if uploaded_file is not None:

        image_bytes = uploaded_file.getvalue()

        if len(image_bytes) > MAX_IMAGE_SIZE:

            st.error(
                "Image is too large. Maximum size is 20 MB."
            )

            st.stop()

        mime_type = (
            uploaded_file.type
            or "image/jpeg"
        )

        if mime_type not in SUPPORTED_IMAGE_TYPES:

            st.error(
                "Please upload JPG, PNG, or WEBP."
            )

            st.stop()

        image_b64 = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        if not question:

            question = (
                "Please analyze this image "
                "and tell me what you can observe."
            )

        # Store user message
        chat["messages"].append({
            "role": "user",
            "content": question,
            "image_b64": image_b64,
            "mime_type": mime_type,
            "timestamp": get_timestamp()
        })

        if chat["title"] == "New Chat":

            chat["title"] = (
                question[:40]
                + (
                    "..."
                    if len(question) > 40
                    else ""
                )
            )

        # Display image immediately
        with st.chat_message("user"):

            st.image(
                image_bytes,
                width=300
            )

            st.markdown(
                "<div class='user-message'>{}</div>"
                .format(question),
                unsafe_allow_html=True
            )

        # Generate vision response
        try:

            generate_response(
                chat,
                question,
                image_b64=image_b64,
                mime_type=mime_type
            )

        except Exception as e:

            logger.exception(
                "Image response error"
            )

            st.error(
                "An error occurred while analyzing the image."
            )

        st.rerun()

    # --------------------------------------------------------
    # Text-only message
    # --------------------------------------------------------

    elif question:

        if contains_sensitive_data(question):

            chat["messages"].append({
                "role": "user",
                "content": question,
                "timestamp": get_timestamp()
            })

            chat["messages"].append({
                "role": "assistant",
                "content": T["sensitive"],
                "timestamp": get_timestamp()
            })

            st.rerun()

        chat["messages"].append({
            "role": "user",
            "content": question,
            "timestamp": get_timestamp()
        })

        if chat["title"] == "New Chat":

            chat["title"] = (
                question[:40]
                + (
                    "..."
                    if len(question) > 40
                    else ""
                )
            )

        with st.chat_message("user"):

            st.markdown(
                "<div class='user-message'>{}</div>"
                .format(question),
                unsafe_allow_html=True
            )

        try:

            generate_response(
                chat,
                question
            )

        except Exception:

            logger.exception(
                "Response generation error"
            )

            st.error(
                "An error occurred while generating the response."
            )

        st.rerun()


# ============================================================
# SUMMARY
# ============================================================

if chat["messages"]:

    st.divider()

    with st.expander(
        T["summary"]
    ):

        if st.button(
            "Generate Summary"
        ):

            with st.spinner(
                T["summary_generating"]
            ):

                summary = create_summary(
                    chat["messages"]
                )

                st.subheader(
                    T["summary_generated"]
                )

                st.markdown(
                    summary
                )

                chat["summary"] = summary


# ============================================================
# ESCALATION
# ============================================================

if chat["escalated"]:

    st.warning(
        T["escalation_requested"]
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Powered by Groq AI | Fitness AI Trainer v1.0"
)
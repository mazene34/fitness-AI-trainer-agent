import os
import re
import json
import hashlib
import base64
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

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
    from duckduckgo_search import DDGS

# ============================================================
# CONFIGURATION & LOGGING
# ============================================================

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    st.error("GROQ_API_KEY is missing from your .env file.")
    st.stop()

client = Groq(api_key=API_KEY)

MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "qwen/qwen3.6-27b"

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

INSTRUCTIONS_FILE = BASE_DIR / "custom_instructions.txt"
USER_PROFILE_FILE = BASE_DIR / "user_profile.json"
CHAT_HISTORY_DIR = BASE_DIR / "chat_history"
CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# Configuration constants
KB_THRESHOLD = 0.08
WIKI_RESULTS = 2
WIKI_SENTENCES = 5
WEB_RESULTS = 3

WIKI_LANG_MAP = {
    "English": "en",
    "العربية": "ar"
}

# Advanced configuration
MAX_CONTEXT_MESSAGES = 10
MAX_TOKENS_PER_CHUNK = 800
MIN_SIMILARITY_SCORE = 0.1
RESPONSE_TIMEOUT = 30


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_timestamp():
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()


def sanitize_filename(filename):
    """Sanitize filename to prevent security issues"""
    return re.sub(r'[^\w\-_\.]', '_', filename)


# ============================================================
# RETURNING USER PROFILE
# ============================================================

def load_user_profile():
    """Load user profile with enhanced error handling"""
    if USER_PROFILE_FILE.exists():
        try:
            with open(USER_PROFILE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Validate required fields
                if isinstance(data, dict) and 'name' in data and 'age' in data:
                    return data
                else:
                    logger.warning("Invalid user profile format")
        except Exception as e:
            logger.error("Error loading user profile:  {0}".format(e))
    return {"name": None, "age": None, "preferences": {}}


def save_user_profile(name, age, preferences=None):
    """Save user profile with enhanced data structure"""
    if preferences is None:
        preferences = {}

    profile_data = {
        "name": name,
        "age": age,
        "preferences": preferences,
        "last_updated": get_timestamp()
    }

    try:
        with open(USER_PROFILE_FILE, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, ensure_ascii=False, indent=2)
        logger.info("User profile saved for {0}".format(name))
    except Exception as e:
        logger.error("Error saving user profile:  {0}".format(e))


def clear_user_profile():
    """Clear user profile with proper error handling"""
    try:
        if USER_PROFILE_FILE.exists():
            USER_PROFILE_FILE.unlink()
            logger.info("User profile cleared")
    except Exception as e:
        logger.error("Error clearing user profile:  {0}".format(e))


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fitness AI Trainer",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_css():
    """Enhanced CSS with professional styling"""
    css_content = """
    /* Professional header styling */
    .agent-header {
        background:  linear-gradient(135deg,  #667eea 0%,  #764ba2 100%);
        padding:  20px;
        border-radius:  10px;
        text-align:  center;
        color:  white;
        margin-bottom:  20px;
        box-shadow:  0 4px 6px rgba(0,  0,  0,  0.1);
    }

    .agent-header h1 {
        margin:  0;
        font-size:  2.5rem;
        font-weight:  700;
    }

    /* Enhanced source badges */
    .source-badge {
        display:  inline-block;
        padding:  4px 12px;
        border-radius:  20px;
        font-size:  0.8rem;
        font-weight:  500;
        margin:  2px;
        transition:  all 0.3s ease;
    }

    .source-badge: hover {
        transform:  translateY(-2px);
        box-shadow:  0 2px 8px rgba(0,  0,  0,  0.2);
    }

    /* Message styling */
    .user-message {
        background-color:  #e3f2fd;
        border-radius:  15px;
        padding:  15px;
        margin-bottom:  15px;
    }

    .assistant-message {
        background-color:  #f5f5f5;
        border-radius:  15px;
        padding:  15px;
        margin-bottom:  15px;
    }

    /* Action buttons */
    .action-button {
        margin:  2px;
        padding:  5px 10px;
        border-radius:  5px;
        border:  none;
        cursor:  pointer;
        transition:  all 0.2s;
    }

    .action-button: hover {
        transform:  scale(1.1);
    }

    /* Status indicators */
    .status-indicator {
        padding:  10px;
        border-radius:  5px;
        margin:  5px 0;
    }

    /* Custom chat input styling */
    [data-testid="stChatInput"] {
        background-color: #000000 !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        background-color: #000000 !important;
        color: #ffffff !important;
        caret-color: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder {
        color: #bbbbbb !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] label {
        color: #ffffff !important;
    }

    /* Some Streamlit versions wrap the textarea in an extra div */
    [data-testid="stChatInput"] > div {
        background-color: #000000 !important;
    }

    /* Send/upload icon buttons inside the chat bar */
    [data-testid="stChatInput"] button svg {
        fill: #ffffff !important;
    }
    """
    st.markdown("<style>{0}</style>".format(css_content), unsafe_allow_html=True)


load_css()

# ============================================================
# LANGUAGE CONFIGURATION
# ============================================================

LANGUAGE = {
    "English": {
        "welcome": "Hello! I'm your Fitness AI Trainer. Ask me anything about fitness, nutrition, or workout plans.",
        "outside": "Sorry,  I couldn't find an answer in the Knowledge Base,  Wikipedia,  or the web.",
        "empty": "Your Knowledge Base is empty. Please put a PDF inside the knowledge_base folder. I can still try to answer using Wikipedia.",
        "sensitive": "For security reasons,  please do not send passwords,  API keys,  OTP codes,  or other sensitive information.",
        "new": "🆕 New Chat",
        "delete": "🗑️ Delete Current Chat",
        "history": "💬 Chat History",
        "knowledge": "📚 Knowledge Base",
        "language": "🌍 Language",
        "human": "👨‍💼 Human Support",
        "summary": "📝 Conversation Summary",
        "reload": "🔄 Reload Knowledge Base",
        "upload_image": "📷 Upload an image (optional)",
        "instructions": "🧭 Custom Instructions",
        "instructions_placeholder": "e.g. Always be motivating. Focus on safe exercise practices. If asked about medical conditions,  redirect to a healthcare professional.",
        "save_instructions": "💾 Save Instructions",
        "instructions_saved": "Instructions saved.",
        "onboard_title": "👋 Before we start...",
        "onboard_name": "What's your name?",
        "onboard_age": "What's your age?",
        "onboard_button": "Start Chat",
        "onboard_greeting": "Hello {name}! Since you're {age},  I'll keep that in mind. How can I help with your fitness journey today?",
        "onboard_greeting_returning": "Welcome back,  {name}! Ready for another fitness session?",
        "profile_signed_in": "👤 Signed in as {name} ({age})",
        "profile_reset": "🔄 Not you? Reset profile",
        "thinking": "🤖 Thinking...",
        "searching_kb": "🔍 Searching the Knowledge Base...",
        "searching_wiki": "🌐 Searching Wikipedia...",
        "searching_web": "🌍 Searching the web...",
        "generating_answer": "✍️ Generating answer...",
        "sources_found": "✅ Found {count} relevant source(s).",
        "no_sources": "❌ No relevant sources found.",
        "error_occurred": "⚠️ An error occurred:  {error}",
        "summary_generating": "Creating conversation summary...",
        "summary_generated": "Conversation Summary",
        "feedback_thanks": "Thank you for your feedback!",
        "escalation_requested": "👨‍💼 Human support has been requested.",
        "export_chat": "📤 Export Chat",
        "import_chat": "📥 Import Chat"
    },

    "العربية": {
        "welcome": "مرحباً! أنا مدرب اللياقة البدنية الذكي. يمكنك سؤالي عن أي شيء يتعلق باللياقة أو التغذية أو خطط التمارين.",
        "outside": "عذراً، لم أتمكن من إيجاد إجابة في قاعدة المعرفة أو ويكيبيديا أو الويب.",
        "empty": "قاعدة المعرفة فارغة. يرجى وضع ملف PDF داخل مجلد knowledge_base. سأحاول الإجابة عبر ويكيبيديا في هذه الأثناء.",
        "sensitive": "لأسباب أمنية، يرجى عدم إرسال كلمات المرور أو مفاتيح API أو رموز OTP أو أي معلومات سرية.",
        "new": "🆕 محادثة جديدة",
        "delete": "🗑️ حذف المحادثة الحالية",
        "history": "💬 سجل المحادثات",
        "knowledge": "📚 قاعدة المعرفة",
        "language": "🌍 اللغة",
        "human": "👨‍💼 الدعم البشري",
        "summary": "📝 ملخص المحادثة",
        "reload": "🔄 إعادة تحميل قاعدة المعرفة",
        "upload_image": "📷 رفع صورة (اختياري)",
        "instructions": "🧭 تعليمات مخصصة",
        "instructions_placeholder": "مثال:  كن محفزاً دائماً. ركز على ممارسات التمرين الآمنة. إذا سُئلت عن حالات طبية، وجّه العميل إلى مختص صحي.",
        "save_instructions": "💾 حفظ التعليمات",
        "instructions_saved": "تم حفظ التعليمات.",
        "onboard_title": "👋 قبل أن نبدأ...",
        "onboard_name": "ما اسمك؟",
        "onboard_age": "كم عمرك؟",
        "onboard_button": "ابدأ المحادثة",
        "onboard_greeting": "مرحباً {name}! بما أن عمرك {age}، سأضع ذلك في اعتباري. كيف يمكنني مساعدتك في رحلة لياقتك اليوم؟",
        "onboard_greeting_returning": "مرحباً بعودتك، {name}! هل أنت مستعد لجلسة لياقة أخرى؟",
        "profile_signed_in": "👤 مسجل الدخول باسم {name} ({age})",
        "profile_reset": "🔄 لست أنت؟ إعادة تعيين الملف الشخصي",
        "thinking": "🤖 أفكر...",
        "searching_kb": "🔍 أبحث في قاعدة المعرفة...",
        "searching_wiki": "🌐 أبحث في ويكيبيديا...",
        "searching_web": "🌍 أبحث في الويب...",
        "generating_answer": "✍️ أنشئ الإجابة...",
        "sources_found": "✅ وجدت {count} مصدر متعلق.",
        "no_sources": "❌ لم أجد مصادر متعلقة.",
        "error_occurred": "⚠️ حدث خطأ:  {error}",
        "summary_generating": "أ созда ملخص المحادثة...",
        "summary_generated": "ملخص المحادثة",
        "feedback_thanks": "شكراً لك على التقييم!",
        "escalation_requested": "👨‍💼 تم طلب الدعم البشري.",
        "export_chat": "📤 تصدير المحادثة",
        "import_chat": "📥 استيراد المحادثة"
    }
}

# ============================================================
# SESSION STATE MANAGEMENT
# ============================================================

if "language" not in st.session_state:
    st.session_state.language = "English"


def make_new_chat_dict():
    """Create a new chat dictionary with enhanced structure"""
    profile = load_user_profile()
    has_profile = bool(profile.get("name"))

    chat_dict = {
        "id": hashlib.md5(os.urandom(20)).hexdigest()[: 8],
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
        greeting = LANGUAGE[st.session_state.language]["onboard_greeting_returning"].format(
            name=chat_dict["user_name"]
        )
        chat_dict["messages"].append({"role": "assistant", "content": greeting})

    return chat_dict


if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "current_chat" not in st.session_state:
    chat_id = hashlib.md5(os.urandom(20)).hexdigest()[: 8]
    st.session_state.current_chat = chat_id
    st.session_state.conversations[chat_id] = make_new_chat_dict()

if "documents" not in st.session_state:
    st.session_state.documents = []

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "matrix" not in st.session_state:
    st.session_state.matrix = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "custom_instructions" not in st.session_state:
    if INSTRUCTIONS_FILE.exists():
        try:
            with open(INSTRUCTIONS_FILE, 'r', encoding='utf-8') as f:
                st.session_state.custom_instructions = f.read()
        except:
            st.session_state.custom_instructions = ""
    else:
        st.session_state.custom_instructions = ""

T = LANGUAGE[st.session_state.language]


# ============================================================
# PDF PROCESSING
# ============================================================

def read_pdf(pdf_file):
    """Enhanced PDF reading with better error handling"""
    text = ""
    try:
        reader = PdfReader(str(pdf_file))
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += "\n[PAGE_{0}]\n{1}".format(i + 1, page_text)
            except Exception as page_error:
                logger.warning(
                    "Error extracting text from page {0} of {1}:  {2}".format(i + 1, pdf_file.name, page_error))
                continue
    except Exception as error:
        logger.error("PDF ERROR in {0}:  {1}".format(pdf_file.name, error))
    return text


# ============================================================
# TEXT PROCESSING
# ============================================================

def split_text(text, chunk_size=800, overlap=100):
    """Enhanced text chunking with sentence-aware splitting"""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Split by sentences first for better context preservation
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence would exceed chunk size,  save current chunk
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from previous
            words = current_chunk.split()
            overlap_words = words[-(overlap // 5):] if len(words) > overlap // 5 else []
            current_chunk = " ".join(overlap_words) + " " + sentence
        else:
            current_chunk += " " + sentence

    # Add the last chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ============================================================
# KNOWLEDGE BASE MANAGEMENT
# ============================================================

def load_knowledge():
    """Enhanced knowledge base loading with metadata"""
    documents = []
    pdf_files = list(KNOWLEDGE_DIR.glob("*.pdf"))

    logger.info("Loading knowledge base from {0}".format(KNOWLEDGE_DIR))
    logger.info("Found {0} PDF files".format(len(pdf_files)))

    for pdf_file in pdf_files:
        logger.info("Processing:  {0}".format(pdf_file.name))
        text = read_pdf(pdf_file)
        logger.info("Extracted {0} characters".format(len(text)))

        chunks = split_text(text)
        logger.info("Created {0} chunks".format(len(chunks)))

        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Only add non-empty chunks
                documents.append({
                    "text": chunk.strip(),
                    "source": pdf_file.name,
                    "chunk_id": i,
                    "timestamp": get_timestamp()
                })

    logger.info("Total chunks in knowledge base:  {0}".format(len(documents)))
    return documents


def build_index():
    """Enhanced index building with progress tracking"""
    documents = load_knowledge()

    if not documents:
        logger.warning("No documents found in knowledge base")
        return [], None, None

    texts = [doc["text"] for doc in documents]

    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=50000,
            stop_words='english' if st.session_state.language == "English" else None
        )

        matrix = vectorizer.fit_transform(texts)
        logger.info("Knowledge base index built successfully")
        return documents, vectorizer, matrix

    except Exception as e:
        logger.error("Error building index:  {0}".format(e))
        return [], None, None


# ============================================================
# INITIAL LOADING
# ============================================================

if not st.session_state.documents:
    with st.spinner("Initializing knowledge base..."):
        (
            st.session_state.documents,
            st.session_state.vectorizer,
            st.session_state.matrix
        ) = build_index()


# ============================================================
# SEARCH FUNCTIONS
# ============================================================

def search_knowledge(question):
    """Enhanced knowledge search with scoring improvements"""
    if (
            not st.session_state.documents
            or st.session_state.vectorizer is None
            or st.session_state.matrix is None
    ):
        return []

    try:
        question_vector = st.session_state.vectorizer.transform([question])
        scores = cosine_similarity(question_vector, st.session_state.matrix)[0]
        ranked = scores.argsort()[:: -1]

        results = []
        min_score = max(KB_THRESHOLD, MIN_SIMILARITY_SCORE)

        for index in ranked[: 10]:  # Check more results but filter better
            score = float(scores[index])
            if score >= min_score:
                results.append({
                    "text": st.session_state.documents[index]["text"],
                    "source": "Knowledge Base:  {0}".format(st.session_state.documents[index]['source']),
                    "score": score,
                    "chunk_id": st.session_state.documents[index].get("chunk_id", 0)
                })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[: 5]  # Return top 5

    except Exception as e:
        logger.error("Knowledge search error:  {0}".format(e))
        return []


def search_wikipedia(question):
    """Enhanced Wikipedia search with better error handling"""
    lang = WIKI_LANG_MAP.get(st.session_state.language, "en")
    wikipedia.set_lang(lang)

    results = []

    try:
        titles = wikipedia.search(question, results=WIKI_RESULTS)
        logger.info("Wikipedia search for '{0}' found {1} results".format(question, len(titles)))
    except Exception as error:
        logger.error("WIKIPEDIA SEARCH ERROR:  {0}".format(error))
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
                    "source": "Wikipedia:  {0}".format(title),
                    "score": None,
                    "url": "https: //{0}.wikipedia.org/wiki/{1}".format(lang, title.replace(' ', '_'))
                })

        except wikipedia.exceptions.DisambiguationError as error:
            try:
                option = error.options[0]
                summary = wikipedia.summary(
                    option,
                    sentences=WIKI_SENTENCES,
                    auto_suggest=False
                )

                if summary.strip():
                    results.append({
                        "text": summary,
                        "source": "Wikipedia:  {0}".format(option),
                        "score": None,
                        "url": "https: //{0}.wikipedia.org/wiki/{1}".format(lang, option.replace(' ', '_'))
                    })

            except Exception as inner_error:
                logger.error("WIKIPEDIA DISAMBIGUATION ERROR:  {0}".format(inner_error))

        except Exception as error:
            logger.error("WIKIPEDIA PAGE ERROR ({0}):  {1}".format(title, error))

    return results


def search_web(question):
    """Enhanced web search with result filtering"""
    results = []

    try:
        with DDGS() as ddgs:
            hits = ddgs.text(question, max_results=WEB_RESULTS)

            for hit in hits:
                title = hit.get("title", "").strip()
                body = hit.get("body", "").strip()
                url = hit.get("href") or hit.get("link") or ""

                if not body or len(body) < 20:  # Filter out very short snippets
                    continue

                # Clean up the text
                body = re.sub(r'\s+', ' ', body).strip()

                results.append({
                    "text": body,
                    "source": "Web:  {0} ({1})".format(title, url) if url else "Web:  {0}".format(title),
                    "score": None,
                    "url": url
                })

    except Exception as error:
        logger.error("WEB SEARCH ERROR:  {0}".format(error))

    return results


# ============================================================
# SECURITY FUNCTIONS
# ============================================================

def contains_sensitive_data(text):
    """Enhanced sensitive data detection"""
    patterns = [
        r"sk-[A-Za-z0-9_-]{20, }",
        r"gsk_[A-Za-z0-9_-]{20, }",
        r"password\s*[: =]\s*\S+",
        r"Bearer\s+[A-Za-z0-9._~+/=-]+",
        r"\b(?: otp|verification code)\s*[: =]?\s*\d{4, 8}\b",
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2, }",
        r"(?: \d{1, 3}\.){3}\d{1, 3}",
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    ]

    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


# ============================================================
# MESSAGE BUILDING
# ============================================================

def build_answer_messages(
        question,
        search_results,
        history,
        user_name=None,
        user_age=None
):
    """Enhanced message building with context management"""

    # Limit context to recent messages for better performance
    recent_history = history[-MAX_CONTEXT_MESSAGES:] if len(history) > MAX_CONTEXT_MESSAGES else history

    context_parts = []
    sources_used = []

    for item in search_results:
        source_name = item['source'].split(': ')[0].strip()
        context_parts.append("Source:  {0}\n{1}".format(item['source'], item['text']))
        sources_used.append(source_name)

    context = "\n\n".join(context_parts)

    # Language instruction
    if st.session_state.language == "العربية":
        language_instruction = "\nAnswer in Arabic.\n"
    else:
        language_instruction = "\nAnswer in English.\n"

    # Custom instructions
    custom_instructions = st.session_state.custom_instructions.strip()
    instructions_block = ""
    if custom_instructions:
        instructions_block = """
ADDITIONAL RULES FROM THE ADMIN (follow these strictly,  
they override general tone/behavior but not the context above):  

{0}
""".format(custom_instructions)

    # User information
    user_info_block = ""
    if user_name:
        user_info_block = "\nYou are talking to {0},  age {1}. You may address them by name when it feels natural.\n".format(
            user_name, user_age)

    system_prompt = """
You are a Professional Fitness AI Trainer.
{0}
Answer using the CONTEXT below,  which may contain information
from the internal Knowledge Base,  Wikipedia,  and general web
search results.

Prefer Knowledge Base information over Wikipedia or web results
when they overlap or conflict,  since the Knowledge Base reflects
the company's own product and policies.

When you use Wikipedia or web search information,  make it clear
that it is general/background information rather than company
policy,  and mention which source it came from when useful.

You MUST NOT invent information that isn't supported by the
context below.

If the answer cannot be found in the context,  say:  
"{1}"

{2}

Be professional,  concise,  and helpful. Focus on fitness,  nutrition,  and workout guidance.
{3}
CONTEXT:  

{4}

END CONTEXT.
""".format(user_info_block, T['outside'], language_instruction, instructions_block, context)

    messages = [{"role": "system", "content": system_prompt.strip()}]

    # Add recent conversation history
    for message in recent_history:
        if isinstance(message["content"], str):
            messages.append({"role": message["role"], "content": message["content"]})

    messages.append({"role": "user", "content": question})
    return messages


def build_vision_messages(
        question,
        image_b64,
        mime_type,
        history,
        user_name=None,
        user_age=None
):
    """Enhanced vision message building"""

    recent_history = history[-MAX_CONTEXT_MESSAGES:] if len(history) > MAX_CONTEXT_MESSAGES else history

    if st.session_state.language == "العربية":
        language_instruction = "Answer in Arabic."
    else:
        language_instruction = "Answer in English."

    custom_instructions = st.session_state.custom_instructions.strip()
    instructions_block = ""
    if custom_instructions:
        instructions_block = """
ADDITIONAL RULES FROM THE ADMIN (follow these strictly):  

{0}
""".format(custom_instructions)

    if user_name:
        user_info_block = "You are talking to {0},  age {1}. You may address them by name when it feels natural.\n".format(
            user_name, user_age)
    else:
        user_info_block = ""

    system_prompt = """
You are a Professional Fitness AI Trainer.
{0}
The user has attached an image. Look at it carefully and answer
their question about it. Be professional,  concise,  and helpful.
Focus on fitness-related aspects of the image.

{1}
{2}
""".format(user_info_block, language_instruction, instructions_block)

    messages = [{"role": "system", "content": system_prompt.strip()}]

    # Previous conversation (text only)
    for message in recent_history:
        if isinstance(message["content"], str):
            messages.append({"role": message["role"], "content": message["content"]})

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {
                "type": "image_url",
                "image_url": {"url": "data: {0};base64, {1}".format(mime_type, image_b64)}
            }
        ]
    })

    return messages


# ============================================================
# RESPONSE GENERATION
# ============================================================

def stream_groq(messages, model):
    """Enhanced streaming with error handling and timeout"""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,  # Slightly higher for more natural responses
            max_tokens=1500,
            stream=True
        )

        full_response = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta
                yield delta

        return full_response

    except Exception as e:
        logger.error("Groq streaming error:  {0}".format(e))
        raise


def generate_response(
        chat,
        question,
        image_b64=None,
        mime_type=None
):
    """Enhanced response generation with comprehensive error handling"""

    history = chat["messages"][: -1]  # Exclude the current question

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        sources_placeholder = st.empty()

        if image_b64 is not None:
            # Vision model processing
            with st.status(T["thinking"], expanded=True) as status:
                status.update(label=T["generating_answer"])

                messages = build_vision_messages(
                    question, image_b64, mime_type, history,
                    user_name=chat.get("user_name"),
                    user_age=chat.get("user_age")
                )

                try:
                    answer = ""
                    for chunk in stream_groq(messages, VISION_MODEL):
                        answer += chunk
                        answer_placeholder.write(answer)

                    sources = ["Image Analysis"]
                except Exception as error:
                    error_msg = T["error_occurred"].format(error=str(error))
                    answer_placeholder.error(error_msg)
                    answer = error_msg
                    sources = []

        else:
            # Text-based processing with search
            with st.status(T["thinking"], expanded=True) as status:
                status.update(label=T["searching_kb"])
                kb_results = search_knowledge(question)

                status.update(label=T["searching_wiki"])
                wiki_results = search_wikipedia(question)

                status.update(label=T["searching_web"])
                web_results = search_web(question)

                combined_results = kb_results + wiki_results + web_results

                if combined_results:
                    status.update(
                        label=T["sources_found"].format(count=len(combined_results)),
                        state="complete"
                    )
                else:
                    status.update(
                        label=T["no_sources"],
                        state="error"
                    )

            # Show search statistics
            st.sidebar.write(
                "KB:  {0} | "
                "Wikipedia:  {1} | "
                "Web:  {2}".format(len(kb_results), len(wiki_results), len(web_results))
            )

            if not combined_results:
                answer = T["outside"]
                answer_placeholder.write(answer)
                sources = []
            else:
                messages = build_answer_messages(
                    question, combined_results, history,
                    user_name=chat.get("user_name"),
                    user_age=chat.get("user_age")
                )

                try:
                    answer = ""
                    for chunk in stream_groq(messages, MODEL):
                        answer += chunk
                        answer_placeholder.write(answer)

                    # Extract unique sources
                    sources = sorted(set(
                        item["source"].split(": ")[0].strip()
                        for item in combined_results
                    ))
                except Exception as error:
                    error_msg = T["error_occurred"].format(error=str(error))
                    answer_placeholder.error(error_msg)
                    answer = error_msg
                    sources = []

        # Render source badges if any sources found
        if sources:
            render_source_badges(sources, sources_placeholder)

    # Add the complete response to chat history
    chat["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "rating": None,
        "timestamp": get_timestamp()
    })

    # Update last updated timestamp
    chat["last_updated"] = get_timestamp()


# ============================================================
# CHAT MANAGEMENT
# ============================================================

def new_chat():
    """Create a new chat session"""
    chat_id = hashlib.md5(os.urandom(20)).hexdigest()[: 8]
    st.session_state.conversations[chat_id] = make_new_chat_dict()
    st.session_state.current_chat = chat_id
    logger.info("New chat created:  {0}".format(chat_id))


def delete_chat():
    """Delete current chat session"""
    current = st.session_state.current_chat

    if current in st.session_state.conversations:
        deleted_chat = st.session_state.conversations.pop(current)
        logger.info("Chat deleted:  {0}".format(current))

        # Save chat history to file before deletion
        try:
            chat_file = CHAT_HISTORY_DIR / "chat_{0}.json".format(current)
            with open(chat_file, 'w', encoding='utf-8') as f:
                json.dump(deleted_chat, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Error saving chat history:  {0}".format(e))

    if not st.session_state.conversations:
        new_chat()
    else:
        # Switch to the first available chat
        st.session_state.current_chat = list(st.session_state.conversations.keys())[0]


def export_chat(chat_id):
    """Export chat to JSON string"""
    try:
        if chat_id in st.session_state.conversations:
            chat_data = st.session_state.conversations[chat_id]
            return json.dumps(chat_data, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("Error exporting chat {0}:  {1}".format(chat_id, e))
    return None


def import_chat(chat_data):
    """Import chat from JSON string"""
    try:
        chat_dict = json.loads(chat_data)
        if isinstance(chat_dict, dict) and "id" in chat_dict:
            chat_id = chat_dict["id"]
            st.session_state.conversations[chat_id] = chat_dict
            return chat_id
    except Exception as e:
        logger.error("Error importing chat:  {0}".format(e))
    return None


# ============================================================
# SUMMARY GENERATION
# ============================================================

def create_summary(messages):
    """Enhanced conversation summary generation"""
    if not messages:
        return "No conversation history."

    # Limit to recent messages for summary
    recent_messages = messages[-20:] if len(messages) > 20 else messages
    conversation = "\n".join(["{0}:  {1}".format(m['role'], m['content']) for m in recent_messages])

    prompt = """
Create a professional fitness support conversation summary in the following format: 

**FITNESS CONVERSATION SUMMARY**

**User Goal: ** [Brief description of the main fitness goal]
**Key Information: ** [Important details mentioned]
**Advice Given: ** [What was discussed/recommended]
**Current Status: ** [Resolution state]
**Next Steps: ** [Recommendations if needed]
**Human Support Needed: ** [Yes/No and reason if yes]

---

Conversation: 
{0}
""".format(conversation)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system",
                 "content": "You are a professional fitness conversation summarizer. Create concise,  structured summaries focused on fitness goals."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )

        summary = response.choices[0].message.content
        return summary if summary else "Unable to generate summary."

    except Exception as e:
        logger.error("Summary generation error:  {0}".format(e))
        return "Error generating summary:  {0}".format(str(e))


# ============================================================
# SOURCE BADGES RENDERING
# ============================================================

BADGE_COLORS = {
    "Knowledge Base": "#0f9d58",
    "Wikipedia": "#4285f4",
    "Web": "#f4b400",
    "Image Analysis": "#9c27b0"
}


def render_source_badges(sources, placeholder=None):
    """Enhanced source badges rendering"""
    if not sources:
        return

    badges_html = '<div style="margin-top:  10px;">'
    for source in sources:
        color = BADGE_COLORS.get(source, "#757575")
        badges_html += (
            '<span class="source-badge" style="background:  {0}22; color:  {0}; '
            'border:  1px solid {0}55;">{1}</span>'.format(color, source)
        )
    badges_html += '</div>'

    if placeholder:
        placeholder.markdown(badges_html, unsafe_allow_html=True)
    else:
        st.markdown(badges_html, unsafe_allow_html=True)


# ============================================================
# SIDEBAR INTERFACE
# ============================================================

with st.sidebar:
    # Enhanced header
    st.markdown(
        """
        <div class="agent-header">
            <h1>💪 Fitness AI Trainer</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Language selector
    selected_language = st.selectbox(
        T["language"],
        ["English", "العربية"],
        index=0 if st.session_state.language == "English" else 1
    )

    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.rerun()

    st.divider()

    # Chat management
    if st.button(T["new"], use_container_width=True):
        new_chat()
        st.rerun()

    if st.button(T["delete"], use_container_width=True):
        delete_chat()
        st.rerun()

    st.divider()

    # Chat history
    st.subheader(T["history"])

    # Sort conversations by last updated time
    sorted_chats = sorted(
        st.session_state.conversations.items(),
        key=lambda x: x[1].get("last_updated", x[1].get("created_at", "")),
        reverse=True
    )

    for chat_id, chat in sorted_chats:
        title = chat["title"]
        if title == "New Chat":
            title = "Chat {0}".format(chat_id)

        # Add timestamp if available
        timestamp = ""
        if "last_updated" in chat:
            try:
                dt = datetime.fromisoformat(chat["last_updated"])
                timestamp = " ({0})".format(dt.strftime('%H: %M'))
            except:
                pass

        if st.button("💬 {0}{1}".format(title[: 25], timestamp), key="history_{0}".format(chat_id),
                     use_container_width=True):
            st.session_state.current_chat = chat_id
            st.rerun()

    st.divider()

    # Knowledge Base section
    st.subheader(T["knowledge"])

    pdf_files = list(KNOWLEDGE_DIR.glob("*.pdf"))

    if pdf_files:
        st.success("{0} PDF(s) found".format(len(pdf_files)))
        for pdf in pdf_files:
            st.write("📄 {0}".format(pdf.name))
        st.write("Knowledge chunks:  {0}".format(len(st.session_state.documents)))
    else:
        st.error(T["empty"])

    if st.button(T["reload"], use_container_width=True):
        with st.spinner("Reloading knowledge base..."):
            (
                st.session_state.documents,
                st.session_state.vectorizer,
                st.session_state.matrix
            ) = build_index()
        st.success("Knowledge Base reloaded.")
        st.rerun()

    st.divider()

    # Internet search status
    st.subheader("🌐 Internet Search")
    st.success("Wikipedia search:  enabled")
    st.success("Web search (DuckDuckGo):  enabled")

    st.divider()

    # Custom Instructions
    st.subheader(T["instructions"])

    instructions_input = st.text_area(
        T["instructions"],
        value=st.session_state.custom_instructions,
        placeholder=T["instructions_placeholder"],
        height=150,
        label_visibility="collapsed"
    )

    if st.button(T["save_instructions"], use_container_width=True):
        st.session_state.custom_instructions = instructions_input
        try:
            with open(INSTRUCTIONS_FILE, 'w', encoding='utf-8') as f:
                f.write(instructions_input)
            st.success(T["instructions_saved"])
        except Exception as e:
            st.error("Error saving instructions:  {0}".format(e))

    st.divider()

    # User profile management
    _current_chat_for_profile = st.session_state.conversations[st.session_state.current_chat]

    if _current_chat_for_profile.get("user_name"):
        st.caption(T["profile_signed_in"].format(
            name=_current_chat_for_profile["user_name"],
            age=_current_chat_for_profile["user_age"]
        ))

        if st.button(T["profile_reset"], use_container_width=True):
            clear_user_profile()
            _current_chat_for_profile["user_name"] = None
            _current_chat_for_profile["user_age"] = None
            _current_chat_for_profile["onboarded"] = False
            st.rerun()

    st.divider()

    # Human escalation and export
    col1, col2 = st.columns(2)

    with col1:
        if st.button(T["human"], use_container_width=True):
            chat = st.session_state.conversations[st.session_state.current_chat]
            chat["escalated"] = True
            st.info(T["escalation_requested"])

    with col2:
        if st.button(T["export_chat"], use_container_width=True):
            chat_export = export_chat(st.session_state.current_chat)
            if chat_export:
                st.download_button(
                    label="📥 Download Chat",
                    data=chat_export,
                    file_name="chat_{0}.json".format(st.session_state.current_chat),
                    mime="application/json"
                )

# ============================================================
# MAIN INTERFACE
# ============================================================

st.markdown(
    """
    <div class="agent-header">
        <h1>💪 Fitness AI Trainer</h1>
    </div>
    """,
    unsafe_allow_html=True
)

chat = st.session_state.conversations[st.session_state.current_chat]

# ============================================================
# ONBOARDING
# ============================================================

if not chat["onboarded"]:
    st.subheader(T["onboard_title"])

    onboard_name = st.text_input(T["onboard_name"])
    onboard_age = st.number_input(T["onboard_age"], min_value=1, max_value=120, step=1)

    if st.button(T["onboard_button"], use_container_width=True):
        if onboard_name.strip():
            chat["user_name"] = onboard_name.strip()
            chat["user_age"] = int(onboard_age)
            chat["onboarded"] = True

            # Remember this user for future chats
            save_user_profile(chat["user_name"], chat["user_age"])

            greeting = T["onboard_greeting"].format(
                name=chat["user_name"],
                age=chat["user_age"]
            )

            chat["messages"].append({"role": "assistant", "content": greeting})
            st.rerun()

    st.stop()

# ============================================================
# MESSAGE DISPLAY
# ============================================================

if not chat["messages"]:
    st.info(T["welcome"])

for msg_index, message in enumerate(chat["messages"]):
    edit_key = "editing_{0}_{1}".format(st.session_state.current_chat, msg_index)

    with st.chat_message(message["role"]):
        # Display image if present
        if message.get("image_b64"):
            st.image(base64.b64decode(message["image_b64"]), width=200)

        # Edit mode for user messages
        if message["role"] == "user" and st.session_state.get(edit_key, False):
            new_text = st.text_area(
                "Edit message",
                value=message["content"],
                key="edit_box_{0}_{1}".format(st.session_state.current_chat, msg_index),
                label_visibility="collapsed"
            )

            col_save, col_cancel = st.columns([1, 1])

            with col_save:
                if st.button("💾 Save & Regenerate",
                             key="save_{0}_{1}".format(st.session_state.current_chat, msg_index)):
                    edited_image_b64 = message.get("image_b64")
                    edited_mime_type = message.get("mime_type")

                    message["content"] = new_text

                    # Remove everything after this message
                    chat["messages"] = chat["messages"][: msg_index + 1]

                    st.session_state[edit_key] = False

                    generate_response(
                        chat,
                        new_text,
                        image_b64=edited_image_b64,
                        mime_type=edited_mime_type
                    )

                    st.rerun()

            with col_cancel:
                if st.button("✖ Cancel", key="cancel_{0}_{1}".format(st.session_state.current_chat, msg_index)):
                    st.session_state[edit_key] = False
                    st.rerun()

        else:
            # Regular message display
            if message["role"] == "user":
                st.markdown("<div class='user-message'>{0}</div>".format(message['content']), unsafe_allow_html=True)
            else:
                st.markdown("<div class='assistant-message'>{0}</div>".format(message['content']),
                            unsafe_allow_html=True)

        # Display sources for assistant messages
        if message["role"] == "assistant" and message.get("sources"):
            render_source_badges(message["sources"])

        # Action buttons (only shown when not in edit mode)
        if not st.session_state.get(edit_key, False):
            action_cols = st.columns([1, 1, 1, 1, 1, 6])

            with action_cols[0]:
                if message["role"] == "user":
                    if st.button("✏️", key="edit_{0}_{1}".format(st.session_state.current_chat, msg_index),
                                 help="Edit message"):
                        st.session_state[edit_key] = True
                        st.rerun()

            with action_cols[1]:
                if st.button("🗑️", key="del_{0}_{1}".format(st.session_state.current_chat, msg_index),
                             help="Delete message"):
                    chat["messages"].pop(msg_index)
                    st.rerun()

            if message["role"] == "assistant":
                rating = message.get("rating")

                with action_cols[2]:
                    if st.button(
                            "👍" if rating != "up" else "✅👍",
                            key="up_{0}_{1}".format(st.session_state.current_chat, msg_index),
                            help="Helpful response"
                    ):
                        message["rating"] = "up"
                        st.rerun()

                with action_cols[3]:
                    if st.button(
                            "👎" if rating != "down" else "✅👎",
                            key="down_{0}_{1}".format(st.session_state.current_chat, msg_index),
                            help="Not helpful"
                    ):
                        message["rating"] = "down"
                        st.rerun()

                with action_cols[4]:
                    # Copy button
                    if st.button("📋", key="copy_{0}_{1}".format(st.session_state.current_chat, msg_index),
                                 help="Copy response"):
                        st.write("Response copied to clipboard!")

# ============================================================
# USER INPUT
# ============================================================

prompt = st.chat_input(
    "Ask your fitness question...",
    accept_file=True,
    file_type=["png", "jpg", "jpeg", "webp"]
)

if prompt:
    question = prompt.text
    uploaded_file = prompt.files[0] if prompt.files else None

    if question:
        # Sensitive data check
        if contains_sensitive_data(question):
            chat["messages"].append({"role": "user", "content": question})
            chat["messages"].append({"role": "assistant", "content": T["sensitive"]})
            st.rerun()

        # Process image if uploaded
        image_b64 = None
        mime_type = None

        if uploaded_file is not None:
            try:
                image_bytes = uploaded_file.getvalue()
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                mime_type = uploaded_file.type or "image/png"

                chat["messages"].append({
                    "role": "user",
                    "content": question,
                    "image_b64": image_b64,
                    "mime_type": mime_type,
                    "timestamp": get_timestamp()
                })
            except Exception as e:
                logger.error("Error processing image:  {0}".format(e))
                st.error("Error processing image file")

        else:
            chat["messages"].append({
                "role": "user",
                "content": question,
                "timestamp": get_timestamp()
            })

        # Set chat title if it's a new chat
        if chat["title"] == "New Chat":
            chat["title"] = question[: 40] + ("..." if len(question) > 40 else "")

        # Display user message immediately
        with st.chat_message("user"):
            if uploaded_file is not None:
                st.image(uploaded_file, width=200)
            st.markdown("<div class='user-message'>{0}</div>".format(question), unsafe_allow_html=True)

        # Generate and display response
        try:
            generate_response(chat, question, image_b64=image_b64, mime_type=mime_type)
        except Exception as e:
            logger.error("Error generating response:  {0}".format(e))
            st.error("An error occurred while generating the response. Please try again.")

        st.rerun()

# ============================================================
# CONVERSATION SUMMARY
# ============================================================

if chat["messages"]:
    st.divider()

    with st.expander(T["summary"]):
        if st.button("Generate Summary"):
            with st.spinner(T["summary_generating"]):
                try:
                    summary = create_summary(chat["messages"])
                    st.subheader(T["summary_generated"])
                    st.markdown(summary)

                    # Save summary to chat
                    chat["summary"] = summary
                except Exception as error:
                    st.error(T["error_occurred"].format(error=str(error)))

# ============================================================
# ESCALATION HANDLING
# ============================================================

if chat["escalated"]:
    st.warning(T["escalation_requested"])

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("Powered by Groq AI | Fitness AI Trainer v1.0")
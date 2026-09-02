import os
import re
import json
import hashlib
import base64
import logging
from io import BytesIO
from pathlib import Path
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
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

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


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

PROGRESS_PHOTOS_DIR = BASE_DIR / "progress_photos"
PROGRESS_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)


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

try:
    API_KEY = st.secrets.get("GROQ_API_KEY")
except Exception:
    API_KEY = None

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
# ADMIN PASSWORD CONFIGURATION
# ============================================================

ADMIN_PASSWORD = None

try:
    ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD")
except Exception:
    ADMIN_PASSWORD = None

if not ADMIN_PASSWORD:
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

if not ADMIN_PASSWORD:
    # Sensible default so the app still runs out of the box.
    # Change this via .env / Streamlit Secrets for real deployments.
    ADMIN_PASSWORD = "admin123"


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
MAX_PDF_SIZE = 30 * 1024 * 1024

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

WIKI_LANG_MAP = {"English": "en", "العربية": "ar"}

ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very_active": 1.9
}

STREAK_MILESTONES = [3, 7, 14, 30, 60, 100]

MEASUREMENT_FIELDS = [
    ("chest_cm", "Chest (cm)"),
    ("waist_cm", "Waist (cm)"),
    ("hips_cm", "Hips (cm)"),
    ("arms_cm", "Arms (cm)"),
    ("thighs_cm", "Thighs (cm)")
]

DEFAULT_TRACKING = {
    "weight_log": [],
    "workout_log": [],
    "prs": {},
    "streak": {"current": 0, "longest": 0, "last_log_date": None},
    "calc_profile": {},
    "meal_log": [],
    "measurements_log": [],
    "progress_photos": [],
    "weekly_recaps": []
}


# ============================================================
# LANGUAGE
# ============================================================

LANGUAGE = {
    "English": {
        "welcome": (
            "I'm your Fitness AI Trainer, and I'm here to help you actually get "
            "results. Ask me anything about training, nutrition, or workout plans "
            "— let's get moving."
        ),
        "outside": "Sorry, I couldn't find an answer in the Knowledge Base, Wikipedia, or the web.",
        "empty": "Your Knowledge Base is empty. Please put a PDF inside the knowledge_base folder.",
        "sensitive": "For security reasons, please do not send passwords, API keys, OTP codes, or other sensitive information.",
        "new": "🆕 New Chat",
        "delete": "🗑️ Delete Current Chat",
        "history": "💬 Chat History",
        "knowledge": "📚 Knowledge Base",
        "language": "🌍 Language",
        "human": "👨‍💼 Human Support",
        "summary": "📝 Conversation Summary",
        "reload": "🔄 Reload Knowledge Base",
        "instructions": "🧭 Custom Instructions",
        "instructions_placeholder": "e.g. Always be motivating. Focus on safe exercise practices.",
        "save_instructions": "💾 Save Instructions",
        "instructions_saved": "Instructions saved.",
        "onboard_title": "👋 Before we start...",
        "onboard_name": "What's your name?",
        "onboard_age": "What's your age?",
        "onboard_button": "Start Chat",
        "onboard_greeting": "Let's go, {name}! At {age}, you've got exactly the right moment to start — I'll tailor everything to that. What's the first thing you want to tackle today?",
        "onboard_greeting_returning": "Welcome back, {name} — good to see you again! Let's build on what you've already put in. What are we working on today?",
        "onboard_recognized": "Welcome back, {name} ({age})! We already have your profile saved.",
        "onboard_continue": "Continue as {name}",
        "onboard_not_you": "Not {name}? Just clear the name field and type a different one.",
        "profile_signed_in": "👤 Signed in as {name} ({age})",
        "profile_reset": "🔄 Not you? Reset profile",
        "thinking": "🤖 Thinking...",
        "error_occurred": "⚠️ An error occurred: {error}",
        "summary_generating": "Creating conversation summary...",
        "summary_generated": "Conversation Summary",
        "escalation_requested": "👨‍💼 Human support has been requested.",
        "export_chat": "📤 Export Chat",
        "calculators": "🧮 Fitness Calculators",
        "progress_tracking": "📈 Progress Tracking",
        "personal_records": "🏆 Personal Records",
        "workout_plan": "🏋️ Workout Plan",
        "grocery_list": "🛒 Grocery List",
        "progress_report": "📄 Progress Report",
        "admin_panel": "🔐 Admin Panel",
        "admin_locked": "This section is locked. Enter the admin password to manage the Knowledge Base, Custom Instructions, and escalations.",
        "admin_password_label": "Admin password",
        "admin_login": "🔓 Unlock Admin Panel",
        "admin_wrong_password": "Incorrect password.",
        "admin_unlocked": "Admin Panel unlocked for this session.",
        "admin_logout": "🔒 Lock Admin Panel",
        "measurements_photos": "📏 Measurements & Progress Photos",
        "weekly_recap": "🗓️ Weekly Recap",
        "pro_nav": "⭐ Go Pro",
        "pro_back": "⬅️ Back to Chat",
        "pro_title": "Go Pro",
        "pro_subtitle": "Pick a plan, then choose which perks matter most to you.",
        "pro_currency": "LE / month",
        "pro_choose_features": "Choose up to {n} perk(s) for this plan:",
        "pro_subscribe": "Subscribe to {plan}",
        "pro_current_plan": "✅ Current plan: {plan}",
        "pro_active_since": "Active since {date}",
        "pro_pick_warning": "Pick at least one perk before subscribing.",
        "pro_subscribed_success": "🎉 You're now on the {plan} plan!",
        "pro_cancel": "Cancel Subscription",
        "pro_cancelled": "Subscription cancelled.",
        "pro_popular": "MOST POPULAR",
        "pro_included": "Included in every plan",
    },
    "العربية": {
        "welcome": "أنا مدرب اللياقة البدنية الذكي، وهدفي مساعدتك على تحقيق نتائج حقيقية. اسألني عن التمرين أو التغذية أو خطط التمارين - لننطلق معاً.",
        "outside": "عذراً، لم أتمكن من إيجاد إجابة في قاعدة المعرفة أو ويكيبيديا أو الويب.",
        "empty": "قاعدة المعرفة فارغة. يرجى وضع ملف PDF داخل مجلد knowledge_base.",
        "sensitive": "لأسباب أمنية، يرجى عدم إرسال كلمات المرور أو مفاتيح API أو رموز OTP أو أي معلومات سرية.",
        "new": "🆕 محادثة جديدة",
        "delete": "🗑️ حذف المحادثة الحالية",
        "history": "💬 سجل المحادثات",
        "knowledge": "📚 قاعدة المعرفة",
        "language": "🌍 اللغة",
        "human": "👨‍💼 الدعم البشري",
        "summary": "📝 ملخص المحادثة",
        "reload": "🔄 إعادة تحميل قاعدة المعرفة",
        "instructions": "🧭 تعليمات مخصصة",
        "instructions_placeholder": "مثال: كن محفزاً دائماً. ركز على ممارسات التمرين الآمنة.",
        "save_instructions": "💾 حفظ التعليمات",
        "instructions_saved": "تم حفظ التعليمات.",
        "onboard_title": "👋 قبل أن نبدأ...",
        "onboard_name": "ما اسمك؟",
        "onboard_age": "كم عمرك؟",
        "onboard_button": "ابدأ المحادثة",
        "onboard_greeting": "هيا بنا يا {name}! في عمر {age}، أنت في اللحظة المثالية للبدء - سأضع ذلك في اعتباري في كل خطوة. ما أول شيء تريد أن نبدأ به اليوم؟",
        "onboard_greeting_returning": "أهلاً بعودتك يا {name}! سعيد برؤيتك مجدداً. لنبنِ على ما أنجزته حتى الآن. ما الذي سنعمل عليه اليوم؟",
        "onboard_recognized": "أهلاً بعودتك يا {name} ({age})! لدينا ملفك الشخصي محفوظ بالفعل.",
        "onboard_continue": "متابعة باسم {name}",
        "onboard_not_you": "لست {name}؟ فقط امسح حقل الاسم واكتب اسماً مختلفاً.",
        "profile_signed_in": "👤 مسجل الدخول باسم {name} ({age})",
        "profile_reset": "🔄 لست أنت؟ إعادة تعيين الملف الشخصي",
        "thinking": "🤖 أفكر...",
        "error_occurred": "⚠️ حدث خطأ: {error}",
        "summary_generating": "إنشاء ملخص المحادثة...",
        "summary_generated": "ملخص المحادثة",
        "escalation_requested": "👨‍💼 تم طلب الدعم البشري.",
        "export_chat": "📤 تصدير المحادثة",
        "calculators": "🧮 حاسبات اللياقة",
        "progress_tracking": "📈 تتبع التقدم",
        "personal_records": "🏆 الأرقام القياسية الشخصية",
        "workout_plan": "🏋️ خطة التمرين",
        "grocery_list": "🛒 قائمة التسوق",
        "progress_report": "📄 تقرير التقدم",
        "admin_panel": "🔐 لوحة التحكم",
        "admin_locked": "هذا القسم مقفل. أدخل كلمة مرور المسؤول لإدارة قاعدة المعرفة والتعليمات المخصصة والتصعيدات.",
        "admin_password_label": "كلمة مرور المسؤول",
        "admin_login": "🔓 فتح لوحة التحكم",
        "admin_wrong_password": "كلمة المرور غير صحيحة.",
        "admin_unlocked": "تم فتح لوحة التحكم لهذه الجلسة.",
        "admin_logout": "🔒 قفل لوحة التحكم",
        "measurements_photos": "📏 القياسات وصور التقدم",
        "weekly_recap": "🗓️ ملخص الأسبوع",
        "pro_nav": "⭐ اشترك في Pro",
        "pro_back": "⬅️ العودة إلى المحادثة",
        "pro_title": "اشترك في Pro",
        "pro_subtitle": "اختر خطة، ثم حدد المزايا التي تهمك أكثر.",
        "pro_currency": "جنيه / شهرياً",
        "pro_choose_features": "اختر حتى {n} ميزة لهذه الخطة:",
        "pro_subscribe": "اشترك في {plan}",
        "pro_current_plan": "✅ خطتك الحالية: {plan}",
        "pro_active_since": "مفعّلة منذ {date}",
        "pro_pick_warning": "اختر ميزة واحدة على الأقل قبل الاشتراك.",
        "pro_subscribed_success": "🎉 أنت الآن مشترك في خطة {plan}!",
        "pro_cancel": "إلغاء الاشتراك",
        "pro_cancelled": "تم إلغاء الاشتراك.",
        "pro_popular": "الأكثر طلباً",
        "pro_included": "متوفر في كل الخطط",
    }
}


# ============================================================
# PRO SUBSCRIPTION PLANS
# ============================================================
# Each plan has a fixed monthly price and a pool of optional perks.
# The plan's "max_features" controls how many perks the user is
# allowed to pick for that tier (higher tiers unlock more picks).

PRO_FEATURE_POOL = [
    {"key": "priority_ai", "label_en": "Priority AI responses (faster replies)", "label_ar": "ردود ذكاء اصطناعي بأولوية (أسرع)"},
    {"key": "unlimited_kb", "label_en": "Unlimited Knowledge Base PDF uploads", "label_ar": "رفع ملفات PDF غير محدود لقاعدة المعرفة"},
    {"key": "extra_plans", "label_en": "Unlimited saved workout plans", "label_ar": "خطط تمارين محفوظة غير محدودة"},
    {"key": "meal_photo", "label_en": "Unlimited meal-photo calorie estimates", "label_ar": "تقدير سعرات غير محدود من صور الوجبات"},
    {"key": "form_check", "label_en": "Unlimited exercise form checks", "label_ar": "فحص أداء التمارين غير محدود"},
    {"key": "weekly_recap", "label_en": "Automatic weekly recap every Monday", "label_ar": "ملخص أسبوعي تلقائي كل يوم اثنين"},
    {"key": "pdf_report", "label_en": "Branded PDF progress reports", "label_ar": "تقارير تقدم PDF بتصميم مميز"},
    {"key": "human_support", "label_en": "Priority human coach support", "label_ar": "دعم مدرب بشري بأولوية"},
]

PRO_PLANS = [
    {
        "id": "basic",
        "name_en": "Basic",
        "name_ar": "أساسي",
        "price": 100,
        "max_features": 2,
        "popular": False,
    },
    {
        "id": "plus",
        "name_en": "Plus",
        "name_ar": "بلس",
        "price": 200,
        "max_features": 4,
        "popular": True,
    },
    {
        "id": "premium",
        "name_en": "Premium",
        "name_ar": "بريميوم",
        "price": 400,
        "max_features": len(PRO_FEATURE_POOL),
        "popular": False,
    },
]


def _plan_by_id(plan_id):
    for plan in PRO_PLANS:
        if plan["id"] == plan_id:
            return plan
    return None


def _feature_label(feature_key):
    lang = st.session_state.language
    for feature in PRO_FEATURE_POOL:
        if feature["key"] == feature_key:
            return feature["label_ar"] if lang == "العربية" else feature["label_en"]
    return feature_key


# ============================================================
# CSS — PREMIUM 3D THEME
# ============================================================

def load_css():
    css = """
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #0a0a0a;
        --crimson: #dc2626;
        --crimson-2: #7f1d1d;
        --ember: #f97316;
        --blood: #450a0a;
        --paper: #fff5f2;
        --muted: #d8b9b3;
        --glass: rgba(255,70,70,0.07);
        --glass-strong: rgba(255,45,45,0.12);
        --glass-solid: #1c0a0a;
        --glass-border: rgba(255,110,90,0.30);
        --shadow-deep: 0 16px 34px rgba(0,0,0,0.55);
    }

    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Oswald', 'Manrope', sans-serif; letter-spacing: 0.2px; }

    /* =========================================================
       BASE SURFACES — every layer Streamlit stacks the page on
       (stApp is only the outermost layer; newer Streamlit adds
       stAppViewContainer / stMain / stBottom on top of it with
       their own default backgrounds, which is what was leaving
       the chat panel and chat-input bar washed out and white)
    ========================================================= */

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    [data-testid="stHeader"],
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"] {
        background: transparent !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(220,38,38,0.30), transparent 45%),
            radial-gradient(circle at 88% 12%, rgba(249,115,22,0.18), transparent 40%),
            radial-gradient(circle at 50% 105%, rgba(127,29,29,0.35), transparent 50%),
            linear-gradient(180deg, #050505 0%, #0d0505 55%, #050505 100%) !important;
        background-attachment: fixed !important;
    }

    [data-testid="stBottom"] > div { background: transparent !important; }

    /* Default, unstyled text anywhere in the app reads pale pink-white
       on the dark background — this is the base contrast fix. */
    html, body, p, span, div, label, li, td, th,
    .stMarkdown, [data-testid="stMarkdownContainer"] {
        color: #fff1f1;
    }
    h1, h2, h3, h4 { color: #ffffff; }

    a { color: var(--ember) !important; text-decoration: none; border-bottom: 1px solid rgba(249,115,22,0.35); }
    a:hover { border-bottom-color: var(--ember); }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(5,5,5,0.97), rgba(15,5,5,0.97));
        border-right: 1px solid var(--glass-border);
    }
    [data-testid="stSidebar"] * { color: #fee2e2 !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,80,80,0.15); }

    /* ---------- SIDEBAR LOGO (used by _logo_svg) ---------- */

    .hero-logo {
        flex-shrink: 0;
        filter: drop-shadow(0 8px 14px rgba(220,38,38,0.5));
        animation: logoFloat 4s ease-in-out infinite;
    }
    @keyframes logoFloat {
        0%, 100% { transform: translateY(0px) rotate(-3deg); }
        50% { transform: translateY(-6px) rotate(3deg); }
    }

    /* ---------- GLASS PANELS (3D elevation) ---------- */

    [data-testid="stChatMessage"] {
        background: var(--glass-strong);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid var(--glass-border);
        border-radius: 18px;
        padding: 6px 10px;
        margin-bottom: 12px;
        box-shadow: var(--shadow-deep), inset 0 1px 0 rgba(255,255,255,0.08);
    }

    [data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stExpander"] {
        background: var(--glass);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid var(--glass-border) !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 26px rgba(0,0,0,0.5);
    }

    div[data-testid="stExpander"] summary { border-radius: 16px !important; }
    div[data-testid="stExpander"] details { background: transparent !important; }
    .streamlit-expanderHeader, [data-testid="stExpanderToggleIcon"] { color: #fff1f1 !important; font-weight: 700; }

    /* ---------- 3D BUTTONS (tactile press) ---------- */

    .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
        border-radius: 12px;
        font-weight: 700;
        border: 1px solid rgba(255,100,100,0.3);
        background: linear-gradient(145deg, #ef4444, #7f1d1d);
        color: #ffffff !important;
        padding: 0.55rem 1.2rem;
        box-shadow:
            0 6px 0 rgba(69,10,10,0.9),
            0 12px 24px rgba(0,0,0,0.5),
            0 0 18px rgba(220,38,38,0.25);
        transition: all 0.12s ease;
    }
    .stButton > button *, .stDownloadButton > button *, .stFormSubmitButton > button * { color: #ffffff !important; }
    .stButton > button:hover, .stDownloadButton > button:hover {
        filter: brightness(1.15); transform: translateY(-1px);
        box-shadow: 0 6px 0 rgba(69,10,10,0.9), 0 12px 24px rgba(0,0,0,0.5), 0 0 26px rgba(220,38,38,0.45);
    }
    .stButton > button:active, .stDownloadButton > button:active {
        transform: translateY(4px);
        box-shadow: 0 2px 0 rgba(69,10,10,0.9), 0 4px 10px rgba(0,0,0,0.4);
    }
    .stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
        outline: 2px solid var(--ember); outline-offset: 2px;
    }

    /* ---------- CHAT INPUT ---------- */

    [data-testid="stChatInput"] {
        background: var(--glass-solid) !important;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid var(--glass-border) !important;
        border-radius: 16px !important;
        box-shadow: 0 12px 30px rgba(0,0,0,0.55), 0 0 0 rgba(220,38,38,0);
        transition: box-shadow 0.3s ease;
    }
    [data-testid="stChatInput"]:focus-within {
        box-shadow: 0 12px 30px rgba(0,0,0,0.55), 0 0 24px rgba(220,38,38,0.35);
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #ffffff !important;
        caret-color: #f97316 !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    [data-testid="stChatInput"] textarea::placeholder { color: rgba(255,220,220,0.55) !important; -webkit-text-fill-color: rgba(255,220,220,0.55) !important; }
    [data-testid="stChatInput"] button svg { fill: #f97316 !important; }
    [data-testid="stChatInputFileUploaded"], [data-testid="stChatInput"] section {
        background: rgba(255,255,255,0.05) !important; color: #fff1f1 !important;
    }

    /* ---------- TEXT / NUMBER / TEXTAREA INPUTS ---------- */

    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background: var(--glass-solid) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: rgba(255,220,220,0.5) !important; -webkit-text-fill-color: rgba(255,220,220,0.5) !important;
    }
    .stNumberInput button, [data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {
        background: rgba(255,70,70,0.14) !important;
        border-color: var(--glass-border) !important;
    }
    .stNumberInput svg { fill: #fff1f1 !important; }

    /* ---------- SELECTBOX (closed control) ---------- */

    .stSelectbox > div > div, [data-baseweb="select"] > div {
        background: var(--glass-solid) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    [data-baseweb="select"] * { color: #ffffff !important; }
    [data-baseweb="select"] svg { fill: #fff1f1 !important; }

    /* ---------- DROPDOWN / SELECT POPOVERS ----------
       These render in a portal attached to <body>, outside the
       app's own dark container, so BaseWeb's default white menu
       is what made every dropdown's options unreadable. */

    div[data-baseweb="popover"] [data-baseweb="menu"],
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li,
    [role="listbox"], [role="option"] {
        background: var(--glass-solid) !important;
        color: #fff1f1 !important;
    }
    [role="option"]:hover, [role="option"][aria-selected="true"] {
        background: rgba(220,38,38,0.35) !important;
        color: #ffffff !important;
    }
    div[data-baseweb="popover"] { border: 1px solid var(--glass-border) !important; border-radius: 12px !important; overflow: hidden; }

    /* ---------- SLIDER ---------- */

    [data-testid="stSlider"] [role="slider"] {
        background-color: var(--ember) !important;
        box-shadow: 0 0 0 6px rgba(249,115,22,0.2);
    }
    [data-testid="stSlider"] div[data-baseweb="slider"] > div > div { background: rgba(255,255,255,0.12) !important; }
    [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: #ffd7c9 !important; }
    [data-testid="stThumbValue"] { color: #ffffff !important; background: var(--crimson-2) !important; }

    /* ---------- ALERT / STATUS BOXES (info, success, warning, error) ---------- */

    [data-testid="stAlert"] {
        border-radius: 14px !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
        border: 1px solid var(--glass-border) !important;
    }
    [data-testid="stAlert"] p, [data-testid="stAlert"] div, [data-testid="stAlert"] span { color: #fff1f1 !important; }
    [data-testid="stAlertContentInfo"] { background: rgba(59,130,246,0.14) !important; }
    [data-testid="stAlertContentSuccess"] { background: rgba(34,197,94,0.14) !important; }
    [data-testid="stAlertContentWarning"] { background: rgba(249,115,22,0.16) !important; }
    [data-testid="stAlertContentError"] { background: rgba(220,38,38,0.18) !important; }

    /* ---------- STATUS ("thinking...") CONTAINER ---------- */

    [data-testid="stStatusWidget"], [data-testid="stExpanderDetails"] {
        background: var(--glass) !important;
        color: #fff1f1 !important;
    }

    /* ---------- FILE UPLOADER ---------- */

    [data-testid="stFileUploaderDropzone"] {
        background: var(--glass-solid) !important;
        border: 1px dashed var(--glass-border) !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploaderDropzone"] * { color: #fff1f1 !important; }

    /* ---------- DATAFRAME / CHART CONTAINERS ---------- */

    [data-testid="stDataFrame"], [data-testid="stArrowVegaLiteChart"] {
        border-radius: 12px !important;
        overflow: hidden;
        border: 1px solid var(--glass-border);
        background: var(--glass-solid) !important;
    }

    /* ---------- BADGES / CARDS ---------- */

    .source-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600; margin: 2px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.4);
    }

    .user-message, .assistant-message {
        border-radius: 14px; padding: 14px 16px; margin-bottom: 4px;
        background: rgba(255,255,255,0.04);
        color: #fff1f1 !important;
        line-height: 1.55;
    }
    .user-message *, .assistant-message * { color: #fff1f1 !important; }

    .pr-card {
        background: linear-gradient(135deg, rgba(220,38,38,0.18), rgba(249,115,22,0.12));
        border: 1px solid rgba(220,38,38,0.35);
        border-left: 4px solid var(--ember);
        padding: 10px 14px; border-radius: 10px; margin-bottom: 8px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.4);
    }

    .streak-badge {
        background: linear-gradient(145deg, #f97316, #b91c1c);
        color: white; padding: 12px 18px; border-radius: 14px;
        text-align: center; font-weight: 800; font-size: 1.05rem;
        box-shadow: 0 8px 0 rgba(69,10,10,0.8), 0 14px 26px rgba(0,0,0,0.5), 0 0 22px rgba(249,115,22,0.35);
    }

    /* ---------- PRICING CARDS (Go Pro page) ---------- */

    .pricing-card {
        position: relative;
        background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,45,45,0.06));
        border: 1px solid var(--glass-border);
        border-radius: 18px;
        padding: 22px 20px 10px 20px;
        margin-bottom: 10px;
        box-shadow: var(--shadow-deep), inset 0 1px 0 rgba(255,255,255,0.06);
        text-align: center;
    }
    .pricing-card.popular {
        border: 1px solid rgba(249,115,22,0.65);
        box-shadow: 0 0 0 1px rgba(249,115,22,0.35), var(--shadow-deep), 0 0 30px rgba(249,115,22,0.25);
        transform: translateY(-6px);
    }
    .pricing-badge {
        position: absolute; top: -12px; left: 50%; transform: translateX(-50%);
        background: linear-gradient(145deg, #f97316, #b91c1c);
        color: #fff; font-size: 0.68rem; font-weight: 800; letter-spacing: 0.6px;
        padding: 4px 14px; border-radius: 999px;
        box-shadow: 0 6px 14px rgba(0,0,0,0.5);
        white-space: nowrap;
    }
    .pricing-name {
        font-family: 'Oswald', sans-serif; text-transform: uppercase;
        letter-spacing: 1px; font-size: 1.15rem; color: #ffffff; font-weight: 700; margin-top: 4px;
    }
    .pricing-price {
        font-family: 'Oswald', sans-serif; font-size: 2.1rem; font-weight: 800;
        color: #ffb020; margin: 6px 0 0 0; line-height: 1;
    }
    .pricing-unit { font-size: 0.78rem; color: rgba(255,220,220,0.6); font-weight: 600; margin-bottom: 4px; }

    .plan-active-badge {
        display: inline-block; margin-top: 6px; padding: 3px 12px; border-radius: 999px;
        background: rgba(34,197,94,0.18); border: 1px solid rgba(34,197,94,0.45);
        color: #86efac !important; font-size: 0.75rem; font-weight: 700;
    }

    /* ---------- SCROLLBAR ---------- */
    ::-webkit-scrollbar { width: 9px; }
    ::-webkit-scrollbar-thumb { background: rgba(220,38,38,0.5); border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(220,38,38,0.8); }
    """
    st.markdown("<style>{}</style>".format(css), unsafe_allow_html=True)


load_css()


# ============================================================
# PREMIUM LOGO + HERO HEADER
# ============================================================

def _logo_svg(size=48):
    return """
    <svg class="hero-logo" width="{size}" height="{size}" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <radialGradient id="dbGrad" cx="35%" cy="30%" r="70%">
                <stop offset="0%" stop-color="#3f3f46"/>
                <stop offset="55%" stop-color="#18181b"/>
                <stop offset="100%" stop-color="#000000"/>
            </radialGradient>
            <linearGradient id="rimGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#f97316"/>
                <stop offset="100%" stop-color="#dc2626"/>
            </linearGradient>
        </defs>
        <circle cx="24" cy="24" r="23" fill="url(#dbGrad)" stroke="url(#rimGrad)" stroke-width="2"/>
        <ellipse cx="17" cy="13" rx="11" ry="6" fill="rgba(255,255,255,0.10)"/>
        <g transform="translate(24,24)">
            <rect x="-15" y="-3.2" width="30" height="6.4" rx="3.2" fill="url(#rimGrad)"/>
            <rect x="-18.5" y="-8" width="7" height="16" rx="2.5" fill="url(#rimGrad)"/>
            <rect x="11.5" y="-8" width="7" height="16" rx="2.5" fill="url(#rimGrad)"/>
            <rect x="-17.2" y="-6" width="2.2" height="12" rx="1.1" fill="rgba(255,255,255,0.35)"/>
            <rect x="13" y="-6" width="2.2" height="12" rx="1.1" fill="rgba(255,255,255,0.35)"/>
        </g>
    </svg>
    """.format(size=size)


def render_hero_header():
    profile = st.session_state.profile
    tracking = profile.get("tracking", {})
    streak = tracking.get("streak", {})
    prs_count = len(tracking.get("prs", {}))
    workouts_count = len(tracking.get("workout_log", []))
    streak_current = streak.get("current", 0)

    def _ring_pct(value, cap):
        return max(0.06, min(value / cap, 1.0)) if cap else 0.06

    ring_streak_pct = _ring_pct(streak_current, 30)
    ring_prs_pct = _ring_pct(prs_count, 10)
    ring_workouts_pct = _ring_pct(workouts_count, 20)

    RING_R = 30
    RING_C = round(2 * 3.14159265 * RING_R, 2)

    def _ring_offset(pct):
        return round(RING_C * (1 - pct), 2)

    html = """
    <div id="hero-root">
      <style>
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; overflow: hidden; }}

        #hero-root {{
            position: relative;
            font-family: 'Manrope', -apple-system, sans-serif;
            border-radius: 20px;
            padding: 26px 32px;
            overflow: hidden;
            background:
                repeating-linear-gradient(
                    115deg,
                    rgba(255,255,255,0.018) 0px,
                    rgba(255,255,255,0.018) 1px,
                    transparent 1px,
                    transparent 5px
                ),
                radial-gradient(circle at 85% 100%, rgba(255,176,32,0.10), transparent 55%),
                linear-gradient(160deg, #131316 0%, #0a0a0c 60%, #050506 100%);
            box-shadow: 0 22px 48px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.05);
            border: 1px solid rgba(255,176,32,0.16);
        }}

        #stage {{
            position: relative;
            display: flex;
            align-items: center;
            gap: 28px;
            z-index: 2;
        }}

        /* ---- live EKG / heart-rate readout ---- */
        #ekg-wrap {{
            position: relative;
            flex-shrink: 0;
            width: 132px;
            height: 76px;
            border-radius: 12px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,176,32,0.20);
            overflow: hidden;
        }}
        #ekg-wrap svg {{ position: absolute; top: 0; left: 0; height: 100%; width: 200%; }}
        #ekg-track {{ display: flex; animation: scrollEkg 3.2s linear infinite; }}
        @keyframes scrollEkg {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-50%); }} }}
        #bpm-label {{
            position: absolute; left: 8px; bottom: 6px;
            font-family: 'Oswald', sans-serif; font-size: 0.62rem; font-weight: 600;
            letter-spacing: 0.5px; color: #ff5c5c;
            display: flex; align-items: center; gap: 4px;
        }}
        .dot {{ width: 5px; height: 5px; border-radius: 50%; background: #ff5c5c; animation: dotPulse 1s ease-in-out infinite; }}
        @keyframes dotPulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.25; }} }}

        /* ---- title block with one-shot load bar ---- */
        #title-block {{ flex: 1; min-width: 0; }}
        #title-block h1 {{
            margin: 0; font-family: 'Oswald', sans-serif;
            font-size: 1.9rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 1px; color: #f5f3ef;
        }}
        #title-block p {{
            margin: 5px 0 10px 0; color: rgba(245,243,239,0.62);
            font-weight: 500; font-size: 0.9rem;
        }}
        #loadbar {{
            width: 100%; max-width: 340px; height: 5px; border-radius: 3px;
            background: rgba(255,255,255,0.08); overflow: hidden;
        }}
        #loadbar-fill {{
            height: 100%; width: 0%;
            background: linear-gradient(90deg, #ffb020, #ff7a1a);
            border-radius: 3px;
            animation: loadIn 1.4s cubic-bezier(.2,.8,.2,1) 0.2s forwards;
            box-shadow: 0 0 10px rgba(255,176,32,0.5);
        }}
        @keyframes loadIn {{ to {{ width: 100%; }} }}

        /* ---- stat rings ---- */
        #rings {{ display: flex; gap: 18px; flex-shrink: 0; }}
        .ring-item {{ display: flex; flex-direction: column; align-items: center; gap: 4px; }}
        .ring-item svg {{ transform: rotate(-90deg); }}
        .ring-track {{ fill: none; stroke: rgba(255,255,255,0.08); stroke-width: 6; }}
        .ring-fill {{
            fill: none; stroke: #ffb020; stroke-width: 6; stroke-linecap: round;
            stroke-dasharray: {ring_c};
            stroke-dashoffset: {ring_c};
            animation: drawRing 1.1s cubic-bezier(.2,.8,.2,1) 0.3s forwards;
        }}
        .ring-item:nth-child(2) .ring-fill {{ stroke: #ff5c5c; animation-delay: 0.45s; }}
        .ring-item:nth-child(3) .ring-fill {{ stroke: #f5f3ef; animation-delay: 0.6s; }}
        @keyframes drawRing {{ to {{ stroke-dashoffset: var(--target); }} }}
        .ring-num {{ font-family: 'Oswald', sans-serif; font-size: 1.05rem; font-weight: 700; fill: #f5f3ef; }}
        .ring-label {{ font-size: 0.68rem; font-weight: 600; color: rgba(245,243,239,0.55); text-transform: uppercase; letter-spacing: 0.5px; }}
      </style>

      <div id="stage">

        <div id="ekg-wrap">
          <div id="ekg-track">
            <svg viewBox="0 0 220 76" preserveAspectRatio="none">
              <polyline points="0,38 20,38 28,38 34,14 40,60 46,20 52,38 70,38 96,38 104,38 110,16 116,58 122,22 128,38 146,38 172,38 180,38 186,14 192,60 198,20 204,38 220,38"
                fill="none" stroke="#ff5c5c" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"
                style="filter: drop-shadow(0 0 4px rgba(255,92,92,0.6));" />
            </svg>
            <svg viewBox="0 0 220 76" preserveAspectRatio="none">
              <polyline points="0,38 20,38 28,38 34,14 40,60 46,20 52,38 70,38 96,38 104,38 110,16 116,58 122,22 128,38 146,38 172,38 180,38 186,14 192,60 198,20 204,38 220,38"
                fill="none" stroke="#ff5c5c" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"
                style="filter: drop-shadow(0 0 4px rgba(255,92,92,0.6));" />
            </svg>
          </div>
          <div id="bpm-label"><span class="dot"></span>LIVE COACHING</div>
        </div>

        <div id="title-block">
          <h1>Fitness AI Trainer</h1>
          <p>Your personal coach for training, nutrition &amp; progress</p>
          <div id="loadbar"><div id="loadbar-fill"></div></div>
        </div>

        <div id="rings">
          <div class="ring-item">
            <svg width="68" height="68" viewBox="0 0 68 68">
              <circle class="ring-track" cx="34" cy="34" r="{ring_r}"/>
              <circle class="ring-fill" cx="34" cy="34" r="{ring_r}" style="--target:{off_streak}px"/>
              <text x="34" y="39" text-anchor="middle" class="ring-num" transform="rotate(90 34 34)">{streak}</text>
            </svg>
            <span class="ring-label">Streak</span>
          </div>
          <div class="ring-item">
            <svg width="68" height="68" viewBox="0 0 68 68">
              <circle class="ring-track" cx="34" cy="34" r="{ring_r}"/>
              <circle class="ring-fill" cx="34" cy="34" r="{ring_r}" style="--target:{off_prs}px"/>
              <text x="34" y="39" text-anchor="middle" class="ring-num" transform="rotate(90 34 34)">{prs}</text>
            </svg>
            <span class="ring-label">PRs</span>
          </div>
          <div class="ring-item">
            <svg width="68" height="68" viewBox="0 0 68 68">
              <circle class="ring-track" cx="34" cy="34" r="{ring_r}"/>
              <circle class="ring-fill" cx="34" cy="34" r="{ring_r}" style="--target:{off_workouts}px"/>
              <text x="34" y="39" text-anchor="middle" class="ring-num" transform="rotate(90 34 34)">{workouts}</text>
            </svg>
            <span class="ring-label">Workouts</span>
          </div>
        </div>

      </div>
    </div>
    """.format(
        ring_c=RING_C, ring_r=RING_R,
        streak=streak_current, prs=prs_count, workouts=workouts_count,
        off_streak=_ring_offset(ring_streak_pct),
        off_prs=_ring_offset(ring_prs_pct),
        off_workouts=_ring_offset(ring_workouts_pct)
    )

    components.html(html, height=190, scrolling=False)

def render_sidebar_header():
    logo = _logo_svg(38)

    parts = [
        '<div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">',
        logo,
        '<div>',
        '<div style="font-weight:800; font-size:1.2rem; color:#ffffff; letter-spacing:-0.3px;">',
        'Fitness AI Trainer',
        '</div>',
        '<div style="font-size:0.75rem; color:rgba(255,180,180,0.75); font-weight:500;">',
        'Train &bull; Fuel &bull; Progress',
        '</div>',
        '</div>',
        '</div>',
    ]

    st.markdown("".join(parts), unsafe_allow_html=True)


# ============================================================
# UTILITY
# ============================================================

def get_timestamp():
    return datetime.now().isoformat()


def get_date_str():
    return datetime.now().strftime("%Y-%m-%d")


def contains_sensitive_data(text):
    patterns = [
        r"sk-[A-Za-z0-9_-]{20,}",
        r"gsk_[A-Za-z0-9_-]{20,}",
        r"password\s*[:=]\s*\S+",
        r"Bearer\s+[A-Za-z0-9._~+/=-]+",
        r"\b(?:otp|verification code)\s*[:=]?\s*\d{4,8}\b",
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        r"(?:\d{1,3}\.){3}\d{1,3}",
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


# ============================================================
# USER PROFILES (multiple people can share this app; each
# person's name/age + tracking data is saved under their own
# name, so they're recognized on return without re-asking, but
# a different name always gets its own fresh profile)
# ============================================================

def _default_profile():
    return {
        "name": None,
        "age": None,
        "preferences": {},
        "tracking": json.loads(json.dumps(DEFAULT_TRACKING)),
        "saved_plan": None,
        "grocery_list": None,
        "subscription": None
    }


def _name_key(name):
    return (name or "").strip().lower()


def _merge_with_defaults(data):
    merged = _default_profile()
    merged.update(data)

    tracking = json.loads(json.dumps(DEFAULT_TRACKING))
    tracking.update(data.get("tracking", {}) or {})
    merged["tracking"] = tracking

    return merged


def load_all_profiles():
    """Returns {name_key: profile_dict} for every saved person."""
    if not USER_PROFILE_FILE.exists():
        return {}

    try:
        with open(USER_PROFILE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and isinstance(data.get("profiles"), dict):
            return {
                key: _merge_with_defaults(value)
                for key, value in data["profiles"].items()
                if isinstance(value, dict)
            }

        # Backward compatibility with the old single-profile file format.
        if isinstance(data, dict) and data.get("name"):
            key = _name_key(data["name"])
            return {key: _merge_with_defaults(data)}

    except Exception as e:
        logger.error("Profiles loading error: %s", e)

    return {}


def save_all_profiles(profiles):
    try:
        with open(USER_PROFILE_FILE, "w", encoding="utf-8") as f:
            json.dump({"profiles": profiles}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("Profiles save error: %s", e)


def get_profile_by_name(name):
    if not name or not name.strip():
        return None
    return load_all_profiles().get(_name_key(name))


def save_profile(profile):
    """Saves/updates a single named profile inside the shared profiles file."""
    name = profile.get("name")
    if not name:
        return
    profile["last_updated"] = get_timestamp()
    profiles = load_all_profiles()
    profiles[_name_key(name)] = profile
    save_all_profiles(profiles)


def create_or_load_profile(name, age):
    """Loads the existing saved profile for this name if one exists
    (keeping their saved age + tracking history), otherwise creates
    a brand new profile for them."""
    existing = get_profile_by_name(name)
    if existing:
        return existing, True

    profile = _default_profile()
    profile["name"] = name.strip()
    profile["age"] = age
    save_profile(profile)
    return profile, False


def clear_named_profile(name):
    if not name:
        return
    profiles = load_all_profiles()
    key = _name_key(name)
    if key in profiles:
        del profiles[key]
        save_all_profiles(profiles)


# ---------- Tracking helpers ----------

def update_streak(tracking):
    streak = tracking.setdefault("streak", {"current": 0, "longest": 0, "last_log_date": None})
    today = get_date_str()
    last = streak.get("last_log_date")

    if last == today:
        return streak

    last_date = None
    if last:
        try:
            last_date = datetime.strptime(last, "%Y-%m-%d").date()
        except Exception:
            last_date = None

    today_date = datetime.strptime(today, "%Y-%m-%d").date()

    if last_date and (today_date - last_date).days == 1:
        streak["current"] += 1
    else:
        streak["current"] = 1

    streak["longest"] = max(streak.get("longest", 0), streak["current"])
    streak["last_log_date"] = today
    return streak


def log_weight(profile, weight_kg, note=""):
    tracking = profile.setdefault("tracking", json.loads(json.dumps(DEFAULT_TRACKING)))
    tracking.setdefault("weight_log", []).append({
        "date": get_date_str(), "weight_kg": weight_kg, "note": note
    })
    save_profile(profile)


def log_workout(profile, workout_type, note=""):
    tracking = profile.setdefault("tracking", json.loads(json.dumps(DEFAULT_TRACKING)))
    tracking.setdefault("workout_log", []).append({
        "date": get_date_str(), "type": workout_type, "note": note
    })
    streak = update_streak(tracking)
    save_profile(profile)
    return streak


def log_pr(profile, lift, weight, unit, reps, note=""):
    tracking = profile.setdefault("tracking", json.loads(json.dumps(DEFAULT_TRACKING)))
    prs = tracking.setdefault("prs", {})
    existing = prs.get(lift)

    is_new = (not existing) or (weight > existing.get("weight", 0))

    if is_new:
        prs[lift] = {
            "weight": weight, "unit": unit, "reps": reps,
            "date": get_date_str(), "note": note
        }
        save_profile(profile)

    return is_new


def log_meal_estimate(profile, analysis_text):
    tracking = profile.setdefault("tracking", json.loads(json.dumps(DEFAULT_TRACKING)))

    calories = None
    match = re.search(r"(\d{2,4})\s*(?:kcal|calories)", analysis_text, re.IGNORECASE)
    if match:
        try:
            calories = int(match.group(1))
        except Exception:
            calories = None

    tracking.setdefault("meal_log", []).append({
        "date": get_date_str(), "estimate": analysis_text[:400], "calories": calories
    })
    save_profile(profile)
    return calories


def log_measurements(profile, measurements, note=""):
    """measurements: dict like {'chest_cm': 100, 'waist_cm': 80, ...} - only non-None values kept."""
    tracking = profile.setdefault("tracking", json.loads(json.dumps(DEFAULT_TRACKING)))
    entry = {"date": get_date_str(), "note": note}
    for key, _ in MEASUREMENT_FIELDS:
        value = measurements.get(key)
        if value is not None and value > 0:
            entry[key] = value
    tracking.setdefault("measurements_log", []).append(entry)
    save_profile(profile)
    return entry


def save_progress_photo(profile, image_bytes, mime_type, note=""):
    """Saves a progress photo to disk (per-user folder-free, single-user app) and logs metadata."""
    tracking = profile.setdefault("tracking", json.loads(json.dumps(DEFAULT_TRACKING)))

    ext = "jpg"
    if mime_type == "image/png":
        ext = "png"
    elif mime_type == "image/webp":
        ext = "webp"

    filename = "{}_{}.{}".format(get_date_str(), hashlib.md5(os.urandom(8)).hexdigest()[:6], ext)
    filepath = PROGRESS_PHOTOS_DIR / filename

    try:
        with open(filepath, "wb") as f:
            f.write(image_bytes)
    except Exception as e:
        logger.error("Progress photo save error: %s", e)
        return None

    entry = {
        "date": get_date_str(), "filename": filename, "note": note, "timestamp": get_timestamp()
    }
    tracking.setdefault("progress_photos", []).append(entry)
    save_profile(profile)
    return entry


def delete_progress_photo(profile, filename):
    tracking = profile.setdefault("tracking", json.loads(json.dumps(DEFAULT_TRACKING)))
    photos = tracking.setdefault("progress_photos", [])
    tracking["progress_photos"] = [p for p in photos if p.get("filename") != filename]
    save_profile(profile)

    try:
        filepath = PROGRESS_PHOTOS_DIR / filename
        if filepath.exists():
            filepath.unlink()
    except Exception as e:
        logger.error("Progress photo delete error: %s", e)


def set_subscription(profile, plan_id, feature_keys):
    profile["subscription"] = {
        "plan_id": plan_id,
        "features": feature_keys,
        "date": get_date_str(),
        "timestamp": get_timestamp()
    }
    save_profile(profile)


def cancel_subscription(profile):
    profile["subscription"] = None
    save_profile(profile)


# ============================================================
# FITNESS CALCULATORS
# ============================================================

def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    return round(bmi, 1), category


def calculate_tdee(weight_kg, height_cm, age, sex, activity_level):
    if sex == "male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.2)
    return round(bmr * multiplier)


def calculate_macros(tdee, weight_kg, goal):
    if goal == "cut":
        calories = tdee - 500
        protein_g = weight_kg * 2.2
    elif goal == "bulk":
        calories = tdee + 300
        protein_g = weight_kg * 1.8
    else:
        calories = tdee
        protein_g = weight_kg * 1.8

    fat_calories = calories * 0.25
    fat_g = fat_calories / 9

    protein_calories = protein_g * 4
    carb_calories = max(calories - protein_calories - fat_calories, 0)
    carb_g = carb_calories / 4

    return {
        "calories": round(calories),
        "protein_g": round(protein_g),
        "fat_g": round(fat_g),
        "carbs_g": round(carb_g)
    }


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

if "profile" not in st.session_state:
    # No one has identified themselves in this browser session yet,
    # so start with a blank profile — it gets swapped for the right
    # saved (or brand new) named profile once onboarding completes.
    st.session_state.profile = _default_profile()

if "custom_instructions" not in st.session_state:
    if INSTRUCTIONS_FILE.exists():
        try:
            with open(INSTRUCTIONS_FILE, "r", encoding="utf-8") as f:
                st.session_state.custom_instructions = f.read()
        except Exception:
            st.session_state.custom_instructions = ""
    else:
        st.session_state.custom_instructions = ""

if "image_purpose" not in st.session_state:
    st.session_state.image_purpose = "general"

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

# Onboarding ("what's your name/age") is scoped to THIS browser
# session, not to the shared user_profile.json file. The file is
# still used to persist tracking data (weight, workouts, PRs, etc.)
# across restarts for whoever is currently using the app, but it
# must never be used to silently recognize a *different* visitor
# as a returning user and skip asking them for their name/age.
if "session_profile_confirmed" not in st.session_state:
    st.session_state.session_profile_confirmed = False

if "session_user_name" not in st.session_state:
    st.session_state.session_user_name = None

if "session_user_age" not in st.session_state:
    st.session_state.session_user_age = None

if "show_pro_page" not in st.session_state:
    st.session_state.show_pro_page = False


T = LANGUAGE[st.session_state.language]


# ============================================================
# CHAT CREATION
# ============================================================

def make_new_chat_dict():
    has_profile = st.session_state.session_profile_confirmed
    session_name = st.session_state.session_user_name
    session_age = st.session_state.session_user_age

    chat_id = hashlib.md5(os.urandom(20)).hexdigest()[:8]

    chat = {
        "id": chat_id,
        "title": "New Chat",
        "messages": [],
        "escalated": False,
        "user_name": session_name if has_profile else None,
        "user_age": session_age if has_profile else None,
        "onboarded": has_profile,
        "created_at": get_timestamp(),
        "last_updated": get_timestamp(),
        "tags": [],
        "summary": ""
    }

    if has_profile:
        greeting = T["onboard_greeting_returning"].format(name=chat["user_name"])
        chat["messages"].append({
            "role": "assistant", "content": greeting, "timestamp": get_timestamp()
        })

    return chat


if "current_chat" not in st.session_state:
    chat = make_new_chat_dict()
    st.session_state.current_chat = chat["id"]
    st.session_state.conversations[chat["id"]] = chat


# ============================================================
# PDF FUNCTIONS (Knowledge Base)
# ============================================================

def read_pdf(pdf_file):
    text = ""
    try:
        reader = PdfReader(str(pdf_file))
        for page_number, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += "\n[PAGE_{}]\n{}".format(page_number + 1, page_text)
            except Exception as e:
                logger.warning("PDF page error: %s", e)
    except Exception as e:
        logger.error("PDF reading error: %s", e)
    return text


def split_text(text, chunk_size=800, overlap=100):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) > chunk_size and current:
            chunks.append(current.strip())
            words = current.split()
            overlap_words = words[-(overlap // 5):]
            current = " ".join(overlap_words) + " " + sentence
        else:
            current += " " + sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks


def load_knowledge():
    documents = []
    pdf_files = list(KNOWLEDGE_DIR.glob("*.pdf"))

    for pdf_file in pdf_files:
        text = read_pdf(pdf_file)
        chunks = split_text(text)

        for index, chunk in enumerate(chunks):
            if chunk.strip():
                documents.append({
                    "text": chunk.strip(), "source": pdf_file.name,
                    "chunk_id": index, "timestamp": get_timestamp()
                })

    return documents


def build_index():
    documents = load_knowledge()
    if not documents:
        return [], None, None

    texts = [document["text"] for document in documents]

    try:
        vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), max_features=50000,
            stop_words=("english" if st.session_state.language == "English" else None)
        )
        matrix = vectorizer.fit_transform(texts)
        return documents, vectorizer, matrix
    except Exception as e:
        logger.error("Index error: %s", e)
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
    if (not st.session_state.documents or st.session_state.vectorizer is None
            or st.session_state.matrix is None):
        return []

    try:
        question_vector = st.session_state.vectorizer.transform([question])
        scores = cosine_similarity(question_vector, st.session_state.matrix)[0]
        ranked = scores.argsort()[::-1]

        results = []
        minimum = max(KB_THRESHOLD, MIN_SIMILARITY_SCORE)

        for index in ranked[:10]:
            score = float(scores[index])
            if score >= minimum:
                document = st.session_state.documents[index]
                results.append({
                    "text": document["text"],
                    "source": "Knowledge Base: {}".format(document["source"]),
                    "score": score
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:5]

    except Exception as e:
        logger.error("Knowledge search error: %s", e)
        return []


def search_wikipedia(question):
    language = WIKI_LANG_MAP.get(st.session_state.language, "en")
    wikipedia.set_lang(language)
    results = []

    try:
        titles = wikipedia.search(question, results=WIKI_RESULTS)
    except Exception as e:
        logger.error("Wikipedia search error: %s", e)
        return results

    for title in titles:
        try:
            summary = wikipedia.summary(title, sentences=WIKI_SENTENCES, auto_suggest=False)
            if summary.strip():
                results.append({
                    "text": summary,
                    "source": "Wikipedia: {}".format(title),
                    "score": None,
                    "url": "https://{}.wikipedia.org/wiki/{}".format(language, title.replace(" ", "_"))
                })
        except wikipedia.exceptions.DisambiguationError as e:
            if not e.options:
                continue
            try:
                option = e.options[0]
                summary = wikipedia.summary(option, sentences=WIKI_SENTENCES, auto_suggest=False)
                if summary.strip():
                    results.append({
                        "text": summary, "source": "Wikipedia: {}".format(option), "score": None
                    })
            except Exception:
                continue
        except Exception:
            continue

    return results


def search_web(question):
    results = []
    if DDGS is None:
        logger.warning("ddgs package is not installed.")
        return results

    try:
        with DDGS() as ddgs:
            hits = ddgs.text(question, max_results=WEB_RESULTS)

            for hit in hits:
                title = hit.get("title", "").strip()
                body = hit.get("body", "").strip()
                url = hit.get("href") or hit.get("link") or ""

                if not body or len(body) < 20:
                    continue

                body = re.sub(r"\s+", " ", body).strip()

                results.append({
                    "text": body,
                    "source": "Web: {} ({})".format(title, url),
                    "score": None,
                    "url": url
                })

    except Exception as e:
        logger.error("Web search error: %s", e)

    return results


# ============================================================
# SOURCE BADGES
# ============================================================

BADGE_COLORS = {
    "Knowledge Base": "#0f9d58",
    "Wikipedia": "#4285f4",
    "Web": "#f4b400",
    "Image Analysis": "#9c27b0",
    "Meal Analysis": "#e91e63",
    "Form Check": "#00897b"
}


def render_source_badges(sources, placeholder=None):
    if not sources:
        return

    html = '<div style="margin-top:10px;">'
    for source in sources:
        color = BADGE_COLORS.get(source, "#757575")
        html += (
            '<span class="source-badge" style="background:{0}22;color:{0};'
            'border:1px solid {0}55;">{1}</span>'
        ).format(color, source)
    html += "</div>"

    if placeholder:
        placeholder.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


# ============================================================
# MESSAGE BUILDERS
# ============================================================

def _language_instruction():
    return "Answer in Arabic." if st.session_state.language == "العربية" else "Answer in English."


def _user_info(user_name, user_age):
    if user_name:
        return "You are talking to {}, age {}.".format(user_name, user_age)
    return ""


COACH_ENERGY_INSTRUCTIONS = """
COACHING ENERGY:
Bring real energy to this. You are the kind of coach a client is
genuinely glad to hear from — upbeat, warm, and motivating, like
someone who is invested in their progress and believes they can do
this. Open with momentum instead of a flat restatement of the
question. Acknowledge effort and progress specifically when the
user mentions it (a workout logged, a habit kept, a hard set
finished) rather than skipping past it to the facts. When the news
is a setback — a missed session, a plateau, a bad number on the
scale — reframe it as a normal part of the process and point
forward, don't just sympathize and stop there. Close with a short,
concrete nudge toward the next action rather than trailing off.
Energy comes from word choice, pacing, and genuine specificity —
not from stacking exclamation points, emoji, or hype for its own
sake. Never let energy come at the cost of being accurate or safe:
if something needs a caution or a "check with a professional,"
still say it plainly, just without being cold about it.
"""


def build_answer_messages(question, search_results, history, user_name=None, user_age=None):
    recent_history = history[-MAX_CONTEXT_MESSAGES:]

    context_parts = [
        "Source: {}\n{}".format(item["source"], item["text"]) for item in search_results
    ]
    context = "\n\n".join(context_parts)

    custom = st.session_state.custom_instructions.strip()

    system_prompt = """
You are a Professional Fitness AI Trainer.

{language}

{user_info}

Answer using the CONTEXT below.

Prefer Knowledge Base information when available.

Do not invent information that is not supported by the context.

If the answer cannot be found in the context, say that you could not find the answer.

Be professional, concise, helpful, and clear.

Focus on fitness, nutrition, exercise, and general wellness guidance.

{energy}

CUSTOM ADMIN INSTRUCTIONS:
{custom}

CONTEXT:
{context}

END CONTEXT.
""".format(
        language=_language_instruction(),
        user_info=_user_info(user_name, user_age),
        energy=COACH_ENERGY_INSTRUCTIONS,
        custom=custom,
        context=context
    )

    messages = [{"role": "system", "content": system_prompt.strip()}]

    for message in recent_history:
        content = message.get("content")
        if isinstance(content, str):
            messages.append({"role": message["role"], "content": content})

    messages.append({"role": "user", "content": question})
    return messages


def _vision_messages_base(question, image_b64, mime_type, history, system_prompt):
    recent_history = history[-MAX_CONTEXT_MESSAGES:]
    messages = [{"role": "system", "content": system_prompt.strip()}]

    for message in recent_history:
        content = message.get("content")
        if isinstance(content, str):
            messages.append({"role": message["role"], "content": content})

    image_data_url = "data:{};base64,{}".format(mime_type, image_b64)

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": question or "Analyze this image."},
            {"type": "image_url", "image_url": {"url": image_data_url}}
        ]
    })

    return messages


def build_vision_messages(question, image_b64, mime_type, history, user_name=None, user_age=None):
    custom = st.session_state.custom_instructions.strip()

    system_prompt = """
You are a Professional Fitness AI Trainer.

{language}

{user_info}

The user has attached an image. Carefully analyze the image and answer
the user's question about it. Only describe what can reasonably be
observed from the image. Do not claim certainty about things that
cannot be determined visually.

Be professional, concise, and helpful.

{energy}

CUSTOM ADMIN INSTRUCTIONS:
{custom}
""".format(
        language=_language_instruction(), user_info=_user_info(user_name, user_age),
        energy=COACH_ENERGY_INSTRUCTIONS, custom=custom
    )

    return _vision_messages_base(question, image_b64, mime_type, history, system_prompt)


def build_meal_vision_messages(question, image_b64, mime_type, history, user_name=None, user_age=None):
    custom = st.session_state.custom_instructions.strip()

    system_prompt = """
You are a Professional Fitness AI Trainer specialized in nutrition.

{language}

{user_info}

The user has attached a photo of a MEAL. Identify the visible foods and
give a realistic ESTIMATED calorie count and macro breakdown (protein,
carbs, fat) for the whole plate. Always state a clear total, e.g.
"Estimated total: ~550 kcal". Make clear this is a rough visual
estimate, not a precise measurement, since exact ingredients and
portion sizes can't be confirmed from a photo.

Be encouraging and non-judgmental about food choices.

{energy}

CUSTOM ADMIN INSTRUCTIONS:
{custom}
""".format(
        language=_language_instruction(), user_info=_user_info(user_name, user_age),
        energy=COACH_ENERGY_INSTRUCTIONS, custom=custom
    )

    return _vision_messages_base(
        question or "Estimate the calories and macros in this meal.",
        image_b64, mime_type, history, system_prompt
    )


def build_form_check_vision_messages(question, image_b64, mime_type, history, user_name=None, user_age=None):
    custom = st.session_state.custom_instructions.strip()

    system_prompt = """
You are a Professional Fitness AI Trainer specialized in exercise form.

{language}

{user_info}

The user has attached a photo of themselves performing an EXERCISE.
Identify the exercise if possible, and give general observations about
form (joint alignment, posture, range of motion) based only on what is
visible. Point out 1-3 specific things that look good and 1-3 things
that could be adjusted, if any.

You are NOT a physical therapist or doctor. Never diagnose an injury.
If the image suggests pain, strain, or clearly unsafe form, advise the
user to consult a qualified coach or medical professional and to stop
the movement if it hurts.

{energy}

CUSTOM ADMIN INSTRUCTIONS:
{custom}
""".format(
        language=_language_instruction(), user_info=_user_info(user_name, user_age),
        energy=COACH_ENERGY_INSTRUCTIONS, custom=custom
    )

    return _vision_messages_base(
        question or "Check my form in this exercise photo.",
        image_b64, mime_type, history, system_prompt
    )


# ============================================================
# GROQ STREAM
# ============================================================

def stream_groq(messages, model):
    stream = client.chat.completions.create(
        model=model, messages=messages, temperature=0.7, max_tokens=1500, stream=True
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ============================================================
# GENERATE RESPONSE (text or image, with vision_mode)
# Live, step-by-step "thinking" status: each retrieval/generation
# stage updates the status label in real time instead of a single
# static "Thinking..." message.
# ============================================================

STEP_LABELS = {
    "start": "🤖 Getting started...",
    "kb": "🔍 Searching Knowledge Base...",
    "wiki": "🌐 Checking Wikipedia...",
    "web": "🌍 Searching the web...",
    "vision": "🖼️ Analyzing your image...",
    "writing": "✍️ Writing your answer...",
    "done": "✅ Done"
}


def generate_response(chat, question, image_b64=None, mime_type=None, vision_mode="general"):
    history = chat["messages"][:-1]

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        sources_placeholder = st.empty()

        if image_b64:
            with st.status(STEP_LABELS["start"], expanded=True) as status:
                status.update(label=STEP_LABELS["vision"], state="running")

                if vision_mode == "meal":
                    messages = build_meal_vision_messages(
                        question, image_b64, mime_type, history,
                        user_name=chat.get("user_name"), user_age=chat.get("user_age")
                    )
                elif vision_mode == "form":
                    messages = build_form_check_vision_messages(
                        question, image_b64, mime_type, history,
                        user_name=chat.get("user_name"), user_age=chat.get("user_age")
                    )
                else:
                    messages = build_vision_messages(
                        question, image_b64, mime_type, history,
                        user_name=chat.get("user_name"), user_age=chat.get("user_age")
                    )

                status.update(label=STEP_LABELS["writing"], state="running")

                try:
                    answer = ""
                    for chunk in stream_groq(messages, VISION_MODEL):
                        answer += chunk
                        answer_placeholder.markdown(answer)

                    if vision_mode == "meal":
                        sources = ["Meal Analysis"]
                        log_meal_estimate(st.session_state.profile, answer)
                    elif vision_mode == "form":
                        sources = ["Form Check"]
                    else:
                        sources = ["Image Analysis"]

                    status.update(label=STEP_LABELS["done"], state="complete")

                except Exception as e:
                    logger.exception("Vision error")
                    answer = T["error_occurred"].format(error=str(e))
                    answer_placeholder.error(answer)
                    sources = []
                    status.update(label="⚠️ Error", state="error")

        else:
            with st.status(STEP_LABELS["start"], expanded=True) as status:

                status.update(label=STEP_LABELS["kb"], state="running")
                kb_results = search_knowledge(question)

                status.update(label=STEP_LABELS["wiki"], state="running")
                wiki_results = search_wikipedia(question)

                status.update(label=STEP_LABELS["web"], state="running")
                web_results = search_web(question)

                combined_results = kb_results + wiki_results + web_results

                if not combined_results:
                    answer = T["outside"]
                    answer_placeholder.write(answer)
                    sources = []
                    status.update(label="⚠️ No matching sources found", state="error")
                else:
                    status.update(label=STEP_LABELS["writing"], state="running")

                    messages = build_answer_messages(
                        question, combined_results, history,
                        user_name=chat.get("user_name"), user_age=chat.get("user_age")
                    )
                    try:
                        answer = ""
                        for chunk in stream_groq(messages, MODEL):
                            answer += chunk
                            answer_placeholder.markdown(answer)

                        sources = sorted(set(
                            item["source"].split(":")[0].strip() for item in combined_results
                        ))
                        status.update(label=STEP_LABELS["done"], state="complete")
                    except Exception as e:
                        logger.exception("Text generation error")
                        answer = T["error_occurred"].format(error=str(e))
                        answer_placeholder.error(answer)
                        sources = []
                        status.update(label="⚠️ Error", state="error")

        if sources:
            render_source_badges(sources, sources_placeholder)

    chat["messages"].append({
        "role": "assistant", "content": answer, "sources": sources,
        "rating": None, "timestamp": get_timestamp()
    })
    chat["last_updated"] = get_timestamp()


# ============================================================
# WORKOUT PLAN / GROCERY LIST GENERATORS
# ============================================================

def generate_workout_plan(goal, days_per_week, equipment, level):
    prompt = (
        "Create a {days}-day-per-week workout plan for a {level} level person "
        "whose goal is {goal}. Available equipment: {equipment}. "
        "Format it as a clean day-by-day breakdown with exercises, sets, and reps. "
        "Include a brief warm-up note and a safety reminder. Keep it practical."
    ).format(days=days_per_week, level=level, goal=goal, equipment=equipment)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional, safety-conscious fitness coach. "
                    "Write the plan with real energy and belief in the client — "
                    "an intro line that builds motivation, not just a bare table, "
                    "and a closing line that pushes them toward day one. Keep the "
                    "actual exercise data precise and practical; energy lives in "
                    "the framing, not in padding or hype for its own sake."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.5, max_tokens=1400
    )
    return response.choices[0].message.content


def generate_grocery_list(context_text):
    prompt = (
        "Based on this nutrition target/context, create a categorized grocery list "
        "(Produce, Protein, Dairy, Pantry & Grains, Other) for one week of meals:\n\n"
        + context_text
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful, practical nutrition assistant. Frame the "
                    "list with a short, encouraging line about fueling this "
                    "week's training well — then keep the list itself tight and "
                    "scannable. Energy in the framing, precision in the list."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.5, max_tokens=900
    )
    return response.choices[0].message.content


# ============================================================
# WEEKLY RECAP GENERATOR
# ============================================================

def _last_n_days_entries(entries, days=7, date_key="date"):
    if not entries:
        return []
    cutoff = datetime.now().date()
    result = []
    for entry in entries:
        raw_date = entry.get(date_key)
        if not raw_date:
            continue
        try:
            entry_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
        except Exception:
            continue
        if (cutoff - entry_date).days <= days:
            result.append(entry)
    return result


def build_weekly_recap_context(profile):
    tracking = profile.get("tracking", {})

    workouts = _last_n_days_entries(tracking.get("workout_log", []))
    weigh_ins = _last_n_days_entries(tracking.get("weight_log", []))
    meals = _last_n_days_entries(tracking.get("meal_log", []))
    measurements = _last_n_days_entries(tracking.get("measurements_log", []))
    streak = tracking.get("streak", {})
    prs = tracking.get("prs", {})

    recent_prs = {
        lift: data for lift, data in prs.items()
        if data.get("date") and (datetime.now().date() - datetime.strptime(data["date"], "%Y-%m-%d").date()).days <= 7
    }

    lines = []
    lines.append("Workouts logged this week: {}".format(len(workouts)))
    for w in workouts:
        note = " - {}".format(w.get("note")) if w.get("note") else ""
        lines.append("  - {}: {}{}".format(w.get("date"), w.get("type"), note))

    lines.append("Weigh-ins this week: {}".format(len(weigh_ins)))
    for w in weigh_ins:
        lines.append("  - {}: {} kg".format(w.get("date"), w.get("weight_kg")))

    lines.append("New PRs this week: {}".format(len(recent_prs)))
    for lift, data in recent_prs.items():
        lines.append("  - {}: {} {} x {}".format(lift, data.get("weight"), data.get("unit"), data.get("reps")))

    lines.append("Meals/photos analyzed this week: {}".format(len(meals)))

    lines.append("Measurement check-ins this week: {}".format(len(measurements)))

    lines.append("Current streak: {} days (longest ever: {} days)".format(
        streak.get("current", 0), streak.get("longest", 0)
    ))

    return "\n".join(lines), {
        "workouts": len(workouts), "weigh_ins": len(weigh_ins),
        "new_prs": len(recent_prs), "meals": len(meals), "measurements": len(measurements)
    }


def generate_weekly_recap(profile):
    context_text, stats = build_weekly_recap_context(profile)
    name = profile.get("name") or "there"

    prompt = (
        "Here is this user's raw fitness activity data from the last 7 days:\n\n"
        + context_text +
        "\n\nWrite a short, motivating weekly recap addressed to {name}. "
        "Structure it as:\n"
        "1) A one-line headline for the week\n"
        "2) 'What you did' - a few bullet highlights from the data above\n"
        "3) 'Consistency' - one line on their streak\n"
        "4) 'Next week' - 1-2 concrete, realistic suggestions\n"
        "If there is very little or no data, gently encourage them to log a "
        "workout or weigh-in so future recaps have something to celebrate - "
        "do not invent activity that isn't in the data."
    ).format(name=name)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an encouraging, honest fitness coach writing a "
                        "weekly recap. Only reference activity actually present "
                        "in the data provided. Never fabricate workouts, numbers, "
                        "or PRs. Keep it warm but concise."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.6, max_tokens=700
        )
        recap_text = response.choices[0].message.content
    except Exception as e:
        logger.error("Weekly recap error: %s", e)
        recap_text = "Could not generate a recap right now: {}".format(e)

    tracking = profile.setdefault("tracking", json.loads(json.dumps(DEFAULT_TRACKING)))
    recap_entry = {
        "date": get_date_str(), "text": recap_text, "stats": stats, "timestamp": get_timestamp()
    }
    tracking.setdefault("weekly_recaps", []).append(recap_entry)
    save_profile(profile)

    return recap_entry


# ============================================================
# PDF PROGRESS REPORT
# ============================================================

def generate_progress_report_pdf(profile):
    if not FPDF_AVAILABLE:
        return None

    tracking = profile.get("tracking", {})
    weight_log = tracking.get("weight_log", [])
    workout_log = tracking.get("workout_log", [])
    prs = tracking.get("prs", {})
    streak = tracking.get("streak", {})

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Fitness Progress Report", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, "Name: {}   Generated: {}".format(
        profile.get("name") or "-", datetime.now().strftime("%Y-%m-%d")
    ), ln=True)
    pdf.ln(4)

    if MATPLOTLIB_AVAILABLE and len(weight_log) >= 2:
        try:
            dates = [w["date"] for w in weight_log]
            weights = [w["weight_kg"] for w in weight_log]

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(dates, weights, marker="o", color="#6366f1")
            ax.set_title("Weight Trend (kg)")
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            fig.tight_layout()

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=150)
            plt.close(fig)
            buf.seek(0)

            pdf.image(buf, w=170)
            pdf.ln(4)
        except Exception as e:
            logger.error("Chart embed error: %s", e)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Personal Records", ln=True)
    pdf.set_font("Helvetica", "", 10)

    if prs:
        for lift, data in prs.items():
            pdf.cell(0, 6, "{}: {} {} x {} ({})".format(
                lift.title(), data.get("weight"), data.get("unit", "kg"),
                data.get("reps", 1), data.get("date", "")
            ), ln=True)
    else:
        pdf.cell(0, 6, "No PRs logged yet.", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Consistency", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Current streak: {} days | Longest streak: {} days".format(
        streak.get("current", 0), streak.get("longest", 0)
    ), ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Recent Workouts", ln=True)
    pdf.set_font("Helvetica", "", 10)

    if workout_log:
        for w in list(reversed(workout_log))[:10]:
            note = " - {}".format(w.get("note")) if w.get("note") else ""
            pdf.multi_cell(0, 6, "{}: {}{}".format(w.get("date", ""), w.get("type", ""), note))
    else:
        pdf.cell(0, 6, "No workouts logged yet.", ln=True)

    return bytes(pdf.output())


# ============================================================
# CHAT MANAGEMENT
# ============================================================

def new_chat():
    chat = make_new_chat_dict()
    st.session_state.conversations[chat["id"]] = chat
    st.session_state.current_chat = chat["id"]


def delete_chat():
    current = st.session_state.current_chat

    if current in st.session_state.conversations:
        chat = st.session_state.conversations.pop(current)
        try:
            chat_file = CHAT_HISTORY_DIR / "chat_{}.json".format(current)
            with open(chat_file, "w", encoding="utf-8") as f:
                json.dump(chat, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Chat history error: %s", e)

    if not st.session_state.conversations:
        new_chat()
    else:
        st.session_state.current_chat = list(st.session_state.conversations.keys())[0]


def export_chat(chat_id):
    chat = st.session_state.conversations.get(chat_id)
    if chat:
        return json.dumps(chat, ensure_ascii=False, indent=2)
    return None


# ============================================================
# SUMMARY
# ============================================================

def create_summary(messages):
    if not messages:
        return "No conversation history."

    recent = messages[-20:]
    conversation = "\n".join(
        "{}: {}".format(m["role"], m.get("content", "")) for m in recent
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
""".format(conversation)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a professional fitness conversation summarizer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3, max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Summary error: %s", e)
        return "Error generating summary."


# ============================================================
# GO PRO PAGE
# ============================================================

def render_pro_page():
    st.subheader("⭐ {}".format(T["pro_title"]))
    st.caption(T["pro_subtitle"])

    if st.button(T["pro_back"]):
        st.session_state.show_pro_page = False
        st.rerun()

    st.write("")

    profile = st.session_state.profile
    current_sub = profile.get("subscription")
    current_plan_id = current_sub.get("plan_id") if current_sub else None

    lang = st.session_state.language
    cols = st.columns(3)

    for col, plan in zip(cols, PRO_PLANS):
        with col:
            plan_name = plan["name_ar"] if lang == "العربية" else plan["name_en"]
            card_class = "pricing-card popular" if plan["popular"] else "pricing-card"
            badge_html = (
                '<div class="pricing-badge">{}</div>'.format(T["pro_popular"])
                if plan["popular"] else ""
            )

            card_html = (
                '<div class="{card_class}">{badge}'
                '<div class="pricing-name">{name}</div>'
                '<div class="pricing-price">{price}</div>'
                '<div class="pricing-unit">{unit}</div>'
                '{active_badge}'
                '</div>'
            ).format(
                card_class=card_class,
                badge=badge_html,
                name=plan_name,
                price=plan["price"],
                unit=T["pro_currency"],
                active_badge=(
                    '<div class="plan-active-badge">{}</div>'.format(
                        T["pro_current_plan"].format(plan=plan_name)
                    ) if current_plan_id == plan["id"] else ""
                )
            )

            st.markdown(card_html, unsafe_allow_html=True)

            max_features = plan["max_features"]
            feature_options = [f["key"] for f in PRO_FEATURE_POOL]

            default_selection = (
                current_sub.get("features", [])
                if current_sub and current_plan_id == plan["id"]
                else feature_options[:max_features] if max_features < len(feature_options) else feature_options
            )
            # Keep only defaults that respect this plan's cap
            default_selection = default_selection[:max_features]

            st.caption(T["pro_choose_features"].format(n=max_features))

            selected = st.multiselect(
                "Perks for {}".format(plan_name),
                options=feature_options,
                default=default_selection,
                format_func=_feature_label,
                max_selections=max_features,
                key="pro_select_{}".format(plan["id"]),
                label_visibility="collapsed"
            )

            subscribe_label = T["pro_subscribe"].format(plan=plan_name)

            if st.button(subscribe_label, use_container_width=True, key="pro_sub_btn_{}".format(plan["id"])):
                if not selected:
                    st.warning(T["pro_pick_warning"])
                else:
                    set_subscription(profile, plan["id"], selected)
                    st.session_state.profile = profile
                    st.success(T["pro_subscribed_success"].format(plan=plan_name))
                    st.balloons()
                    st.rerun()

    st.divider()

    if current_sub:
        active_plan = _plan_by_id(current_sub.get("plan_id"))
        active_name = (
            (active_plan["name_ar"] if lang == "العربية" else active_plan["name_en"])
            if active_plan else current_sub.get("plan_id")
        )
        st.info(T["pro_active_since"].format(date=current_sub.get("date", "")))
        chosen = current_sub.get("features", [])
        if chosen:
            st.markdown("**{}**".format(active_name))
            for feature_key in chosen:
                st.markdown("- {}".format(_feature_label(feature_key)))

        if st.button(T["pro_cancel"], key="pro_cancel_btn"):
            cancel_subscription(profile)
            st.session_state.profile = profile
            st.success(T["pro_cancelled"])
            st.rerun()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_sidebar_header()

    selected_language = st.selectbox(
        T["language"], ["English", "العربية"],
        index=(0 if st.session_state.language == "English" else 1)
    )

    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.rerun()

    st.divider()

    if st.button(T["pro_nav"], use_container_width=True, key="go_pro_nav_btn"):
        st.session_state.show_pro_page = True
        st.rerun()

    if st.session_state.profile.get("subscription"):
        active_plan = _plan_by_id(st.session_state.profile["subscription"].get("plan_id"))
        if active_plan:
            plan_label = (
                active_plan["name_ar"] if st.session_state.language == "العربية" else active_plan["name_en"]
            )
            st.caption(T["pro_current_plan"].format(plan=plan_label))

    st.divider()

    if st.button(T["new"], use_container_width=True):
        new_chat()
        st.session_state.show_pro_page = False
        st.rerun()

    if st.button(T["delete"], use_container_width=True):
        delete_chat()
        st.rerun()

    st.divider()
    st.subheader(T["history"])

    sorted_chats = sorted(
        st.session_state.conversations.items(),
        key=lambda item: item[1].get("last_updated", item[1].get("created_at", "")),
        reverse=True
    )

    for chat_id, chat_item in sorted_chats:
        title = chat_item.get("title", "Chat")
        if title == "New Chat":
            title = "Chat {}".format(chat_id)

        if st.button("💬 {}".format(title[:25]), key="history_{}".format(chat_id), use_container_width=True):
            st.session_state.current_chat = chat_id
            st.session_state.show_pro_page = False
            st.rerun()

    # ------------------------------------------------------
    # ADMIN PANEL (password protected)
    # Locks: Custom Instructions editing, Knowledge Base
    # visibility/reload, and the escalated-chats overview.
    # ------------------------------------------------------
    st.divider()

    with st.expander(T["admin_panel"], expanded=st.session_state.admin_authenticated):

        if not st.session_state.admin_authenticated:

            st.caption(T["admin_locked"])

            admin_password_input = st.text_input(
                T["admin_password_label"], type="password", key="admin_password_input"
            )

            if st.button(T["admin_login"], use_container_width=True, key="admin_login_btn"):
                if admin_password_input and admin_password_input == ADMIN_PASSWORD:
                    st.session_state.admin_authenticated = True
                    st.rerun()
                else:
                    st.error(T["admin_wrong_password"])

        else:

            st.success(T["admin_unlocked"])

            if st.button(T["admin_logout"], use_container_width=True, key="admin_logout_btn"):
                st.session_state.admin_authenticated = False
                st.rerun()

            st.divider()

            # ---- Knowledge Base (admin-only visibility + reload) ----
            st.subheader(T["knowledge"])

            pdf_files = list(KNOWLEDGE_DIR.glob("*.pdf"))

            if pdf_files:
                st.success("{} PDF(s) found".format(len(pdf_files)))
                for pdf in pdf_files:
                    st.write("📄 {}".format(pdf.name))
                st.write("Knowledge chunks: {}".format(len(st.session_state.documents)))
            else:
                st.warning(T["empty"])

            if st.button(T["reload"], use_container_width=True, key="kb_reload_btn"):
                (
                    st.session_state.documents,
                    st.session_state.vectorizer,
                    st.session_state.matrix
                ) = build_index()
                st.success("Knowledge Base reloaded.")
                st.rerun()

            st.caption("🌐 Wikipedia search: enabled")
            st.caption("🌐 Web search: {}".format("enabled" if DDGS else "package unavailable"))

            st.divider()

            # ---- Custom Instructions (admin-only) ----
            st.subheader(T["instructions"])

            instructions_input = st.text_area(
                "Instructions", value=st.session_state.custom_instructions,
                placeholder=T["instructions_placeholder"], height=150, label_visibility="collapsed",
                key="admin_instructions_input"
            )

            if st.button(T["save_instructions"], use_container_width=True, key="admin_save_instructions_btn"):
                st.session_state.custom_instructions = instructions_input
                try:
                    with open(INSTRUCTIONS_FILE, "w", encoding="utf-8") as f:
                        f.write(instructions_input)
                    st.success(T["instructions_saved"])
                except Exception as e:
                    st.error(str(e))

            st.divider()

            # ---- Escalated chats overview (admin-only) ----
            st.subheader("👨‍💼 Escalated Chats")

            escalated_chats = [
                c for c in st.session_state.conversations.values() if c.get("escalated")
            ]
            if escalated_chats:
                for c in escalated_chats:
                    st.write("• {} ({})".format(c.get("title", "Chat"), c.get("user_name") or "anonymous"))
            else:
                st.caption("No chats currently escalated.")

    # ------------------------------------------------------
    # FITNESS CALCULATORS
    # ------------------------------------------------------
    st.divider()
    with st.expander(T["calculators"]):

        calc_profile = st.session_state.profile.get("tracking", {}).get("calc_profile", {})

        c_weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0,
                                    value=float(calc_profile.get("weight_kg", 70.0)), step=0.5, key="calc_weight")
        c_height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0,
                                    value=float(calc_profile.get("height_cm", 170.0)), step=0.5, key="calc_height")
        c_sex = st.selectbox("Sex (for calorie calc)", ["male", "female"],
                              index=0 if calc_profile.get("sex", "male") == "male" else 1, key="calc_sex")
        c_activity = st.selectbox(
            "Activity level",
            ["sedentary", "light", "moderate", "active", "very_active"],
            index=["sedentary", "light", "moderate", "active", "very_active"].index(
                calc_profile.get("activity_level", "moderate")
            ),
            key="calc_activity"
        )
        c_goal = st.selectbox("Goal", ["maintain", "cut", "bulk"],
                               index=["maintain", "cut", "bulk"].index(calc_profile.get("goal", "maintain")),
                               key="calc_goal")

        if st.button("Calculate", use_container_width=True, key="calc_button"):
            current_chat = st.session_state.conversations[st.session_state.current_chat]
            age_for_calc = current_chat.get("user_age") or 25

            bmi, bmi_category = calculate_bmi(c_weight, c_height)
            tdee = calculate_tdee(c_weight, c_height, age_for_calc, c_sex, c_activity)
            macros = calculate_macros(tdee, c_weight, c_goal)

            st.session_state.profile["tracking"]["calc_profile"] = {
                "weight_kg": c_weight, "height_cm": c_height, "sex": c_sex,
                "activity_level": c_activity, "goal": c_goal
            }
            save_profile(st.session_state.profile)

            st.success("BMI: {} ({})".format(bmi, bmi_category))
            st.info(
                "TDEE: ~{} kcal/day\n\nFor your **{}** goal:\n"
                "- Calories: ~{} kcal\n- Protein: {} g\n- Carbs: {} g\n- Fat: {} g".format(
                    tdee, c_goal, macros["calories"], macros["protein_g"], macros["carbs_g"], macros["fat_g"]
                )
            )

    # ------------------------------------------------------
    # PROGRESS TRACKING
    # ------------------------------------------------------
    with st.expander(T["progress_tracking"]):

        tracking = st.session_state.profile.setdefault("tracking", json.loads(json.dumps(DEFAULT_TRACKING)))
        streak = tracking.get("streak", {"current": 0, "longest": 0})

        st.markdown(
            '<div class="streak-badge">🔥 {}-day streak (best: {})</div>'.format(
                streak.get("current", 0), streak.get("longest", 0)
            ),
            unsafe_allow_html=True
        )
        st.write("")

        log_weight_val = st.number_input("Log today's weight (kg)", min_value=20.0, max_value=300.0,
                                          value=70.0, step=0.1, key="log_weight_input")
        if st.button("Log Weight", use_container_width=True, key="log_weight_btn"):
            log_weight(st.session_state.profile, log_weight_val)
            st.success("Weight logged.")
            st.rerun()

        workout_note = st.text_input("Log a workout (e.g. 'Leg day')", key="log_workout_input")
        if st.button("Log Workout", use_container_width=True, key="log_workout_btn"):
            if workout_note.strip():
                before = tracking.get("streak", {}).get("current", 0)
                new_streak = log_workout(st.session_state.profile, workout_note.strip())
                if new_streak["current"] in STREAK_MILESTONES and new_streak["current"] != before:
                    st.balloons()
                    st.success("🎉 {}-day streak milestone!".format(new_streak["current"]))
                else:
                    st.success("Workout logged.")
                st.rerun()

        weight_log = tracking.get("weight_log", [])
        if len(weight_log) >= 2:
            df = pd.DataFrame(weight_log)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")[["weight_kg"]]
            st.line_chart(df)
        elif weight_log:
            st.caption("Log at least 2 weigh-ins to see a trend chart.")

    # ------------------------------------------------------
    # BODY MEASUREMENTS & PROGRESS PHOTOS
    # ------------------------------------------------------
    with st.expander(T["measurements_photos"]):

        tracking = st.session_state.profile.setdefault("tracking", json.loads(json.dumps(DEFAULT_TRACKING)))

        st.caption("Log measurements (leave a field at 0 to skip it).")

        meas_values = {}
        meas_cols = st.columns(2)
        for idx, (field_key, field_label) in enumerate(MEASUREMENT_FIELDS):
            with meas_cols[idx % 2]:
                meas_values[field_key] = st.number_input(
                    field_label, min_value=0.0, max_value=250.0, value=0.0, step=0.5,
                    key="meas_{}".format(field_key)
                )

        meas_note = st.text_input("Note (optional)", key="meas_note_input")

        if st.button("Log Measurements", use_container_width=True, key="meas_log_btn"):
            entry = log_measurements(st.session_state.profile, meas_values, note=meas_note.strip())
            if len(entry) <= 2:
                st.warning("Enter at least one measurement above zero.")
            else:
                st.success("Measurements logged.")
                st.rerun()

        measurements_log = tracking.get("measurements_log", [])
        if len(measurements_log) >= 2:
            available_fields = sorted({
                key for entry in measurements_log for key in entry
                if key not in ("date", "note")
            })
            if available_fields:
                trend_field = st.selectbox(
                    "Show trend for", available_fields,
                    format_func=lambda k: dict(MEASUREMENT_FIELDS).get(k, k),
                    key="meas_trend_field"
                )
                trend_rows = [e for e in measurements_log if trend_field in e]
                if len(trend_rows) >= 2:
                    mdf = pd.DataFrame(trend_rows)
                    mdf["date"] = pd.to_datetime(mdf["date"])
                    mdf = mdf.set_index("date")[[trend_field]]
                    st.line_chart(mdf)

        st.divider()
        st.caption("Upload a progress photo.")

        photo_upload = st.file_uploader(
            "Progress photo", type=["jpg", "jpeg", "png", "webp"],
            key="progress_photo_uploader", label_visibility="collapsed"
        )
        photo_note = st.text_input("Photo note (optional)", key="photo_note_input")

        if st.button("Save Photo", use_container_width=True, key="save_photo_btn"):
            if photo_upload is not None:
                photo_bytes = photo_upload.getvalue()
                mime_type = photo_upload.type or "image/jpeg"
                entry = save_progress_photo(
                    st.session_state.profile, photo_bytes, mime_type, note=photo_note.strip()
                )
                if entry:
                    st.success("Photo saved.")
                    st.rerun()
                else:
                    st.error("Could not save photo.")
            else:
                st.warning("Choose a photo to upload first.")

        photos = sorted(
            tracking.get("progress_photos", []), key=lambda p: p.get("date", ""), reverse=True
        )

        if photos:
            st.caption("Gallery ({} photo(s))".format(len(photos)))
            for photo in photos[:12]:
                photo_path = PROGRESS_PHOTOS_DIR / photo["filename"]
                if photo_path.exists():
                    gallery_cols = st.columns([2, 1])
                    with gallery_cols[0]:
                        st.image(str(photo_path), caption="{}  {}".format(
                            photo.get("date", ""), photo.get("note", "")
                        ), use_container_width=True)
                    with gallery_cols[1]:
                        if st.button("🗑️", key="del_photo_{}".format(photo["filename"])):
                            delete_progress_photo(st.session_state.profile, photo["filename"])
                            st.rerun()
        else:
            st.caption("No progress photos yet.")

    # ------------------------------------------------------
    # PERSONAL RECORDS
    # ------------------------------------------------------
    with st.expander(T["personal_records"]):

        common_lifts = ["Squat", "Bench Press", "Deadlift", "Overhead Press", "Custom"]
        pr_lift_choice = st.selectbox("Lift", common_lifts, key="pr_lift_choice")

        pr_lift_name = (
            st.text_input("Custom lift name", key="pr_custom_lift")
            if pr_lift_choice == "Custom" else pr_lift_choice
        )

        pr_col1, pr_col2 = st.columns(2)
        with pr_col1:
            pr_weight = st.number_input("Weight", min_value=0.0, max_value=1000.0, step=0.5, key="pr_weight_input")
        with pr_col2:
            pr_unit = st.selectbox("Unit", ["kg", "lb"], key="pr_unit_input")

        pr_reps = st.number_input("Reps", min_value=1, max_value=20, value=1, step=1, key="pr_reps_input")

        if st.button("Log PR", use_container_width=True, key="pr_log_btn"):
            if pr_lift_name and pr_lift_name.strip():
                is_new = log_pr(
                    st.session_state.profile, pr_lift_name.strip().lower(),
                    pr_weight, pr_unit, int(pr_reps)
                )
                if is_new:
                    st.balloons()
                    st.success("🎉 New PR for {}!".format(pr_lift_name))
                else:
                    st.info("Logged, but not a new best.")
                st.rerun()

        prs = st.session_state.profile.get("tracking", {}).get("prs", {})
        if prs:
            for lift, data in prs.items():
                st.markdown(
                    '<div class="pr-card"><b>{}</b>: {} {} × {} <br><small>{}</small></div>'.format(
                        lift.title(), data.get("weight"), data.get("unit", "kg"),
                        data.get("reps", 1), data.get("date", "")
                    ),
                    unsafe_allow_html=True
                )
        else:
            st.caption("No PRs logged yet.")

    # ------------------------------------------------------
    # WORKOUT PLAN GENERATOR
    # ------------------------------------------------------
    with st.expander(T["workout_plan"]):

        wp_goal = st.selectbox(
            "Goal", ["general fitness", "fat loss", "muscle gain", "strength"], key="wp_goal"
        )
        wp_days = st.slider("Days per week", 2, 6, 3, key="wp_days")
        wp_equipment = st.selectbox(
            "Equipment", ["bodyweight only", "dumbbells", "full gym"], key="wp_equipment"
        )
        wp_level = st.selectbox(
            "Experience level", ["beginner", "intermediate", "advanced"], key="wp_level"
        )

        if st.button("Generate Plan", use_container_width=True, key="wp_generate_btn"):
            with st.spinner("Building your plan..."):
                try:
                    plan_text = generate_workout_plan(wp_goal, wp_days, wp_equipment, wp_level)
                    st.session_state.profile["saved_plan"] = {
                        "goal": wp_goal, "days_per_week": wp_days, "equipment": wp_equipment,
                        "level": wp_level, "plan": plan_text, "created_at": get_timestamp()
                    }
                    save_profile(st.session_state.profile)
                    st.success("Plan generated and saved.")
                except Exception as e:
                    st.error("Could not generate plan: {}".format(e))

        saved_plan = st.session_state.profile.get("saved_plan")
        if saved_plan:
            with st.container(border=True):
                st.caption("Saved plan ({})".format(saved_plan.get("created_at", "")[:10]))
                st.markdown(saved_plan.get("plan", ""))

    # ------------------------------------------------------
    # GROCERY LIST GENERATOR
    # ------------------------------------------------------
    with st.expander(T["grocery_list"]):

        st.caption("Uses your saved calculator macros or workout plan goal as context.")

        if st.button("Generate Grocery List", use_container_width=True, key="grocery_generate_btn"):
            calc_profile = st.session_state.profile.get("tracking", {}).get("calc_profile", {})
            saved_plan = st.session_state.profile.get("saved_plan")

            if calc_profile:
                context_text = "Daily macro target: {}".format(calc_profile)
            elif saved_plan:
                context_text = "Fitness goal: {}, {} days/week".format(
                    saved_plan.get("goal"), saved_plan.get("days_per_week")
                )
            else:
                context_text = "General healthy balanced eating for an active adult."

            with st.spinner("Building your grocery list..."):
                try:
                    grocery_text = generate_grocery_list(context_text)
                    st.session_state.profile["grocery_list"] = {
                        "text": grocery_text, "created_at": get_timestamp()
                    }
                    save_profile(st.session_state.profile)
                except Exception as e:
                    st.error("Could not generate list: {}".format(e))

        grocery_list = st.session_state.profile.get("grocery_list")
        if grocery_list:
            st.markdown(grocery_list.get("text", ""))
            st.download_button(
                "📥 Download list", data=grocery_list.get("text", ""),
                file_name="grocery_list.txt", mime="text/plain",
                use_container_width=True, key="grocery_download_btn"
            )

    # ------------------------------------------------------
    # WEEKLY RECAP
    # ------------------------------------------------------
    with st.expander(T["weekly_recap"]):

        st.caption("A short AI-generated summary of your last 7 days, built only from what you've logged.")

        if st.button("Generate Weekly Recap", use_container_width=True, key="weekly_recap_btn"):
            with st.spinner("Putting your week together..."):
                try:
                    recap_entry = generate_weekly_recap(st.session_state.profile)
                    st.rerun()
                except Exception as e:
                    st.error("Could not generate recap: {}".format(e))

        recaps = st.session_state.profile.get("tracking", {}).get("weekly_recaps", [])
        if recaps:
            latest = recaps[-1]
            with st.container(border=True):
                st.caption("Generated {}".format(latest.get("date", "")))
                st.markdown(latest.get("text", ""))

            if len(recaps) > 1:
                with st.expander("Past recaps"):
                    for recap in list(reversed(recaps[:-1]))[:5]:
                        st.caption(recap.get("date", ""))
                        st.markdown(recap.get("text", ""))
                        st.divider()
        else:
            st.caption("No recap generated yet.")

    # ------------------------------------------------------
    # PROGRESS REPORT (PDF)
    # ------------------------------------------------------
    with st.expander(T["progress_report"]):

        if not FPDF_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            st.warning(
                "This feature needs two extra packages. Run:\n\n"
                "pip install fpdf2 matplotlib\n\nthen restart the app."
            )
        else:
            if st.button("Generate PDF Report", use_container_width=True, key="report_generate_btn"):
                pdf_bytes = generate_progress_report_pdf(st.session_state.profile)
                if pdf_bytes:
                    st.download_button(
                        "📥 Download Report", data=pdf_bytes,
                        file_name="fitness_progress_report.pdf", mime="application/pdf",
                        use_container_width=True, key="report_download_btn"
                    )

    st.divider()

    current_chat = st.session_state.conversations[st.session_state.current_chat]

    if current_chat.get("user_name"):
        st.caption(T["profile_signed_in"].format(
            name=current_chat["user_name"], age=current_chat["user_age"]
        ))

        if st.button(T["profile_reset"], use_container_width=True):
            # Signs this browser session out so a different person can
            # onboard next. Their saved name/age/tracking history is
            # kept on file — it'll be there again if they come back.
            st.session_state.session_profile_confirmed = False
            st.session_state.session_user_name = None
            st.session_state.session_user_age = None
            st.session_state.profile = _default_profile()
            current_chat["user_name"] = None
            current_chat["user_age"] = None
            current_chat["onboarded"] = False
            st.rerun()

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button(T["human"], use_container_width=True):
            current_chat["escalated"] = True
            st.info(T["escalation_requested"])

    with col2:
        if st.button(T["export_chat"], use_container_width=True):
            exported = export_chat(st.session_state.current_chat)
            if exported:
                st.download_button(
                    "📥 Download", data=exported, file_name="chat.json", mime="application/json"
                )


# ============================================================
# MAIN HEADER
# ============================================================

render_hero_header()


# ============================================================
# GO PRO PAGE (shown instead of the chat when toggled)
# ============================================================

if st.session_state.show_pro_page:
    render_pro_page()
    st.stop()


# ============================================================
# CURRENT CHAT
# ============================================================

chat = st.session_state.conversations[st.session_state.current_chat]


# ============================================================
# ONBOARDING
# ============================================================

if not chat["onboarded"]:

    st.subheader(T["onboard_title"])

    onboard_name = st.text_input(T["onboard_name"], key="onboard_name_input")

    existing_profile = get_profile_by_name(onboard_name) if onboard_name.strip() else None

    if existing_profile:
        # This name was saved before — recognize them, reuse their saved
        # age and tracking history, and skip asking again.
        st.success(T["onboard_recognized"].format(
            name=existing_profile.get("name"), age=existing_profile.get("age")
        ))
        st.caption(T["onboard_not_you"].format(name=existing_profile.get("name")))

        if st.button(T["onboard_continue"].format(name=existing_profile.get("name")), use_container_width=True):
            chat["user_name"] = existing_profile.get("name")
            chat["user_age"] = existing_profile.get("age")
            chat["onboarded"] = True

            st.session_state.session_profile_confirmed = True
            st.session_state.session_user_name = chat["user_name"]
            st.session_state.session_user_age = chat["user_age"]
            st.session_state.profile = existing_profile

            greeting = T["onboard_greeting_returning"].format(name=chat["user_name"])
            chat["messages"].append({
                "role": "assistant", "content": greeting, "timestamp": get_timestamp()
            })

            st.rerun()

    else:
        onboard_age = st.number_input(T["onboard_age"], min_value=1, max_value=120, step=1, key="onboard_age_input")

        if st.button(T["onboard_button"], use_container_width=True):
            if onboard_name.strip():
                chat["user_name"] = onboard_name.strip()
                chat["user_age"] = int(onboard_age)
                chat["onboarded"] = True

                # Remember this person for the rest of THIS browser session,
                # and save them by name so they're recognized next time
                # they type the same name — without asking a different
                # visitor's saved data to any other name.
                st.session_state.session_profile_confirmed = True
                st.session_state.session_user_name = chat["user_name"]
                st.session_state.session_user_age = chat["user_age"]

                new_profile, _was_existing = create_or_load_profile(chat["user_name"], chat["user_age"])
                st.session_state.profile = new_profile

                greeting = T["onboard_greeting"].format(name=chat["user_name"], age=chat["user_age"])
                chat["messages"].append({
                    "role": "assistant", "content": greeting, "timestamp": get_timestamp()
                })

                st.rerun()

    st.stop()


# ============================================================
# MESSAGE DISPLAY
# ============================================================

if not chat["messages"]:
    st.info(T["welcome"])

for msg_index, message in enumerate(chat["messages"]):

    edit_key = "editing_{}_{}".format(st.session_state.current_chat, msg_index)

    with st.chat_message(message["role"]):

        if message.get("image_b64"):
            try:
                image_bytes = base64.b64decode(message["image_b64"])
                st.image(image_bytes, width=300)
            except Exception:
                pass

        if message["role"] == "user":
            st.markdown(
                "<div class='user-message'>{}</div>".format(message.get("content", "")),
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='assistant-message'>{}</div>".format(message.get("content", "")),
                unsafe_allow_html=True
            )

        if message["role"] == "assistant" and message.get("sources"):
            render_source_badges(message["sources"])

        if not st.session_state.get(edit_key, False):

            action_cols = st.columns([1, 1, 1, 1, 1, 6])

            with action_cols[0]:
                if message["role"] == "user":
                    if st.button("✏️", key="edit_{}_{}".format(st.session_state.current_chat, msg_index)):
                        st.session_state[edit_key] = True
                        st.rerun()

            with action_cols[1]:
                if st.button("🗑️", key="del_{}_{}".format(st.session_state.current_chat, msg_index)):
                    chat["messages"].pop(msg_index)
                    st.rerun()

            if message["role"] == "assistant":
                rating = message.get("rating")

                with action_cols[2]:
                    if st.button("👍" if rating != "up" else "✅👍",
                                 key="up_{}_{}".format(st.session_state.current_chat, msg_index)):
                        message["rating"] = "up"
                        st.rerun()

                with action_cols[3]:
                    if st.button("👎" if rating != "down" else "✅👎",
                                 key="down_{}_{}".format(st.session_state.current_chat, msg_index)):
                        message["rating"] = "down"
                        st.rerun()

                with action_cols[4]:
                    is_last_assistant = (
                        msg_index == len(chat["messages"]) - 1
                    )
                    if is_last_assistant:
                        if st.button("🔄", key="regen_{}_{}".format(st.session_state.current_chat, msg_index)):
                            prior_user_msg = None
                            for prev in reversed(chat["messages"][:msg_index]):
                                if prev["role"] == "user":
                                    prior_user_msg = prev
                                    break

                            if prior_user_msg:
                                chat["messages"].pop(msg_index)
                                generate_response(
                                    chat, prior_user_msg["content"],
                                    image_b64=prior_user_msg.get("image_b64"),
                                    mime_type=prior_user_msg.get("mime_type")
                                )
                                st.rerun()


# ============================================================
# CHAT INPUT + IMAGE / PDF UPLOAD
# ============================================================

image_purpose_labels = {
    "general": "💬 General question",
    "meal": "🍽️ Meal calorie estimate",
    "form": "🏋️ Exercise form check"
}

composer_cols = st.columns([2, 3])
with composer_cols[0]:
    st.markdown(
        "<div style='font-size:0.82rem; font-weight:700; color:#ffd7c9; "
        "margin-top:8px;'>📎 If you attach a photo, treat it as</div>",
        unsafe_allow_html=True
    )
with composer_cols[1]:
    st.session_state.image_purpose = st.selectbox(
        "If you attach a photo, what should I do with it?",
        options=list(image_purpose_labels.keys()),
        format_func=lambda key: image_purpose_labels[key],
        index=list(image_purpose_labels.keys()).index(st.session_state.image_purpose),
        key="image_purpose_select",
        label_visibility="collapsed"
    )

prompt = st.chat_input(
    "Ask your fitness question, or attach an image/PDF...",
    accept_file=True,
    file_type=["jpg", "jpeg", "png", "webp", "pdf"],
    max_upload_size=30
)

if prompt:

    question = prompt.text.strip() if prompt.text else ""
    uploaded_files = prompt.files if prompt.files else []
    uploaded_file = uploaded_files[0] if uploaded_files else None

    is_pdf = uploaded_file is not None and uploaded_file.name.lower().endswith(".pdf")
    is_image = uploaded_file is not None and not is_pdf

    # --------------------------------------------------------
    # PDF upload -> add to Knowledge Base
    # --------------------------------------------------------
    if is_pdf:

        pdf_bytes = uploaded_file.getvalue()

        if len(pdf_bytes) > MAX_PDF_SIZE:
            st.error("PDF is too large. Maximum size is 30 MB.")
            st.stop()

        save_path = KNOWLEDGE_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(pdf_bytes)

        (
            st.session_state.documents,
            st.session_state.vectorizer,
            st.session_state.matrix
        ) = build_index()

        user_note = question if question else "📄 Uploaded: {}".format(uploaded_file.name)

        chat["messages"].append({
            "role": "user", "content": user_note, "timestamp": get_timestamp()
        })

        if chat["title"] == "New Chat":
            chat["title"] = user_note[:40]

        with st.chat_message("user"):
            st.markdown("<div class='user-message'>{}</div>".format(user_note), unsafe_allow_html=True)

        confirmation = "✅ Added **{}** to the Knowledge Base ({} chunks total).".format(
            uploaded_file.name, len(st.session_state.documents)
        )

        chat["messages"].append({
            "role": "assistant", "content": confirmation, "sources": ["Knowledge Base"],
            "rating": None, "timestamp": get_timestamp()
        })

        with st.chat_message("assistant"):
            st.success(confirmation)

        if question:
            try:
                generate_response(chat, question)
            except Exception:
                logger.exception("Response generation error after PDF upload")
                st.error("An error occurred while generating the response.")

        st.rerun()

    # --------------------------------------------------------
    # Image upload -> vision (general / meal / form check)
    # --------------------------------------------------------
    elif is_image:

        image_bytes = uploaded_file.getvalue()

        if len(image_bytes) > MAX_IMAGE_SIZE:
            st.error("Image is too large. Maximum size is 20 MB.")
            st.stop()

        mime_type = uploaded_file.type or "image/jpeg"

        if mime_type not in SUPPORTED_IMAGE_TYPES:
            st.error("Please upload JPG, PNG, or WEBP.")
            st.stop()

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        if not question:
            question = "Please analyze this image and tell me what you can observe."

        chat["messages"].append({
            "role": "user", "content": question, "image_b64": image_b64,
            "mime_type": mime_type, "timestamp": get_timestamp()
        })

        if chat["title"] == "New Chat":
            chat["title"] = question[:40]

        with st.chat_message("user"):
            st.image(image_bytes, width=300)
            st.markdown("<div class='user-message'>{}</div>".format(question), unsafe_allow_html=True)

        try:
            generate_response(
                chat, question, image_b64=image_b64, mime_type=mime_type,
                vision_mode=st.session_state.image_purpose
            )
        except Exception:
            logger.exception("Image response error")
            st.error("An error occurred while analyzing the image.")

        st.rerun()

    # --------------------------------------------------------
    # Text-only message
    # --------------------------------------------------------
    elif question:

        if contains_sensitive_data(question):
            chat["messages"].append({"role": "user", "content": question, "timestamp": get_timestamp()})
            chat["messages"].append({
                "role": "assistant", "content": T["sensitive"], "timestamp": get_timestamp()
            })
            st.rerun()

        chat["messages"].append({"role": "user", "content": question, "timestamp": get_timestamp()})

        if chat["title"] == "New Chat":
            chat["title"] = question[:40]

        with st.chat_message("user"):
            st.markdown("<div class='user-message'>{}</div>".format(question), unsafe_allow_html=True)

        try:
            generate_response(chat, question)
        except Exception:
            logger.exception("Response generation error")
            st.error("An error occurred while generating the response.")

        st.rerun()


# ============================================================
# SUMMARY
# ============================================================

if chat["messages"]:
    st.divider()

    with st.expander(T["summary"]):
        if st.button("Generate Summary"):
            with st.spinner(T["summary_generating"]):
                summary = create_summary(chat["messages"])
                st.subheader(T["summary_generated"])
                st.markdown(summary)
                chat["summary"] = summary


# ============================================================
# ESCALATION
# ============================================================

if chat["escalated"]:
    st.warning(T["escalation_requested"])


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "<div style='margin-top:18px; padding-top:14px; border-top:1px solid rgba(255,110,90,0.20); "
    "text-align:center; font-size:0.78rem; color:rgba(255,220,220,0.55);'>"
    "Powered by Groq AI &nbsp;&bull;&nbsp; Fitness AI Trainer v2.0"
    "</div>",
    unsafe_allow_html=True
)

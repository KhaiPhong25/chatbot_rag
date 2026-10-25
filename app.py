# =============================================================================
# STREAMLIT WEB UI - Giao diện hỏi đáp Lịch Sử Đảng CSVN
# Tích hợp: Streaming, Hybrid Search, GPU Reranker, Hộp trích dẫn số trang
# Phong cách: Nền trắng tối giản kết hợp các nút bấm màu đen (Monochrome Minimalist)
# =============================================================================

import os
import importlib
import streamlit as st

# Tự động đồng bộ component registry cho Custom Components v2 (Streamlit 1.66+)
try:
    from streamlit.components.v2.get_bidi_component_manager import get_bidi_component_manager
    _cm = get_bidi_component_manager()
    if _cm.get_component_asset_root("streamlit-shadcn-ui.v2") is None:
        _cm.discover_and_register_components(start_file_watching=False)
except Exception:
    pass

import streamlit_shadcn_ui as ui
from streamlit_elements import elements, mui, html

import rag_core
importlib.reload(rag_core)

from rag_core import (
    get_embedding_model,
    get_reranker_model,
    build_or_update_vectorstore,
    build_hybrid_retriever,
    stream_with_fallback,
    FREE_MODELS,
    CHAT_HISTORY_FILE,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CẤU HÌNH TRANG STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hỏi Đáp Lịch Sử Đảng CSVN",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. BỘ PHONG CÁCH CSS NỀN TRẮNG & NÚT BẤM ĐEN
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Light Background */
    html, body, .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #ffffff !important;
        color: #0f172a !important;
    }

    /* Ẩn header & footer mặc định của Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        display: none !important;
    }
    div[data-testid="stDecoration"] {
        display: none !important;
    }

    /* ──────────────── SIDEBAR ──────────────── */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
        padding-top: 1rem;
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #0f172a !important;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.6rem;
        letter-spacing: -0.01em;
    }

    /* Sidebar Brand Card - Cân đối logo và chữ thẳng hàng */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 12px 14px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .sidebar-brand-icon {
        font-size: 1.6rem;
        line-height: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        flex-shrink: 0;
    }
    .sidebar-brand-info {
        display: flex;
        flex-direction: column;
        justify-content: center;
        overflow: hidden;
    }
    .sidebar-brand-title {
        font-size: 0.98rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        line-height: 1.25;
        white-space: nowrap;
    }
    .sidebar-brand-subtitle {
        font-size: 0.74rem;
        color: #64748b;
        margin: 3px 0 0;
        font-weight: 500;
        line-height: 1.2;
        white-space: nowrap;
    }

    /* Style selectbox */
    section[data-testid="stSidebar"] [data-baseweb="select"] {
        border-radius: 10px !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1.5px solid #0f172a !important;
        border-radius: 10px !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
    }

    /* ──────────────── ĐỒNG BỘ TOÀN BỘ NÚT BẤM LÀ MÀU ĐEN ──────────────── */
    .stButton button,
    .stButton > button,
    button[kind="secondary"],
    button[kind="primary"],
    [data-testid="baseButton-secondary"],
    [data-testid="baseButton-primary"] {
        background-color: #0f172a !important;
        background: #0f172a !important;
        color: #ffffff !important;
        border: 1px solid #0f172a !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.08) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stButton button p,
    .stButton button div,
    .stButton button span,
    button[kind="secondary"] p,
    button[kind="primary"] p {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    .stButton button:hover,
    .stButton > button:hover,
    button[kind="secondary"]:hover,
    button[kind="primary"]:hover,
    [data-testid="baseButton-secondary"]:hover,
    [data-testid="baseButton-primary"]:hover {
        background-color: #1e293b !important;
        background: #1e293b !important;
        border-color: #1e293b !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.16) !important;
    }

    .stButton button:active,
    .stButton > button:active,
    button[kind="secondary"]:active,
    button[kind="primary"]:active {
        background-color: #000000 !important;
        transform: translateY(0px) !important;
    }

    /* ──────────────── MAIN CONTAINER & HEADER ──────────────── */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 6rem !important;
        max-width: 950px !important;
        background-color: #ffffff !important;
    }

    /* Top App Banner - Thiết kế thanh lịch, icon và chữ thẳng hàng đều nhau */
    .main-banner {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 24px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        box-sizing: border-box;
    }
    .banner-title {
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 0 0 8px 0;
    }
    .banner-icon {
        font-size: 1.8rem;
        line-height: 1;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 46px;
        height: 46px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        flex-shrink: 0;
    }
    .banner-text {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.02em;
        line-height: 1.25;
    }
    .banner-subtitle {
        color: #475569;
        font-size: 0.92rem;
        margin: 0;
        line-height: 1.6;
        font-weight: 400;
    }

    /* Container card đồng đều cho các mục gợi ý, không bị đè chữ */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        border: 1px solid #e2e8f0 !important;
        background-color: #ffffff !important;
        padding: 12px 14px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        margin-bottom: 8px !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #cbd5e1 !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04) !important;
    }

    /* ──────────────── WELCOME PROMPTS ──────────────── */
    .welcome-intro {
        text-align: center;
        margin: 28px 0 20px;
    }
    .welcome-intro h3 {
        font-size: 1.3rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 6px;
        letter-spacing: -0.01em;
    }
    .welcome-intro p {
        color: #64748b;
        font-size: 0.9rem;
    }

    /* ──────────────── CHAT MESSAGES STYLING ──────────────── */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        padding: 10px 0 !important;
    }

    /* User Message Bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #f1f5f9 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        padding: 16px 20px !important;
        margin-bottom: 14px !important;
        color: #0f172a !important;
    }

    /* Assistant Message Bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: #ffffff !important;
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 16px !important;
        padding: 18px 22px !important;
        margin-bottom: 14px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04) !important;
        color: #0f172a !important;
    }

    /* Model Tag */
    .chat-model-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        color: #0f172a;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 9999px;
        margin-bottom: 10px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ──────────────── HỘP NGUỒN TRÍCH DẪN SỐ TRANG ──────────────── */
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        background-color: #ffffff !important;
        margin-top: 14px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
    }
    div[data-testid="stExpander"] details summary {
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        color: #0f172a !important;
        padding: 8px 12px !important;
    }
    div[data-testid="stExpander"] details summary:hover {
        color: #000000 !important;
    }
    .source-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 8px;
        transition: border-color 0.15s ease;
    }
    .source-card:last-child {
        margin-bottom: 0;
    }
    .source-card:hover {
        border-color: #0f172a;
    }
    .source-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .source-card-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #0f172a;
    }
    .source-card-score {
        font-size: 0.68rem;
        font-weight: 700;
        background: #0f172a;
        color: #ffffff;
        padding: 2px 8px;
        border-radius: 9999px;
    }
    .source-card-body {
        font-size: 0.78rem;
        color: #475569;
        line-height: 1.5;
        border-left: 2.5px solid #cbd5e1;
        padding-left: 10px;
        margin-top: 4px;
    }

    /* ──────────────── CHAT INPUT BAR ──────────────── */
    [data-testid="stChatInput"] {
        background-color: transparent !important;
    }
    [data-testid="stChatInput"] > div {
        background-color: #ffffff !important;
        border: 1.5px solid #0f172a !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08) !important;
        transition: box-shadow 0.2s ease;
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: #000000 !important;
        box-shadow: 0 0 0 3px rgba(15, 23, 42, 0.15) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #0f172a !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #94a3b8 !important;
    }
    /* Send button */
    [data-testid="stChatInput"] button {
        background: #0f172a !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }
    [data-testid="stChatInput"] button:hover {
        background: #334155 !important;
    }

    /* ──────────────── TABS STYLING (MINIMALIST SEGMENTED) ──────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #f1f5f9;
        padding: 5px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 22px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        color: #64748b;
        font-weight: 600;
        font-size: 0.88rem;
        border: none !important;
        background-color: transparent !important;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0f172a;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #ffffff;
    }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. KHỞI TẠO HỆ THỐNG RAG (SINGLETON CACHED: BGE-M3 + FAISS + GPU RERANKER)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🧠 Đang khởi tạo mô hình BAAI/bge-m3 & GPU Reranker...")
def init_rag_system():
    """Tải embedding model BGE-M3 (GPU), FAISS Vectorstore, Hybrid Retriever và GPU Reranker."""
    emb = get_embedding_model()
    vs  = build_or_update_vectorstore(emb)
    ret = build_hybrid_retriever(vs, k=12)
    # Nạp sẵn mô hình Reranker trên GPU để phản hồi ngay lập tức
    get_reranker_model()
    return ret


retriever = init_rag_system()


# ─────────────────────────────────────────────────────────────────────────────
# 4. QUẢN LÝ SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = "web_user"

if "suggested_prompt" not in st.session_state:
    st.session_state.suggested_prompt = None


def render_sources_box(sources: list):
    """Hiển thị hộp Accordion trích dẫn nguồn số trang từ giáo trình."""
    if not sources:
        return
    with st.expander(f"📚 Nguồn trích dẫn ({len(sources)} đoạn trích từ giáo trình)", expanded=False):
        for src in sources:
            page = src.get("page", "?")
            file_name = src.get("file", "textbook_LSD.pdf")
            score_pct = src.get("score_pct", 0)
            snippet = src.get("snippet", "")

            card_html = (
                f'<div class="source-card">'
                f'<div class="source-card-header">'
                f'<span class="source-card-title">📄 <b>Trang {page}</b> · {file_name}</span>'
                f'<span class="source-card-score">Độ khớp: {score_pct}%</span>'
                f'</div>'
                f'<div class="source-card-body">{snippet}</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, description: str, key: str):
    """Hiển thị Metric Card theo chuẩn shadcn/ui với cơ chế graceful fallback."""
    try:
        ui.metric_card(label=label, value=value, description=description, key=key)
    except Exception:
        card_html = (
            f'<div style="border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 16px; '
            f'background: #ffffff; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">'
            f'<div style="font-size: 0.8rem; font-weight: 500; color: #64748b; margin-bottom: 4px;">{label}</div>'
            f'<div style="font-size: 1.45rem; font-weight: 800; color: #0f172a; margin-bottom: 4px; font-family: \'Plus Jakarta Sans\', sans-serif;">{value}</div>'
            f'<div style="font-size: 0.78rem; color: #94a3b8;">{description}</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)


def render_badges(items: list[dict], key: str):
    """Hiển thị Badges theo chuẩn shadcn/ui với cơ chế graceful fallback."""
    try:
        ui.badges(items=items, key=key)
    except Exception:
        badges_html = " ".join([
            f'<span style="display:inline-block; padding: 3px 10px; border-radius: 9999px; font-size: 0.72rem; font-weight: 600; '
            f'background: {"#f1f5f9" if x.get("variant")=="secondary" else "#ffffff"}; '
            f'border: 1px solid #cbd5e1; color: #0f172a; margin-right: 6px; margin-bottom: 6px;">{x.get("text")}</span>'
            for x in items
        ])
        st.markdown(badges_html, unsafe_allow_html=True)


def render_separator(key: str):
    """Hiển thị đường phân cách theo chuẩn shadcn/ui với cơ chế graceful fallback."""
    try:
        ui.separator(key=key)
    except Exception:
        st.markdown("<hr style='border: none; border-top: 1px solid #e2e8f0; margin: 16px 0;' />", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SIDEBAR - CHỌN MÔ HÌNH VỚI TÊN & ƯU ĐIỂM (TÍCH HỢP SHADCN & EXTRAS)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand Header - Logo và chữ căn đối
    sidebar_brand_html = (
        '<div class="sidebar-brand">'
        '<div class="sidebar-brand-icon">🎓</div>'
        '<div class="sidebar-brand-info">'
        '<div class="sidebar-brand-title">Lịch Sử Đảng CSVN</div>'
        '<div class="sidebar-brand-subtitle">Trợ lý học thuật RAG v2.2</div>'
        '</div>'
        '</div>'
    )
    st.markdown(sidebar_brand_html, unsafe_allow_html=True)

    st.markdown("### 🤖 Mô hình Phản Hồi")

    MODEL_OPTIONS = {
        "auto": "⚡ Tự động (Tối ưu)",
        "nvidia/nemotron-3.5-super-vl:free": "Nemotron 3.5 Super VL (Ngữ cảnh 1M)",
        "nvidia/nemotron-3.5-lightning:free": "Nemotron 3.5 Lightning (Tốc độ cao)",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": "Nemotron 30B Reasoning (Suy luận)",
        "xingchen-agi/xing4.0-29b-a4b:free": "XingChen 29B (Tổng hợp)",
        "recursive/flint-v1.0:free": "Recursive Flint v1.0 (Ổn định)",
    }

    selected_model_key = st.selectbox(
        "Chọn mô hình phản hồi:",
        options=list(MODEL_OPTIONS.keys()),
        format_func=lambda k: MODEL_OPTIONS[k],
        index=0,
        label_visibility="collapsed",
    )

    render_separator(key="side_sep_action")

    # Nút xóa lịch sử hội thoại chuẩn giao diện đen tối giản
    if st.button("🗑️ Làm mới cuộc hội thoại", use_container_width=True):
        st.session_state.messages = []
        if os.path.exists(CHAT_HISTORY_FILE):
            try:
                os.remove(CHAT_HISTORY_FILE)
            except Exception:
                pass
        st.session_state.suggested_prompt = None
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# 6. HEADER CHÍNH CỦA TRANG
# ─────────────────────────────────────────────────────────────────────────────
banner_html = (
    '<div class="main-banner">'
    '<div class="banner-title">'
    '<div class="banner-icon">🎓</div>'
    '<div class="banner-text">Hỏi Đáp Giáo Trình Lịch Sử Đảng</div>'
    '</div>'
    '<p class="banner-subtitle">'
    'Hệ thống tra cứu học thuật thông minh, trích dẫn chính xác theo từng số trang từ giáo trình, '
    'hỗ trợ ghi nhớ đa lượt hội thoại và phân tích chuyên sâu.'
    '</p>'
    '</div>'
)
st.markdown(banner_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. CHAT INPUT & PROMPT RESOLUTION (Root level)
# ─────────────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Nhập câu hỏi về sự kiện, nhân vật, cương lĩnh, mốc thời gian...")

active_prompt = None
if user_input:
    active_prompt = user_input
elif st.session_state.suggested_prompt:
    active_prompt = st.session_state.suggested_prompt
    st.session_state.suggested_prompt = None


# ─────────────────────────────────────────────────────────────────────────────
# 8. HỆ THỐNG TABS ĐA NĂNG (CHAT, TIMELINE, ARCHITECTURE)
# ─────────────────────────────────────────────────────────────────────────────
tab_chat, tab_timeline = st.tabs([
    "💬 Hỏi Đáp & Trích Dẫn",
    "🏛️ Mốc Lịch Sử Đảng (Interactive Timeline)"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: KHÔNG GIAN HỘI THOẠI & TRÍCH DẪN (CHAT)
# ═════════════════════════════════════════════════════════════════════════════
with tab_chat:
    # Gợi ý câu hỏi khi chưa có tin nhắn (dùng st.container chuẩn Native)
    if len(st.session_state.messages) == 0 and not active_prompt:
        st.markdown("""
        <div class="welcome-intro">
            <h3>💡 Bạn muốn tìm hiểu nội dung nào hôm nay?</h3>
            <p>Bấm vào một trong các chủ đề gợi ý bên dưới hoặc tự nhập câu hỏi vào ô chat phía dưới:</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.markdown(
                    "<div style='font-size: 0.78rem; font-weight: 600; color: #64748b; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;'>"
                    "<span>🏛️</span><span>Mốc lịch sử 1930 · Trang 27-35</span>"
                    "</div>",
                    unsafe_allow_html=True
                )
                if st.button("📜 Cương lĩnh chính trị đầu tiên (1930)", key="suggest_1", use_container_width=True):
                    st.session_state.suggested_prompt = "Nội dung cốt lõi và ý nghĩa lịch sử của Cương lĩnh chính trị đầu tiên của Đảng (năm 1930) là gì?"
                    st.rerun()

            with st.container(border=True):
                st.markdown(
                    "<div style='font-size: 0.78rem; font-weight: 600; color: #64748b; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;'>"
                    "<span>⚡</span><span>Mốc lịch sử 1945 · Trang 89-105</span>"
                    "</div>",
                    unsafe_allow_html=True
                )
                if st.button("🚩 Thắng lợi Cách mạng Tháng Tám (1945)", key="suggest_2", use_container_width=True):
                    st.session_state.suggested_prompt = "Phân tích nguyên nhân thắng lợi và bài học kinh nghiệm của Cách mạng Tháng Tám năm 1945?"
                    st.rerun()

        with col2:
            with st.container(border=True):
                st.markdown(
                    "<div style='font-size: 0.78rem; font-weight: 600; color: #64748b; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;'>"
                    "<span>🌐</span><span>Mốc lịch sử 1986 · Trang 280-310</span>"
                    "</div>",
                    unsafe_allow_html=True
                )
                if st.button("🏢 Đường lối Đổi mới - Đại hội VI (1986)", key="suggest_3", use_container_width=True):
                    st.session_state.suggested_prompt = "Đại hội đại biểu toàn quốc lần thứ VI (1986) của Đảng đã đề ra đường lối đổi mới như thế nào?"
                    st.rerun()

            with st.container(border=True):
                st.markdown(
                    "<div style='font-size: 0.78rem; font-weight: 600; color: #64748b; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;'>"
                    "<span>🎯</span><span>Mốc lịch sử 1954 · Trang 145-162</span>"
                    "</div>",
                    unsafe_allow_html=True
                )
                if st.button("🎖️ Chiến thắng Điện Biên Phủ (1954)", key="suggest_4", use_container_width=True):
                    st.session_state.suggested_prompt = "Ý nghĩa lịch sử của chiến thắng Điện Biên Phủ năm 1954 đối với cuộc kháng chiến chống thực dân Pháp?"
                    st.rerun()

    # Hiển thị lịch sử trò chuyện
    for msg in st.session_state.messages:
        avatar_icon = "👤" if msg["role"] == "user" else "🎓"
        with st.chat_message(msg["role"], avatar=avatar_icon):
            if msg["role"] == "assistant" and msg.get("model"):
                clean_model = msg["model"].split("/")[-1].replace(":free", "")
                st.markdown(
                    f'<div class="chat-model-tag">⚡ Phản hồi bởi {clean_model}</div>',
                    unsafe_allow_html=True
                )
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources_box(msg["sources"])

    # Xử lý khi có câu hỏi mới đang active
    if active_prompt:
        st.session_state.messages.append({"role": "user", "content": active_prompt, "model": None, "sources": None})
        with st.chat_message("user", avatar="👤"):
            st.markdown(active_prompt)

        preferred_models = None
        if selected_model_key != "auto":
            other_models = [m for m in FREE_MODELS if m != selected_model_key]
            preferred_models = [selected_model_key] + other_models

        with st.chat_message("assistant", avatar="🎓"):
            model_tag_placeholder = st.empty()
            response_placeholder  = st.empty()

            full_response = ""
            resolved_model = None
            collected_sources = []

            try:
                for token, model_name, source_docs in stream_with_fallback(
                    active_prompt,
                    retriever,
                    session_id=st.session_state.session_id,
                    preferred_models=preferred_models,
                    top_k_rerank=4,
                ):
                    if resolved_model is None:
                        resolved_model = model_name
                        clean_name = model_name.split("/")[-1].replace(":free", "")
                        model_tag_placeholder.markdown(
                            f'<div class="chat-model-tag">⚡ Phản hồi bởi {clean_name} · Streaming</div>',
                            unsafe_allow_html=True
                        )

                    if not collected_sources and source_docs:
                        for doc in source_docs:
                            score = doc.metadata.get("rerank_score", 0.0)
                            collected_sources.append({
                                "page": doc.metadata.get("page", "?"),
                                "file": os.path.basename(doc.metadata.get("source", "textbook_LSD.pdf")),
                                "score_pct": int(score * 100),
                                "snippet": doc.page_content.strip()[:350] + ("..." if len(doc.page_content.strip()) > 350 else ""),
                            })

                    full_response += token
                    response_placeholder.markdown(full_response + " ▌")

                response_placeholder.markdown(full_response)

                if collected_sources:
                    render_sources_box(collected_sources)

            except Exception as e:
                full_response = f"❌ Không thể nhận phản hồi từ các mô hình: {e}"
                response_placeholder.error(full_response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "model": resolved_model,
                "sources": collected_sources,
            })


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: TRỤC THỜI GIAN LỊCH SỬ ĐẢNG (STREAMLIT-ELEMENTS & MUI)
# ═════════════════════════════════════════════════════════════════════════════
with tab_timeline:
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <h3 style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin: 0 0 4px 0;">🏛️ Trục Thời Gian Sự Kiện Lịch Sử Cốt Lõi</h3>
        <p style="font-size: 0.85rem; color: #64748b; margin: 0;">Trích dẫn trực tiếp từ các chương mục trọng tâm của Giáo trình Lịch sử Đảng Cộng sản Việt Nam.</p>
    </div>
    """, unsafe_allow_html=True)

    HISTORICAL_EVENTS = [
        {
            "year": "1930",
            "title": "Hội nghị thành lập Đảng & Cương lĩnh chính trị đầu tiên",
            "page_ref": "Trang 27 - 35",
            "summary": "Hội nghị hợp nhất các tổ chức cộng sản do Nguyễn Ái Quốc chủ trì tại Cửu Long (Hương Cảng). Thông qua Chính cương vắn tắt, Sách lược vắn tắt - Cương lĩnh chính trị đầu tiên xác định đúng đắn con đường cứu nước giải phóng dân tộc.",
            "query": "Trình bày bối cảnh, nội dung và ý nghĩa lịch sử của Hội nghị thành lập Đảng năm 1930?",
        },
        {
            "year": "1945",
            "title": "Cách mạng Tháng Tám thành công & Tuyên ngôn Độc lập",
            "page_ref": "Trang 89 - 105",
            "summary": "Đảng chớp thời cơ ngàn năm có một, lãnh đạo toàn dân tổng khởi nghĩa thắng lợi. Ngày 2/9/1945, Chủ tịch Hồ Chí Minh đọc Tuyên ngôn Độc lập khai sinh nước Việt Nam Dân chủ Cộng hòa.",
            "query": "Phân tích nguyên nhân thắng lợi và bài học kinh nghiệm của Cách mạng Tháng Tám năm 1945?",
        },
        {
            "year": "1954",
            "title": "Chiến thắng Điện Biên Phủ lịch sử",
            "page_ref": "Trang 145 - 162",
            "summary": "Chiến thắng 'lừng lẫy năm châu, chấn động địa cầu', đập tan hoàn toàn kế hoạch Nava của thực dân Pháp, buộc Pháp phải ký kết Hiệp định Giơnevơ lập lại hòa bình ở Đông Dương.",
            "query": "Ý nghĩa lịch sử của chiến thắng Điện Biên Phủ 1954 đối với phong trào giải phóng dân tộc?",
        },
        {
            "year": "1975",
            "title": "Đại thắng Mùa Xuân & Giải phóng hoàn toàn miền Nam",
            "page_ref": "Trang 210 - 235",
            "summary": "Chiến dịch Hồ Chí Minh lịch sử toàn thắng lúc 11h30 ngày 30/4/1975, kết thúc thắng lợi 21 năm kháng chiến chống Mỹ cứu nước, non sông thu về một mối, mở ra kỷ nguyên độc lập, thống nhất.",
            "query": "Phân tích nghệ thuật quân sự và ý nghĩa thời đại của Đại thắng Mùa Xuân 1975?",
        },
        {
            "year": "1986",
            "title": "Đại hội VI & Bước ngoặt Đường lối Đổi mới đất nước",
            "page_ref": "Trang 280 - 310",
            "summary": "Đại hội đại biểu toàn quốc lần thứ VI đề ra đường lối đổi mới toàn diện: Đổi mới tư duy kinh tế, phát triển kinh tế hàng hóa nhiều thành phần, mở cửa chủ động hội nhập quốc tế.",
            "query": "Nội dung cơ bản và ý nghĩa bước ngoặt của đường lối Đổi mới tại Đại hội VI năm 1986?",
        },
    ]

    with elements("history_timeline_view"):
        with mui.Box(sx={"display": "flex", "flexDirection": "column", "gap": 2, "mb": 2}):
            for ev in HISTORICAL_EVENTS:
                with mui.Card(
                    variant="outlined",
                    sx={
                        "borderRadius": "12px",
                        "borderColor": "#e2e8f0",
                        "bgcolor": "#ffffff",
                        "p": 2.5,
                        "transition": "all 0.2s ease-in-out",
                        "&:hover": {
                            "borderColor": "#0f172a",
                            "boxShadow": "0 4px 14px rgba(15,23,42,0.06)",
                        },
                    }
                ):
                    with mui.Box(sx={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "mb": 1.2}):
                        with mui.Box(sx={"display": "flex", "alignItems": "center", "gap": 1.5}):
                            mui.Chip(
                                label=ev["year"],
                                size="small",
                                sx={"bgcolor": "#0f172a", "color": "#ffffff", "fontWeight": 700, "fontSize": "13px"}
                            )
                            mui.Typography(ev["title"], variant="h6", sx={"fontWeight": 700, "fontSize": "16px", "color": "#0f172a"})
                        mui.Chip(
                            label=f"📄 {ev['page_ref']}",
                            size="small",
                            variant="outlined",
                            sx={"borderColor": "#cbd5e1", "color": "#475569", "fontWeight": 600, "fontSize": "12px"}
                        )
                    mui.Typography(
                        ev["summary"],
                        variant="body2",
                        sx={"color": "#475569", "lineHeight": 1.6, "fontSize": "14px"}
                    )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.caption("💡 Bấm vào mốc bên dưới để gửi câu hỏi trực tiếp cho trợ lý:")
    tl_cols = st.columns(len(HISTORICAL_EVENTS))
    for i, ev in enumerate(HISTORICAL_EVENTS):
        with tl_cols[i]:
            if st.button(f"🔍 Mốc {ev['year']}", key=f"btn_tl_{ev['year']}", use_container_width=True):
                st.session_state.suggested_prompt = ev["query"]
                st.rerun()


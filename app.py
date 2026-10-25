# =============================================================================
# STREAMLIT WEB UI - Giao diện hỏi đáp Lịch Sử Đảng CSVN
# Tích hợp: Streaming, Hybrid Search, GPU Reranker, Hộp trích dẫn số trang
# Phong cách: Nền trắng tối giản kết hợp các nút bấm màu đen (Monochrome Minimalist)
# =============================================================================

import os
import importlib
import streamlit as st

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
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #ffffff !important;
        color: #0f172a !important;
    }

    /* Ẩn header & footer mặc định của Streamlit - Loại bỏ hoàn toàn thanh che khuất trên cùng */
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

    /* Sidebar Brand Card */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .sidebar-brand-icon {
        font-size: 1.8rem;
        line-height: 1;
    }
    .sidebar-brand-title {
        font-size: 1.02rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        line-height: 1.2;
    }
    .sidebar-brand-subtitle {
        font-size: 0.74rem;
        color: #64748b;
        margin: 2px 0 0;
        font-weight: 500;
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
        box-shadow: 0 3px 6px rgba(15, 23, 42, 0.12) !important;
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
    button[kind="secondary"]:hover,
    button[kind="primary"]:hover {
        background-color: #1e293b !important;
        background: #1e293b !important;
        border-color: #1e293b !important;
        color: #ffffff !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 16px rgba(15, 23, 42, 0.2) !important;
    }

    .stButton button:active,
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
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
        box-sizing: border-box;
    }
    .banner-title {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0 0 8px 0;
        letter-spacing: -0.02em;
        line-height: 1.25;
    }
    .banner-icon {
        font-size: 1.9rem;
        line-height: 1;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }
    .banner-subtitle {
        color: #475569;
        font-size: 0.92rem;
        margin: 0;
        line-height: 1.6;
        font-weight: 400;
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


# ─────────────────────────────────────────────────────────────────────────────
# 5. SIDEBAR - CHỌN MÔ HÌNH VỚI TÊN & ƯU ĐIỂM
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand Header
    sidebar_brand_html = (
        '<div class="sidebar-brand">'
        '<div class="sidebar-brand-icon">🎓</div>'
        '<div>'
        '<div class="sidebar-brand-title">Lịch Sử Đảng CSVN</div>'
        '<div class="sidebar-brand-subtitle">Trợ lý học thuật RAG v2.1</div>'
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

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Nút xóa lịch sử hội thoại (nền đen, chữ trắng)
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
    '<span class="banner-icon">🎓</span>'
    '<span>Hỏi Đáp Giáo Trình Lịch Sử Đảng</span>'
    '</div>'
    '<p class="banner-subtitle">'
    'Hệ thống tra cứu học thuật thông minh, trích dẫn chính xác theo từng số trang từ giáo trình, '
    'hỗ trợ ghi nhớ đa lượt hội thoại và phân tích chuyên sâu.'
    '</p>'
    '</div>'
)
st.markdown(banner_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. GIAO DIỆN KHI CHƯA CÓ TIN NHẮN (GỢI Ý CÂU HỎI NHANH - NÚT ĐEN)
# ─────────────────────────────────────────────────────────────────────────────
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-intro">
        <h3>💡 Bạn muốn tìm hiểu nội dung nào hôm nay?</h3>
        <p>Bấm vào một trong các câu hỏi gợi ý bên dưới hoặc tự nhập câu hỏi vào ô chat:</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "📜 Cương lĩnh chính trị đầu tiên (1930)",
            key="suggest_1",
            use_container_width=True
        ):
            st.session_state.suggested_prompt = "Nội dung cốt lõi và ý nghĩa lịch sử của Cương lĩnh chính trị đầu tiên của Đảng (năm 1930) là gì?"
            st.rerun()

        if st.button(
            "⚡ Thắng lợi Cách mạng Tháng Tám (1945)",
            key="suggest_2",
            use_container_width=True
        ):
            st.session_state.suggested_prompt = "Phân tích nguyên nhân thắng lợi và bài học kinh nghiệm của Cách mạng Tháng Tám năm 1945?"
            st.rerun()

    with col2:
        if st.button(
            "🏛️ Đường lối Đổi mới - Đại hội VI (1986)",
            key="suggest_3",
            use_container_width=True
        ):
            st.session_state.suggested_prompt = "Đại hội đại biểu toàn quốc lần thứ VI (1986) của Đảng đã đề ra đường lối đổi mới như thế nào?"
            st.rerun()

        if st.button(
            "🎯 Chiến thắng Điện Biên Phủ (1954)",
            key="suggest_4",
            use_container_width=True
        ):
            st.session_state.suggested_prompt = "Ý nghĩa lịch sử của chiến thắng Điện Biên Phủ năm 1954 đối với cuộc kháng chiến chống thực dân Pháp?"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# 8. HIỂN THỊ LỊCH SỬ TIN NHẮN (KÈM HỘP TRÍCH DẪN SỐ TRANG)
# ─────────────────────────────────────────────────────────────────────────────
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

        # Hiển thị trích dẫn nguồn số trang cho các câu trả lời trong lịch sử
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources_box(msg["sources"])


# ─────────────────────────────────────────────────────────────────────────────
# 9. XỬ LÝ NHẬP LIỆU & STREAMING PHẢN HỒI (GPU RERANKED)
# ─────────────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Nhập câu hỏi về sự kiện, nhân vật, cương lĩnh, mốc thời gian...")

active_prompt = None
if user_input:
    active_prompt = user_input
elif st.session_state.suggested_prompt:
    active_prompt = st.session_state.suggested_prompt
    st.session_state.suggested_prompt = None

if active_prompt:
    # 1. Thêm tin nhắn của User
    st.session_state.messages.append({"role": "user", "content": active_prompt, "model": None, "sources": None})
    with st.chat_message("user", avatar="👤"):
        st.markdown(active_prompt)

    # 2. Xác định danh sách model ưu tiên theo lựa chọn người dùng
    preferred_models = None
    if selected_model_key != "auto":
        other_models = [m for m in FREE_MODELS if m != selected_model_key]
        preferred_models = [selected_model_key] + other_models

    # 3. Xử lý phản hồi từ Assistant với streaming
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

                # Thu thập thông tin trích dẫn nguồn từ GPU Reranker
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
                # Hiển thị con trỏ typing ▌
                response_placeholder.markdown(full_response + " ▌")

            # Hoàn tất: bỏ con trỏ typing
            response_placeholder.markdown(full_response)

            # Hiển thị Hộp trích dẫn nguồn số trang ngay bên dưới câu trả lời
            if collected_sources:
                render_sources_box(collected_sources)

        except Exception as e:
            full_response = f"❌ Không thể nhận phản hồi từ các mô hình: {e}"
            response_placeholder.error(full_response)

        # Lưu vào session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "model": resolved_model,
            "sources": collected_sources,
        })

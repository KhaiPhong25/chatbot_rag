# =============================================================================
# RAG CORE - Module lõi dùng chung cho CLI và Web UI
# Features (v2.0):
#   - Embedding cục bộ BAAI/bge-m3 (GPU RTX 4050)
#   - Incremental Indexing: tự động phát hiện & nạp tài liệu mới vào FAISS
#   - Hybrid Search: BM25 (từ khóa chính xác) + FAISS (ngữ nghĩa) kết hợp RRF
#   - Conversational RAG Chain với Multi-Model Fallback
#   - Hỗ trợ cả streaming và non-streaming output
# =============================================================================

import os
import json
import hashlib
import pickle
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

# LangChain - Document Loading & Splitting
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

# LangChain - Embedding & VectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

# LangChain - BM25 & Hybrid Retriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# LangChain - LLM & Chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

# LangChain - Persistent Chat History
from langchain_community.chat_message_histories import FileChatMessageHistory

load_dotenv()

# =============================================================================
# 1. HẰNG SỐ CẤU HÌNH
# =============================================================================

FREE_MODELS = [
    "nvidia/nemotron-3.5-super-vl:free",                  # Ưu tiên 1: NVIDIA Nemotron 3.5 Super VL (context 1M)
    "nvidia/nemotron-3.5-lightning:free",                 # Fallback 1: Nemotron Lightning (tốc độ nhanh)
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", # Fallback 2: Nemotron 30B Reasoning
    "xingchen-agi/xing4.0-29b-a4b:free",                  # Fallback 3: XingChen 29B
    "recursive/flint-v1.0:free",                          # Fallback 4: Recursive Flint
]

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
FAISS_INDEX_DIR   = "faiss_index"
CHAT_HISTORY_FILE = "chat_history.json"
DOCS_DIR          = "./docs"
MANIFEST_FILE     = os.path.join(FAISS_INDEX_DIR, "docs_manifest.json")
BM25_CACHE_FILE   = os.path.join(FAISS_INDEX_DIR, "bm25_retriever.pkl")


# =============================================================================
# 2. EMBEDDING MODEL (BAAI/bge-m3 trên GPU)
# =============================================================================
# Biến toàn cục - chỉ khởi tạo một lần, dùng chung cho cả CLI và Web
_embedding_model = None

def get_embedding_model() -> HuggingFaceEmbeddings:
    """Singleton: chỉ tải model embedding một lần duy nhất vào VRAM GPU."""
    global _embedding_model
    if _embedding_model is None:
        print("\n🧠 Đang khởi động mô hình embedding BAAI/bge-m3 trên GPU...")
        _embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("   ✅ Mô hình embedding đã sẵn sàng.")
    return _embedding_model


# =============================================================================
# 3. INCREMENTAL INDEXING - Quản lý manifest & phát hiện tài liệu mới
# =============================================================================

def _compute_file_hash(filepath: str) -> str:
    """Tính MD5 checksum của một file để phát hiện thay đổi."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_manifest() -> dict:
    """Tải danh sách manifest (file_path -> md5 hash) đã được index."""
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_manifest(manifest: dict) -> None:
    """Lưu manifest sau khi index xong."""
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _get_new_or_changed_files(manifest: dict) -> list[str]:
    """
    Quét thư mục docs/ và trả về danh sách các file PDF:
    - Chưa được index (mới)
    - Đã bị thay đổi nội dung (hash không khớp)
    """
    changed = []
    for pdf_path in sorted(Path(DOCS_DIR).glob("**/*.pdf")):
        pdf_str = str(pdf_path)
        current_hash = _compute_file_hash(pdf_str)
        if manifest.get(pdf_str) != current_hash:
            changed.append(pdf_str)
    return changed


def _load_and_chunk_files(filepaths: list[str], embedding_model: HuggingFaceEmbeddings) -> list:
    """Đọc PDF và phân đoạn ngữ nghĩa bằng SemanticChunker."""
    all_docs = []
    for fp in filepaths:
        print(f"   📄 Đang tải: {os.path.basename(fp)}")
        loader = PyPDFLoader(fp)
        pages  = loader.load()
        all_docs.extend(pages)

    print(f"   ✅ Đã tải {len(all_docs)} trang từ {len(filepaths)} file PDF.")
    print("\n✂️  Đang phân đoạn tài liệu theo ngữ nghĩa (SemanticChunker)...")

    splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
        add_start_index=True,
    )
    splits = splitter.split_documents(all_docs)
    print(f"   ✅ Đã tạo {len(splits)} semantic chunks.")
    return splits


# =============================================================================
# 4. KHỞI TẠO / CẬP NHẬT VECTORSTORE VÀ BM25 (Incremental)
# =============================================================================

def build_or_update_vectorstore(embedding_model: HuggingFaceEmbeddings) -> tuple:
    """
    Kiểm tra và cập nhật FAISS vectorstore cùng BM25 retriever:
    - Nếu chưa có index → khởi tạo mới từ toàn bộ docs/
    - Nếu đã có index + manifest → chỉ xử lý file mới/thay đổi (Incremental)
    - Luôn đồng bộ lại BM25 từ toàn bộ tài liệu hiện có trong FAISS

    Returns:
        (vectorstore, all_splits): vectorstore đã cập nhật và danh sách tất cả chunks
    """
    manifest = _load_manifest()
    index_exists = (
        os.path.exists(FAISS_INDEX_DIR)
        and os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss"))
    )

    if index_exists:
        # ------- Chế độ Incremental Indexing -------
        new_files = _get_new_or_changed_files(manifest)

        if new_files:
            print(f"\n🔄 Phát hiện {len(new_files)} file PDF mới/thay đổi. Đang cập nhật index...")
            # Tải FAISS hiện tại
            vectorstore = FAISS.load_local(
                FAISS_INDEX_DIR,
                embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
                allow_dangerous_deserialization=True,
            )
            # Chunk & embed chỉ các file mới
            new_splits = _load_and_chunk_files(new_files, embedding_model)
            print(f"\n⚙️  Đang embedding {len(new_splits)} chunks mới vào FAISS...")
            vectorstore.add_documents(new_splits)
            vectorstore.save_local(FAISS_INDEX_DIR)

            # Cập nhật manifest
            for fp in new_files:
                manifest[fp] = _compute_file_hash(fp)
            _save_manifest(manifest)
            print(f"   ✅ Đã cập nhật FAISS index với {len(new_splits)} chunks mới.")
        else:
            # Không có gì thay đổi → nạp nhanh từ cache
            print(f"\n📦 Phát hiện FAISS index đã có và không có file mới. Đang tải cache...")
            vectorstore = FAISS.load_local(
                FAISS_INDEX_DIR,
                embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
                allow_dangerous_deserialization=True,
            )
            print(f"   ✅ Đã tải FAISS index thành công.")

    else:
        # ------- Khởi tạo lần đầu -------
        all_pdf_files = sorted(str(p) for p in Path(DOCS_DIR).glob("**/*.pdf"))
        if not all_pdf_files:
            raise FileNotFoundError(f"Không tìm thấy file PDF nào trong thư mục '{DOCS_DIR}/'.")

        print(f"\n⚙️  Chưa có FAISS index. Đang khởi tạo mới từ {len(all_pdf_files)} file PDF...")
        splits = _load_and_chunk_files(all_pdf_files, embedding_model)
        print(f"\n⚙️  Đang embedding {len(splits)} chunks lên GPU...")
        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
        )
        vectorstore.save_local(FAISS_INDEX_DIR)

        # Lưu manifest
        for fp in all_pdf_files:
            manifest[fp] = _compute_file_hash(fp)
        _save_manifest(manifest)
        print(f"   ✅ Đã tạo và lưu FAISS index tại '{FAISS_INDEX_DIR}'.")

    return vectorstore


# =============================================================================
# 5. HYBRID RETRIEVER (BM25 + FAISS với Reciprocal Rank Fusion)
# =============================================================================

def build_hybrid_retriever(vectorstore: FAISS, k: int = 5) -> EnsembleRetriever:
    """
    Tạo Hybrid Retriever kết hợp:
    - BM25Retriever: tìm kiếm chính xác từ khóa (tốt cho ngày tháng, tên người, số hiệu nghị quyết)
    - FAISS Retriever: tìm kiếm ngữ nghĩa vector (tốt cho câu hỏi diễn giải, câu hỏi mở)
    - EnsembleRetriever: gộp kết quả bằng thuật toán Reciprocal Rank Fusion (RRF)
    """
    # Trích xuất toàn bộ documents từ FAISS docstore để dùng cho BM25
    all_docs = list(vectorstore.docstore._dict.values())

    # Khởi tạo BM25 trên toàn bộ corpus
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k

    # FAISS dense retriever
    faiss_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    # Kết hợp: BM25 (40%) + FAISS (60%)
    # → FAISS chiếm tỷ trọng cao hơn vì SemanticChunker đã tối ưu cho ngữ nghĩa
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6],
    )

    print("   ✅ Hybrid Retriever đã sẵn sàng (BM25 40% + FAISS 60%).")
    return hybrid_retriever


# =============================================================================
# 6. PROMPTS
# =============================================================================

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Dựa vào lịch sử trò chuyện và câu hỏi mới nhất của người dùng "
     "(có thể chứa đại từ thay thế như 'nó', 'người đó', 'sự kiện này'...), "
     "hãy viết lại thành một câu hỏi tiếng Việt ĐỘC LẬP và ĐẦY ĐỦ NGỮ CẢNH. "
     "TUYỆT ĐỐI không trả lời câu hỏi. Chỉ viết lại hoặc giữ nguyên nếu câu hỏi đã rõ."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

qa_system_prompt = """Bạn là một trợ lý học thuật chuyên sâu về môn học Lịch sử Đảng Cộng sản Việt Nam.
Nhiệm vụ của bạn là giải đáp câu hỏi của người học dựa CHÍNH XÁC và DUY NHẤT vào các đoạn ngữ cảnh (Context) được cung cấp dưới đây.

---
NGUYÊN TẮC BẮT BUỘC:
1. Tính trung thực với ngữ cảnh:
   - Chỉ sử dụng thông tin có trong phần [NGỮ CẢNH ĐƯỢC CUNG CẤP]. Không tự suy đoán hay bổ sung dữ liệu lịch sử bên ngoài.
   - Nếu ngữ cảnh không đủ thông tin, hãy phản hồi: "Tài liệu học tập được cung cấp hiện chưa có đủ thông tin để trả lời câu hỏi này."

2. Tính chính xác về thuật ngữ và sự kiện:
   - Giữ nguyên các thuật ngữ chính trị - lịch sử chuẩn mực.
   - Chú ý chính xác tuyệt đối về: mốc thời gian, địa điểm, nhân vật, tên văn kiện.

3. Phong cách và văn phong:
   - Khách quan, chuẩn mực sư phạm, mạch lạc và trang trọng.
   - Trình bày rõ ràng bằng tiêu đề con, gạch đầu dòng hoặc đánh số thứ tự.

4. Dẫn nguồn:
   - Cuối mỗi câu trả lời, trích dẫn ngắn gọn nguồn dựa trên metadata có trong ngữ cảnh.

[NGỮ CẢNH ĐƯỢC CUNG CẤP]
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


# =============================================================================
# 7. LLM & CHAIN
# =============================================================================

def get_llm(model_name: str) -> ChatOpenAI:
    """Khởi tạo LLM OpenRouter với streaming."""
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model=model_name,
        temperature=0,
        streaming=True,  # Bật streaming cho tất cả models
        default_headers={
            "HTTP-Referer": "https://localhost:8000",
            "X-Title": "LichSuDang-Agentic-RAG-v2",
        }
    )


def _build_rag_chain(model_name: str, retriever):
    """Xây dựng RAG chain lõi (không có memory wrapper) cho một model cụ thể."""
    llm         = get_llm(model_name)
    context_llm = get_llm(model_name)

    contextualize_q_chain = contextualize_q_prompt | context_llm | StrOutputParser()

    def contextualized_question(input_dict: dict):
        """Viết lại câu hỏi nếu có lịch sử hội thoại."""
        if input_dict.get("chat_history"):
            return contextualize_q_chain
        return input_dict["question"]

    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualized_question | retriever
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def build_conversational_chain(model_name: str, retriever):
    """Bọc RAG chain với bộ nhớ hội thoại FileChatMessageHistory."""
    rag_chain = _build_rag_chain(model_name, retriever)

    def get_session_history(session_id: str) -> FileChatMessageHistory:
        return FileChatMessageHistory(CHAT_HISTORY_FILE)

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    return conversational_chain


# =============================================================================
# 8. INVOKE VỚI FALLBACK (Non-streaming - dùng cho CLI đơn giản)
# =============================================================================

def invoke_with_fallback(
    question: str,
    retriever,
    session_id: str = "default_user",
) -> tuple[str, str]:
    """
    Thử lần lượt qua từng model, trả về (câu_trả_lời, tên_model).
    Dùng .invoke() - không streaming.
    """
    last_error = None
    for model_name in FREE_MODELS:
        try:
            chain  = build_conversational_chain(model_name, retriever)
            answer = chain.invoke(
                {"question": question},
                config={"configurable": {"session_id": session_id}},
            )
            return answer, model_name
        except Exception as e:
            last_error = e
            print(f"   ⚠️  [{model_name}] gặp lỗi: {str(e)[:80]}... Thử model tiếp theo.")
            continue

    raise Exception(f"Tất cả {len(FREE_MODELS)} model đều không phản hồi. Lỗi cuối: {last_error}")


# =============================================================================
# 9. STREAM VỚI FALLBACK (Streaming - dùng cho CLI streaming và Streamlit)
# =============================================================================

def stream_with_fallback(
    question: str,
    retriever,
    session_id: str = "default_user",
) -> Iterator[tuple[str, str]]:
    """
    Generator: stream từng token câu trả lời.
    Yields (token_str, model_name) từ model đầu tiên phản hồi thành công.
    
    Nếu một model báo lỗi trước khi stream bất kỳ token nào
    → tự động chuyển sang model tiếp theo.
    """
    last_error = None

    for model_name in FREE_MODELS:
        try:
            chain  = build_conversational_chain(model_name, retriever)
            token_count = 0

            for token in chain.stream(
                {"question": question},
                config={"configurable": {"session_id": session_id}},
            ):
                token_count += 1
                yield token, model_name

            # Stream thành công → dừng vòng lặp fallback
            if token_count > 0:
                return

        except Exception as e:
            last_error = e
            print(f"\n   ⚠️  [{model_name}] gặp lỗi khi stream: {str(e)[:80]}... Thử model tiếp theo.")
            continue

    raise Exception(f"Tất cả {len(FREE_MODELS)} model đều không stream được. Lỗi cuối: {last_error}")

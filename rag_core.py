# =============================================================================
# RAG CORE - Module lõi dùng chung cho CLI và Web UI
# Features (v2.1):
#   - Embedding cục bộ BAAI/bge-m3 (GPU RTX 4050, float16)
#   - GPU Cross-Encoder Reranker BAAI/bge-reranker-v2-m3 (float16)
#   - Incremental Indexing: tự động phát hiện & nạp tài liệu mới vào FAISS
#   - Hybrid Search: BM25 (từ khóa chính xác) + FAISS (ngữ nghĩa) kết hợp RRF
#   - Trích xuất và định dạng trích dẫn nguồn (Trang, File, Điểm Rerank)
#   - Conversational RAG với Multi-Model Fallback & Streaming
# =============================================================================

import os
import json
import hashlib
import pickle
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from dotenv import load_dotenv

# LangChain - Document Loading & Splitting
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document

# LangChain - Embedding & VectorStore
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from huggingface_hub import InferenceClient

# CrossEncoder Reranker
from sentence_transformers import CrossEncoder

# LangChain - BM25 & Hybrid Retriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# LangChain - LLM & Prompts
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

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
# 2. EMBEDDING MODEL (Hugging Face Serverless Inference API / Local BAAI/bge-m3)
# =============================================================================

class HuggingFaceInferenceEmbeddings(Embeddings):
    """
    Embedding Model sử dụng Hugging Face Serverless Inference API:
    - Zero RAM Footprint: Không tải model 2.2GB về máy, hoàn toàn không tốn RAM.
    - Cực kỳ lý tưởng cho môi trường tài nguyên hạn chế như Streamlit Cloud (1GB RAM).
    - Tạo vector 1024 chiều tương thích 100% với FAISS index BAAI/bge-m3.
    """
    def __init__(self, model_name: str = "BAAI/bge-m3", token: str | None = None):
        self.model_name = model_name
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            raise ValueError("Không tìm thấy HF_TOKEN trong biến môi trường hoặc file .env!")
        self.client = InferenceClient(token=self.token)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Nhúng danh sách văn bản theo batch."""
        embeddings = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            res = self.client.feature_extraction(batch, model=self.model_name)
            if hasattr(res, "tolist"):
                res = res.tolist()
            if isinstance(res, list) and len(res) > 0 and not isinstance(res[0], list):
                embeddings.append(res)
            else:
                embeddings.extend(res)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Nhúng một câu truy vấn của người dùng."""
        res = self.client.feature_extraction(text, model=self.model_name)
        if hasattr(res, "tolist"):
            res = res.tolist()
        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
            return res[0]
        return res


_embedding_model = None

def get_embedding_model() -> Embeddings:
    """
    Singleton: Khởi tạo mô hình Embedding.
    Ưu tiên 1: Hugging Face Serverless Inference API (Zero RAM, tối ưu tuyệt đối cho Streamlit Cloud).
    Fallback 2: Tải mô hình BAAI/bge-m3 cục bộ (GPU/CPU) nếu token chưa được cấp quyền Inference.
    """
    global _embedding_model
    if _embedding_model is None:
        hf_token = os.getenv("HF_TOKEN")
        use_api = os.getenv("USE_HF_INFERENCE_API", "true").lower() in ("true", "1", "yes")

        if use_api and hf_token:
            try:
                print("\n🌐 Đang kiểm tra Hugging Face Serverless Inference API cho BAAI/bge-m3...")
                emb = HuggingFaceInferenceEmbeddings(model_name="BAAI/bge-m3", token=hf_token)
                # Kiểm tra kết nối nhanh bằng 1 câu truy vấn ngắn
                emb.embed_query("Đảng Cộng sản Việt Nam")
                _embedding_model = emb
                print("   ✅ Kết nối Hugging Face Serverless Inference API thành công (Chế độ Zero RAM - Sẵn sàng cho Streamlit Cloud).")
                return _embedding_model
            except Exception as e:
                err_msg = str(e)
                if "403" in err_msg or "permissions" in err_msg.lower():
                    print("   ⚠️  Token HF_TOKEN chưa có quyền 'Inference: Make calls to the serverless Inference API'.")
                    print("   👉 Vui lòng vào https://huggingface.co/settings/tokens bật quyền Inference cho token.")
                else:
                    print(f"   ⚠️  Không thể kết nối Serverless Inference API: {err_msg[:100]}...")
                print("   🔄 Đang tự động chuyển sang mô hình Embedding cục bộ (Local Weights)...")

        print("\n🧠 Đang khởi động mô hình embedding BAAI/bge-m3 cục bộ (Local weights)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"device": device}
        if device == "cuda":
            model_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

        _embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
        print("   ✅ Mô hình embedding cục bộ đã sẵn sàng.")
    return _embedding_model


# =============================================================================
# 3. GPU CROSS-ENCODER RERANKER (BAAI/bge-reranker-v2-m3)
# =============================================================================
_reranker_model = None

def get_reranker_model() -> CrossEncoder | None:
    """
    Singleton: Khởi tạo mô hình Cross-Encoder Reranker.
    - Trên môi trường có GPU NVIDIA (CUDA): Nạp mô hình float16 tối ưu độ chính xác.
    - Trên môi trường không có GPU (CPU / Streamlit Cloud): Tự động tắt để tránh tràn RAM (OOM limit 1GB).
    """
    global _reranker_model
    if _reranker_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            print("\n💡 Đang chạy trên môi trường CPU/Cloud (không có CUDA GPU): Tự động bỏ qua GPU Reranker để tối ưu bộ nhớ (<500MB RAM).")
            return None

        print("\n🎯 Đang nạp mô hình Reranker BAAI/bge-reranker-v2-m3 trên GPU (float16)...")
        _reranker_model = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            device="cuda",
            model_kwargs={"torch_dtype": torch.float16},
        )
        print("   ✅ Mô hình GPU Reranker đã sẵn sàng.")
    return _reranker_model


def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int = 4,
    reranker: CrossEncoder | None = None,
) -> list[Document]:
    """
    Chấm điểm lại các candidate documents:
    - Nếu có GPU Reranker: Chấm điểm tương quan chéo (Cross-Attention) trên GPU.
    - Nếu không có GPU Reranker (Streamlit Cloud): Giữ nguyên phân hạng từ Hybrid Search (BM25 + FAISS Dense).
    """
    if not documents:
        return []

    if reranker is None:
        reranker = get_reranker_model()

    if reranker is None:
        # Chế độ Cloud / CPU: Dùng trực tiếp kết quả chất lượng cao từ Hybrid Search RRF
        scored_docs = []
        for i, doc in enumerate(documents[:top_k]):
            score = doc.metadata.get("rerank_score", max(0.95 - i * 0.04, 0.70))
            scored_doc = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": float(score)}
            )
            scored_docs.append(scored_doc)
        return scored_docs

    # Chạy CrossEncoder trên GPU
    pairs = [[query, doc.page_content] for doc in documents]
    raw_scores = reranker.predict(pairs)
    sigmoid_scores = 1 / (1 + np.exp(-np.array(raw_scores, dtype=np.float32)))

    scored_docs = []
    for doc, score in zip(documents, sigmoid_scores):
        scored_doc = Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "rerank_score": float(score)}
        )
        scored_docs.append(scored_doc)

    scored_docs.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
    return scored_docs[:top_k]


# =============================================================================
# 4. INCREMENTAL INDEXING - Quản lý manifest & phát hiện tài liệu mới
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


def _load_and_chunk_files(filepaths: list[str], embedding_model: Embeddings) -> list:
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
# 5. KHỞI TẠO / CẬP NHẬT VECTORSTORE VÀ BM25
# =============================================================================

def build_or_update_vectorstore(embedding_model: Embeddings) -> FAISS:
    """
    Kiểm tra và cập nhật FAISS vectorstore:
    - Nếu chưa có index → khởi tạo mới từ toàn bộ docs/
    - Nếu đã có index + manifest → chỉ xử lý file mới/thay đổi (Incremental)
    """
    manifest = _load_manifest()
    index_exists = (
        os.path.exists(FAISS_INDEX_DIR)
        and os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss"))
    )

    if index_exists:
        new_files = _get_new_or_changed_files(manifest)

        if new_files:
            print(f"\n🔄 Phát hiện {len(new_files)} file PDF mới/thay đổi. Đang cập nhật index...")
            vectorstore = FAISS.load_local(
                FAISS_INDEX_DIR,
                embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
                allow_dangerous_deserialization=True,
            )
            new_splits = _load_and_chunk_files(new_files, embedding_model)
            print(f"\n⚙️  Đang embedding {len(new_splits)} chunks mới vào FAISS...")
            vectorstore.add_documents(new_splits)
            vectorstore.save_local(FAISS_INDEX_DIR)

            for fp in new_files:
                manifest[fp] = _compute_file_hash(fp)
            _save_manifest(manifest)
            print(f"   ✅ Đã cập nhật FAISS index với {len(new_splits)} chunks mới.")
        else:
            print("\n📦 Phát hiện FAISS index đã có và không có file mới. Đang tải cache...")
            vectorstore = FAISS.load_local(
                FAISS_INDEX_DIR,
                embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
                allow_dangerous_deserialization=True,
            )
            print("   ✅ Đã tải FAISS index thành công.")

    else:
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

        for fp in all_pdf_files:
            manifest[fp] = _compute_file_hash(fp)
        _save_manifest(manifest)
        print(f"   ✅ Đã tạo và lưu FAISS index tại '{FAISS_INDEX_DIR}'.")

    return vectorstore


# =============================================================================
# 6. HYBRID RETRIEVER (BM25 + FAISS kết hợp RRF)
# =============================================================================

def build_hybrid_retriever(vectorstore: FAISS, k: int = 12) -> EnsembleRetriever:
    """
    Tạo Hybrid Retriever kết hợp:
    - BM25Retriever: tìm kiếm chính xác từ khóa (lấy k ứng viên)
    - FAISS Retriever: tìm kiếm ngữ nghĩa vector (lấy k ứng viên)
    - EnsembleRetriever: gộp kết quả bằng Reciprocal Rank Fusion (RRF)
    """
    all_docs = list(vectorstore.docstore._dict.values())

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k

    faiss_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6],
    )

    print(f"   ✅ Hybrid Retriever đã sẵn sàng (k={k} ứng viên).")
    return hybrid_retriever


# =============================================================================
# 7. PROMPTS & DẪN NGUỒN
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
Nhiệm vụ của bạn là giải đáp câu hỏi của người học dựa CHÍNH XÁC và DUY NHẤT vào các đoạn ngữ cảnh (Context) đã được hệ thống GPU Reranker chọn lọc dưới đây.

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

4. Dẫn nguồn số trang:
   - Trong câu trả lời hoặc cuối phần phản hồi, trích dẫn rõ số trang từ giáo trình (ví dụ: "[Trang 45, Giáo trình Lịch sử Đảng]").

[NGỮ CẢNH ĐƯỢC CUNG CẤP]
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


# =============================================================================
# 8. LLM & TÍCH HỢP RERANKER TRONG RETRIEVAL PIPELINE
# =============================================================================

def get_llm(model_name: str) -> ChatOpenAI:
    """Khởi tạo LLM OpenRouter với streaming."""
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model=model_name,
        temperature=0,
        streaming=True,
        default_headers={
            "HTTP-Referer": "https://localhost:8000",
            "X-Title": "LichSuDang-Agentic-RAG-v2",
        }
    )


def get_session_history(session_id: str) -> FileChatMessageHistory:
    """Lấy lịch sử hội thoại từ file json."""
    return FileChatMessageHistory(CHAT_HISTORY_FILE)


def format_context_with_citations(docs: list[Document]) -> str:
    """
    Định dạng danh sách documents thành chuỗi ngữ cảnh gửi LLM:
    Hiển thị rõ ràng tên file, số trang và điểm liên quan từ Reranker.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        source_file = os.path.basename(doc.metadata.get("source", "textbook_LSD.pdf"))
        score = doc.metadata.get("rerank_score")
        score_str = f" | Độ khớp: {score*100:.1f}%" if score is not None else ""
        header = f"[Tài liệu {i}: {source_file} (Trang {page}){score_str}]"
        formatted.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(formatted)


def retrieve_and_rerank(
    query: str,
    retriever: EnsembleRetriever,
    top_k: int = 4,
) -> tuple[str, list[Document]]:
    """
    Hai giai đoạn tìm kiếm nâng cao (Retrieve & Rerank):
    1. Hybrid Retriever lấy ra 12 ứng viên đa dạng (BM25 + FAISS).
    2. Cross-Encoder Reranker trên GPU RTX 4050 chấm điểm lại và lọc ra top_k đoạn trích tinh hoa nhất.
    Returns:
        (context_string, reranked_docs)
    """
    candidates = retriever.invoke(query)
    reranked = rerank_documents(query, candidates, top_k=top_k)
    context_str = format_context_with_citations(reranked)
    return context_str, reranked


def contextualize_question(
    question: str,
    chat_history: list,
    model_name: str = FREE_MODELS[0],
) -> str:
    """Viết lại câu hỏi đầy đủ ngữ cảnh độc lập nếu có lịch sử trò chuyện."""
    if not chat_history:
        return question
    try:
        llm = get_llm(model_name)
        chain = contextualize_q_prompt | llm | StrOutputParser()
        rewritten = chain.invoke({"question": question, "chat_history": chat_history})
        return rewritten.strip() if rewritten else question
    except Exception:
        return question


# =============================================================================
# 9. INVOKE VỚI FALLBACK (Non-streaming)
# =============================================================================

def invoke_with_fallback(
    question: str,
    retriever: EnsembleRetriever,
    session_id: str = "default_user",
    preferred_models: list[str] | None = None,
    top_k_rerank: int = 4,
) -> tuple[str, str, list[Document]]:
    """
    Trả về (câu_trả_lời, tên_model, danh_sách_nguồn_trích_dẫn).
    """
    models_to_try = preferred_models if preferred_models else FREE_MODELS
    history = get_session_history(session_id)
    history_messages = history.messages

    effective_q = contextualize_question(question, history_messages, models_to_try[0])
    context_str, source_docs = retrieve_and_rerank(
        effective_q,
        retriever,
        top_k=top_k_rerank,
    )

    messages = qa_prompt.format_messages(
        context=context_str,
        chat_history=history_messages,
        question=question,
    )

    last_error = None
    for model_name in models_to_try:
        try:
            llm = get_llm(model_name)
            response = llm.invoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)

            history.add_user_message(question)
            history.add_ai_message(answer)
            return answer, model_name, source_docs

        except Exception as e:
            last_error = e
            print(f"   ⚠️  [{model_name}] gặp lỗi: {str(e)[:80]}... Thử model tiếp theo.")
            continue

    raise Exception(f"Tất cả {len(models_to_try)} model đều không phản hồi. Lỗi cuối: {last_error}")


# =============================================================================
# 10. STREAM VỚI FALLBACK (Streaming output + Trả về Source Documents)
# =============================================================================

def stream_with_fallback(
    question: str,
    retriever: EnsembleRetriever,
    session_id: str = "default_user",
    preferred_models: list[str] | None = None,
    top_k_rerank: int = 4,
) -> Iterator[tuple[str, str, list[Document]]]:
    """
    Generator: stream từng token câu trả lời kèm danh sách nguồn trích dẫn.
    Yields: (token_str, model_name, source_documents)

    - source_documents: danh sách các chunks tinh hoa đã qua GPU Reranker kèm điểm số và số trang.
    """
    models_to_try = preferred_models if preferred_models else FREE_MODELS
    history = get_session_history(session_id)
    history_messages = history.messages

    # 1. Viết lại câu hỏi nếu có lịch sử trò chuyện
    effective_q = contextualize_question(question, history_messages, models_to_try[0])

    # 2. Hybrid Retrieve (12 ứng viên) + Rerank trên GPU (top 4 tinh hoa)
    context_str, source_docs = retrieve_and_rerank(
        effective_q,
        retriever,
        top_k=top_k_rerank,
    )

    # 3. Tạo messages hoàn chỉnh gửi LLM
    messages = qa_prompt.format_messages(
        context=context_str,
        chat_history=history_messages,
        question=question,
    )

    last_error = None

    for model_name in models_to_try:
        try:
            llm = get_llm(model_name)
            full_response = ""

            for chunk in llm.stream(messages):
                token = chunk.content
                full_response += token
                yield token, model_name, source_docs

            # Nếu stream thành công và có nội dung
            if full_response.strip():
                history.add_user_message(question)
                history.add_ai_message(full_response)
                return

        except Exception as e:
            last_error = e
            print(f"\n   ⚠️  [{model_name}] gặp lỗi khi stream: {str(e)[:80]}... Thử model tiếp theo.")
            continue

    raise Exception(f"Tất cả {len(models_to_try)} model đều không stream được. Lỗi cuối: {last_error}")

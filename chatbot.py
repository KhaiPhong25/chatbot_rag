# =============================================================================
# RAG CHATBOT - LỊCH SỬ ĐẢNG CỘNG SẢN VIỆT NAM
# Features:
#   - Embedding cục bộ với BAAI/bge-m3 (GPU RTX 4050)
#   - Cache FAISS: chỉ embed 1 lần, lần sau load từ ổ đĩa
#   - Multi-model fallback: tự động chuyển sang model tiếp theo nếu lỗi
#   - Trí nhớ đa lượt (Conversational RAG): lưu lịch sử vĩnh viễn ra file JSON
# =============================================================================

import os
import json
from datetime import datetime

from dotenv import load_dotenv

# LangChain - Document Loading & Splitting
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain - Embedding & VectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

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
# 1. CẤU HÌNH CÁC MÔ HÌNH LLM FREE TRÊN OPENROUTER
#    Danh sách theo thứ tự ưu tiên - tự động fallback nếu model trước bị lỗi
# =============================================================================
FREE_MODELS = [
    "google/gemini-2.0-flash-exp:free",       # Tiếng Việt xuất sắc, nhanh
    "qwen/qwen-2.5-72b-instruct:free",        # 72B, rất mạnh, hiểu văn phong học thuật
    "meta-llama/llama-3.3-70b-instruct:free", # Llama 70B, tuân thủ prompt tốt
    "deepseek/deepseek-r1:free",              # Mô hình suy luận sâu
    "nvidia/nemotron-3.5-super-vl:free",      # Nvidia Nemotron
]

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
FAISS_INDEX_DIR   = "faiss_index"
CHAT_HISTORY_FILE = "chat_history.json"


def get_llm(model_name: str) -> ChatOpenAI:
    """Khởi tạo LLM OpenRouter cho một model cụ thể."""
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model=model_name,
        temperature=0,
        default_headers={
            "HTTP-Referer": "https://localhost:8000",
            "X-Title": "LichSuDang-Agentic-RAG",
        }
    )


# =============================================================================
# 2. LOAD DOCUMENTS & CHUNKING
# =============================================================================
print("\n📂 Đang tải tài liệu PDF...")
loader = DirectoryLoader(
    path="./docs",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
    use_multithreading=True,
)
docs = loader.load()
print(f"   ✅ Đã tải {len(docs)} trang tài liệu.")

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\**\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    add_start_index=True,
    strip_whitespace=True,
    separators=MARKDOWN_SEPARATORS,
)
splits = text_splitter.split_documents(docs)


# =============================================================================
# 3. EMBEDDING MODEL (BAAI/bge-m3 chạy trên GPU RTX 4050)
# =============================================================================
print("\n🧠 Đang khởi động mô hình embedding BAAI/bge-m3 trên GPU...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)
print("   ✅ Mô hình embedding đã sẵn sàng.")


# =============================================================================
# 4. FAISS VECTORSTORE (Cache - chỉ embed 1 lần duy nhất)
# =============================================================================
if os.path.exists(FAISS_INDEX_DIR):
    print(f"\n📦 Phát hiện FAISS index đã có tại '{FAISS_INDEX_DIR}'. Đang tải lại...")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_DIR,
        embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
        allow_dangerous_deserialization=True,
    )
    print("   ✅ Đã tải FAISS index thành công (bỏ qua bước embedding).")
else:
    print(f"\n⚙️  Chưa có FAISS index. Đang embedding {len(splits)} chunks lên GPU (lần đầu tiên)...")
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )
    vectorstore.save_local(FAISS_INDEX_DIR)
    print(f"   ✅ Đã embedding và lưu index vào '{FAISS_INDEX_DIR}' thành công!")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)


# =============================================================================
# 5. PROMPTS
# =============================================================================

# Prompt 1: Viết lại câu hỏi thành câu độc lập (dựa vào lịch sử trò chuyện)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Dựa vào lịch sử trò chuyện và câu hỏi mới nhất của người dùng "
     "(có thể chứa đại từ thay thế như 'nó', 'người đó', 'sự kiện này'...), "
     "hãy viết lại thành một câu hỏi tiếng Việt ĐỘC LẬP và ĐẦY ĐỦ NGỮ CẢNH. "
     "TUYỆT ĐỐI không trả lời câu hỏi. Chỉ viết lại hoặc giữ nguyên nếu câu hỏi đã rõ."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# Prompt 2: Prompt trả lời chính (nghiêm ngặt, học thuật, dẫn nguồn)
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
# 6. XÂY DỰNG CONVERSATIONAL RAG CHAIN VỚI MULTI-MODEL FALLBACK
# =============================================================================

def build_conversational_chain(model_name: str):
    """Xây dựng RAG Chain với trí nhớ đa lượt cho một model cụ thể."""
    llm         = get_llm(model_name)
    context_llm = get_llm(model_name)

    # Chain viết lại câu hỏi thành câu độc lập
    contextualize_q_chain = contextualize_q_prompt | context_llm | StrOutputParser()

    def contextualized_question(input_dict: dict):
        """Nếu có lịch sử -> viết lại câu hỏi. Nếu chưa có -> giữ nguyên."""
        if input_dict.get("chat_history"):
            return contextualize_q_chain
        return input_dict["question"]

    # RAG Chain chính
    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualized_question | retriever
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # Hàm lấy lịch sử từ file JSON (lưu vĩnh viễn)
    def get_session_history(session_id: str) -> FileChatMessageHistory:
        return FileChatMessageHistory(CHAT_HISTORY_FILE)

    # Bọc RAG chain với bộ nhớ lịch sử
    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return conversational_chain


def invoke_with_fallback(question: str, session_id: str = "default_user") -> tuple:
    """
    Thử gọi lần lượt qua từng model free.
    Nếu model bị lỗi (502, rate limit...) thì tự động chuyển sang model tiếp theo.
    Trả về (câu_trả_lời, tên_model_đã_dùng).
    """
    last_error = None

    for model_name in FREE_MODELS:
        try:
            chain = build_conversational_chain(model_name)
            answer = chain.invoke(
                {"question": question},
                config={"configurable": {"session_id": session_id}},
            )
            return answer, model_name  # Thành công -> trả về luôn
        except Exception as e:
            last_error = e
            print(f"   ⚠️  [{model_name}] gặp lỗi: {str(e)[:80]}... Thử model tiếp theo.")
            continue

    raise Exception(f"Tất cả {len(FREE_MODELS)} model đều không phản hồi. Lỗi cuối: {last_error}")


# =============================================================================
# 7. VÒNG LẶP HỎI ĐÁP LIÊN TỤC VỚI TRÍ NHỚ ĐA LƯỢT
# =============================================================================
print("\n" + "="*60)
print("🤖 CHATBOT LỊCH SỬ ĐẢNG CỘNG SẢN VIỆT NAM - ĐÃ SẴN SÀNG!")
print(f"📁 Lịch sử trò chuyện lưu tại: '{CHAT_HISTORY_FILE}'")
print("💡 Lệnh: 'exit'/'q' để thoát | 'clear' để xóa lịch sử hội thoại")
print("="*60 + "\n")

SESSION_ID = "default_user"

while True:
    try:
        question = input("\n👤 Bạn hỏi: ").strip()

        if not question:
            continue

        # Lệnh thoát
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Tạm biệt bạn! Chúc bạn học tốt.")
            break

        # Lệnh xóa lịch sử để bắt đầu hội thoại mới
        if question.lower() in ["clear", "cls", "reset"]:
            if os.path.exists(CHAT_HISTORY_FILE):
                os.remove(CHAT_HISTORY_FILE)
            print("🗑️  Đã xóa toàn bộ lịch sử. Bắt đầu cuộc hội thoại mới!")
            continue

        print("\n⏳ Đang tìm kiếm tài liệu và suy luận câu trả lời...")

        answer, used_model = invoke_with_fallback(question, session_id=SESSION_ID)

        print(f"\n🤖 Trợ lý [via {used_model}]:")
        print(answer)
        print("-" * 60)

    except KeyboardInterrupt:
        print("\n\n👋 Đã dừng chương trình. Tạm biệt!")
        break
    except Exception as e:
        print(f"\n❌ Lỗi không thể phục hồi: {e}")



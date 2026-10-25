# =============================================================================
# CHATBOT CLI - Terminal Interface với Streaming Output & GPU Reranker
# Dùng module rag_core.py để khởi tạo Hybrid Retriever + GPU Reranker + Chain
# =============================================================================

import os
import sys
from rag_core import (
    get_embedding_model,
    get_reranker_model,
    build_or_update_vectorstore,
    build_hybrid_retriever,
    stream_with_fallback,
    CHAT_HISTORY_FILE,
)

# Khởi tạo các thành phần RAG
embedding_model = get_embedding_model()
vectorstore     = build_or_update_vectorstore(embedding_model)
retriever       = build_hybrid_retriever(vectorstore, k=12)

# Nạp GPU Reranker vào VRAM
get_reranker_model()

print("\n" + "="*60)
print("🤖 CHATBOT LỊCH SỬ ĐẢNG CỘNG SẢN VIỆT NAM v2.1 - SẴN SÀNG!")
print("⚡ Tích hợp: Hybrid Search + GPU Reranker (bge-reranker-v2-m3)")
print(f"📁 Lịch sử trò chuyện lưu tại: '{CHAT_HISTORY_FILE}'")
print("💡 Lệnh: 'exit'/'q' để thoát | 'clear' để xóa lịch sử hội thoại")
print("="*60 + "\n")

SESSION_ID = "default_user"

while True:
    try:
        question = input("\n👤 Bạn hỏi: ").strip()

        if not question:
            continue

        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Tạm biệt bạn! Chúc bạn học tốt.")
            break

        if question.lower() in ["clear", "cls", "reset"]:
            if os.path.exists(CHAT_HISTORY_FILE):
                os.remove(CHAT_HISTORY_FILE)
            print("🗑️  Đã xóa toàn bộ lịch sử. Bắt đầu cuộc hội thoại mới!")
            continue

        print("\n⏳ Đang tìm kiếm tài liệu (Hybrid Search + GPU Reranking)...\n")

        # ── STREAMING OUTPUT ──
        used_model   = None
        full_answer  = ""
        first_token  = True
        sources      = None

        for token, model_name, source_docs in stream_with_fallback(question, retriever, session_id=SESSION_ID):
            if sources is None:
                sources = source_docs

            if first_token:
                print(f"🤖 Trợ lý [via {model_name}]:\n")
                used_model  = model_name
                first_token = False

            print(token, end="", flush=True)
            full_answer += token

        # In danh sách nguồn trích dẫn
        if sources:
            print("\n\n📚 [Nguồn trích dẫn - GPU Reranked]:")
            for i, doc in enumerate(sources, 1):
                p = doc.metadata.get("page", "?")
                s = doc.metadata.get("rerank_score", 0.0)
                fn = os.path.basename(doc.metadata.get("source", "textbook_LSD.pdf"))
                print(f"   [{i}] Trang {p} · {fn} (Độ khớp: {s*100:.1f}%)")

        print("\n" + "-"*60)

    except KeyboardInterrupt:
        print("\n\n👋 Đã dừng chương trình. Tạm biệt!")
        break
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")

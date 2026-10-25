# =============================================================================
# CHATBOT CLI - Terminal Interface với Streaming Output
# Dùng module rag_core.py để khởi tạo Hybrid Retriever + Chain
# =============================================================================

import os
import sys
from rag_core import (
    get_embedding_model,
    build_or_update_vectorstore,
    build_hybrid_retriever,
    stream_with_fallback,
    CHAT_HISTORY_FILE,
)

# Khởi tạo các thành phần RAG
embedding_model = get_embedding_model()
vectorstore     = build_or_update_vectorstore(embedding_model)
retriever       = build_hybrid_retriever(vectorstore, k=5)

print("\n" + "="*60)
print("🤖 CHATBOT LỊCH SỬ ĐẢNG CỘNG SẢN VIỆT NAM v2.0 - SẴN SÀNG!")
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

        print("\n⏳ Đang tìm kiếm tài liệu và suy luận câu trả lời...\n")

        # ── STREAMING OUTPUT ──
        used_model   = None
        full_answer  = ""
        first_token  = True

        for token, model_name in stream_with_fallback(question, retriever, session_id=SESSION_ID):
            if first_token:
                print(f"🤖 Trợ lý [via {model_name}]:\n")
                used_model  = model_name
                first_token = False

            print(token, end="", flush=True)
            full_answer += token

        print("\n" + "-"*60)

    except KeyboardInterrupt:
        print("\n\n👋 Đã dừng chương trình. Tạm biệt!")
        break
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")

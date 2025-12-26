import config
from train import train_model
from inference import generate_text_typewriter, chat

if __name__ == "__main__":
    # 1. Train or Load Model
    trained_model, proc = train_model()

    if not trained_model:
        print("❌ Could not load or train model. Exiting.")
        exit()

    # 2. Run Inference
    if not config.CHAT_MODE:
        print("\n🤖 Model Ready for Inference!")
        while True:
            text = input("📝 Enter start text (or 'q'): ")
            if text == "q":
                break
            formatted_prompt = f"<user>: {text}\n<bot>:"
            generate_text_typewriter(
                trained_model,
                proc,
                formatted_prompt,
                num_sentences=config.NUM_SENTENCES,
                temperature=config.TEMPERATURE,
                K=config.TOP_K,
            )
    else:
        print("\n💬 Entering Chat Mode... (Type 'reset' to clear history, 'q' to quit)")
        conversation_history = ""
        while True:
            text = input("📝 You: ")
            if text.lower() == "q":
                break
            if text.lower() == "reset":
                conversation_history = ""
                print("🧹 Conversation history cleared.")
                continue

            conversation_history = chat(
                trained_model,
                proc,
                conversation_history,
                text,
                temperature=config.TEMPERATURE,
                K=config.TOP_K,
            )

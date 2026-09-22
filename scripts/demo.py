"""Run the notebook's original five-turn memory demonstration."""

from dotenv import load_dotenv

from research_memory_agent.app import build_application


def main() -> None:
    load_dotenv()
    app = build_application()
    thread_id = "50000"
    prompts = [
        "Can you get me the paper MemGPT?",
        "Can you save the content of the paper?",
        "What are the main key takeaways from the paper?",
        "Summarize the conversation so far using your tool.",
        "What was my first question?",
    ]
    try:
        for prompt in prompts:
            print(f"\nUSER: {prompt}\nAGENT: {app.agent.ask(prompt, thread_id=thread_id)}")
    finally:
        app.close()


if __name__ == "__main__":
    main()

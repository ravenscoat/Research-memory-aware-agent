from __future__ import annotations

import argparse

from dotenv import load_dotenv

from .app import build_application


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Oracle-backed memory-aware research agent")
    command.add_argument("--thread", default="1", help="Conversation thread identifier")
    command.add_argument(
        "--init-db", action="store_true", help="Create required Oracle memory tables"
    )
    command.add_argument(
        "--reset-db",
        action="store_true",
        help="DROP and recreate memory tables (destructive)",
    )
    command.add_argument("query", nargs="*", help="One research question")
    return command


def main() -> None:
    load_dotenv()
    args = parser().parse_args()
    app = build_application()
    try:
        if args.reset_db:
            app.store.initialize(drop_existing=True)
            app.tools.persist_all()
            print("Memory tables recreated.")
        elif args.init_db:
            app.store.initialize()
            print("Memory tables initialized.")

        if args.query:
            print(app.agent.ask(" ".join(args.query), thread_id=args.thread))
            return

        if args.init_db or args.reset_db:
            return

        print("Memory-aware research agent. Type 'exit' to stop.")
        while True:
            query = input("\nYou: ").strip()
            if query.casefold() in {"exit", "quit"}:
                break
            if query:
                print("\nAgent:", app.agent.ask(query, thread_id=args.thread))
    finally:
        app.close()


if __name__ == "__main__":
    main()

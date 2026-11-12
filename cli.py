"""
CLI Interface for College Admission RAG Agent
Run: python cli.py
"""

import os
import sys
import logging
from dotenv import load_dotenv
from colorama import Fore, Style, init
from tabulate import tabulate

load_dotenv()
init(autoreset=True)

logging.basicConfig(level=logging.WARNING)


BANNER = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║      🎓  Greenfield University — Admission Agent          ║
║         Powered by IBM Granite & watsonx.ai (RAG)         ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}

Type your question and press Enter. Type {Fore.YELLOW}'quit'{Style.RESET_ALL} to exit.
Type {Fore.YELLOW}'help'{Style.RESET_ALL} to see sample questions.
"""

SAMPLE_QUESTIONS = [
    "What are the undergraduate admission requirements?",
    "What is the tuition fee for international students?",
    "When is the application deadline?",
    "What scholarships are available?",
    "Tell me about the Computer Science program",
    "What documents do international students need?",
    "Is housing guaranteed for freshmen?",
    "What is the acceptance rate?",
]


def print_help():
    print(f"\n{Fore.YELLOW}Sample Questions:{Style.RESET_ALL}")
    for i, q in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"  {Fore.CYAN}{i}.{Style.RESET_ALL} {q}")
    print()


def format_sources(sources):
    if not sources:
        return ""
    rows = [[s.title, s.category.replace("_", " ").title()] for s in sources]
    return "\n" + tabulate(rows, headers=["Source", "Category"], tablefmt="simple")


def main():
    print(BANNER)

    # Import here so startup errors are surfaced clearly
    try:
        from src.agent import create_agent
        print(f"{Fore.YELLOW}⚙  Initializing agent (first run indexes the knowledge base)...{Style.RESET_ALL}")
        agent = create_agent()
        print(f"{Fore.GREEN}✓  Agent ready!{Style.RESET_ALL}\n")
    except ValueError as e:
        print(f"\n{Fore.RED}Configuration Error:{Style.RESET_ALL} {e}")
        print(f"\nEdit your {Fore.YELLOW}.env{Style.RESET_ALL} file with valid IBM Cloud credentials and retry.")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Fore.RED}Error initializing agent:{Style.RESET_ALL} {e}")
        sys.exit(1)

    while True:
        try:
            question = input(f"{Fore.CYAN}You:{Style.RESET_ALL} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Fore.YELLOW}Goodbye! Good luck with your application! 🎓{Style.RESET_ALL}")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print(f"\n{Fore.YELLOW}Goodbye! Good luck with your application! 🎓{Style.RESET_ALL}")
            break
        if question.lower() in ("help", "?"):
            print_help()
            continue

        print(f"\n{Fore.YELLOW}⏳ Thinking...{Style.RESET_ALL}")
        try:
            response = agent.query(question)
            print(f"\n{Fore.GREEN}Agent:{Style.RESET_ALL}")
            print(response.answer)
            if response.sources:
                print(f"\n{Fore.CYAN}📚 Retrieved from:{Style.RESET_ALL}{format_sources(response.sources)}")
            print()
        except Exception as e:
            print(f"\n{Fore.RED}Error:{Style.RESET_ALL} {e}\n")


if __name__ == "__main__":
    main()

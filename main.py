import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


def main():
    load_dotenv()
    print("Hello from project-005-rag-docs-assistant!")


if __name__ == "__main__":
    main()

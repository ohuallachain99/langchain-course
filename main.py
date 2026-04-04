import os

from dotenv import load_dotenv
from langchain_core import agents
from langchain_ollama import ChatOllama
from tavily import

load_dotenv()


def main():
    print("Hello from project-003-agents-under-the-hood!")

    llm = ChatOllama(model="qwen2.5:7b", temperature=0)


if __name__ == "__main__":
    main()
    main()

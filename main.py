import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    information = """
Jordan Bernt Peterson was born on June 12, 1962, in Edmonton, Alberta, and grew up in the small town of Fairview.Parents: His father, Walter, was a school teacher, and his mother, Beverley, was a librarian at a local college. He was the eldest of three children.Middle Name: His middle name, Bernt, was given in honor of his Norwegian great-grandfather.Early Influence: During junior high, he became close friends with Rachel Notley (who later became the Premier of Alberta). Her mother, Sandy Notley, was a librarian who introduced Peterson to authors like George Orwell and Aldous Huxley, sparking his intellectual curiosity.Political Roots and DisenchantmentIn his teens, Peterson was highly active in left-wing politics.He joined the New Democratic Party (NDP) at age 13 and remained a member until he was 18.By the time he reached university, he became disillusioned with political activism. He famously noted that he found the activists "too often fueled by a hatred of the rich rather than a love for the poor," leading him to shift his focus from social engineering to the psychology of the individual."""

    summary_template = """
    given the information {information} about a person, I want you to create:
    1) A short summary (max 100 words)
    2) Two interesting facts about them (max 1 sentence and point per fact)
    3) tell a jordan peterson joke
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatOllama(
    #     model="deepseek-r1:1.5b",
    #     temperature=0,
    # )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})

    print(response.content)


if __name__ == "__main__":
    main()

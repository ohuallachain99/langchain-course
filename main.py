import os
from typing import List

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The url of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


llm = ChatOllama(model="qwen2.5:7b", temperature=0)
tools = [TavilySearch()]
system_prompt = "You are a helpful research assistant. Use the search tool and be concise in your answers"
agent = create_agent(
    model=llm, tools=tools, system_prompt=system_prompt, response_format=AgentResponse
)


def main():
    print("starting project..")

    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="Find me a job posting on upwork for an LMS developer using NextJs or Javascript/Typescript."
            )
        }
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()

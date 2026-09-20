from dotenv import load_dotenv

from pydantic import BaseModel,Field
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()

tavil_client = TavilyClient()

class Source(BaseModel):
    """ Schema for sources used by the agent"""

    url:str = Field(description="The URL of the source")


class AgenResponse(BaseModel):
    """ Schema for an agent  response with answer and sources"""

    answer:str = Field(description="The angent's answer to the query")
    sources:list[Source] = Field (default_factory=list,description="list of sources used to generate the answers")
    



# @tool
# def search(query: str):
#     """
#     Tool that searches over the internet
#     Args:
#         querry: The query to search for.
#     Retruns:
#         The Search result

#     """

#     print(f"Searching for {query}")
#     return tavil_client.search(query=query)


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
)
# tool = [search]
tool = [TavilySearch()]
agent = create_agent(model=llm, tools=tool, response_format=AgenResponse)


def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke(
    {
        "messages": [
            HumanMessage(content="search for Ai engineers job opening for langchanin langraph ai llm and make sure these must be remote or guragon and noida only")
        ]
    }
)
    print(result)


if __name__ == "__main__":
    main()

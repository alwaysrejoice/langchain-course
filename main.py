from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool thats Searches the web for the given query and returns the results.
    Args:
        query (str): The search query.
    Returns:
        str: The search results.
    """
    print(f"Search results for {query}")
    return tavily.search(query=query)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    response = agent.invoke({"messages":HumanMessage(content="What's the weather in Tokyo?")})
    print(response)

if __name__ == "__main__":
    main()

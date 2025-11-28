import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_aws import ChatBedrock
load_dotenv()


def main():
    #print(os.environ.get("GOOGLE_API_KEY"))
    information = '''Elon Reeve Musk[b] (born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, Twitter, and xAI. Musk has been the wealthiest person in the world since 2021; as of October 2025, Forbes estimates his net worth to be around $500 billion.
Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; his Canadian citizenship is congenital, his mother having been born there. He received bachelor's degrees in 1997 from the University of Pennsylvania in Philadelphia, United States, before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became an American citizen.'''
    summary_template ='''given the information {information} about a person I want you to create:
    1. A short summary about the person.
    2. A list of 3 unique facts about the person.'''
    prompt = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )
    #llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
    #llm = ChatOllama(model="llama3.2:1b", temperature=0)
    llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0", temperature=0)
    chain = prompt | llm
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()

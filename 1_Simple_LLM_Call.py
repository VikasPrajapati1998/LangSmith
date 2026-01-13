from dotenv import load_dotenv, find_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv(find_dotenv())
os.environ['LANGCHAIN_PROJECT'] = "Simple_LLM"

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

model = ChatOllama(
    model="llama3.2:1b",   # llama3.2:1b, qwen2.5:0.5b
    temperature=0.3
)
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of India?"})
print(result)


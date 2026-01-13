from langchain_ollama import ChatOllama
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv(find_dotenv())
os.environ['LANGCHAIN_PROJECT'] = "Sequential_Chain_V1"

# =============== Prompts =====================
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

# =============== Model =====================
model = ChatOllama(
    model="qwen2.5:0.5b",   # llama3.2:1b, qwen2.5:0.5b
    temperature=0.3
)

# =============== Parser =====================
parser = StrOutputParser()

# =============== Chain =====================
chain = prompt1 | model | parser | prompt2 | model | parser

# =============== Invoke =====================
result = chain.invoke({'topic': 'Unemployment in India'})
print(result)

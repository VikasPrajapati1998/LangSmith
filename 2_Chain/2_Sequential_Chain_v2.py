from langchain_ollama import ChatOllama
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv(find_dotenv())
os.environ['LANGCHAIN_PROJECT'] = "Sequential_Chain_V2"

# =============== Prompts =====================
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

# ================ Models =====================
model1 = ChatOllama(
    model="llama3.2:1b",   # llama3.2:1b, qwen2.5:0.5b
    temperature=0.4
)

model2 = ChatOllama(
    model="qwen2.5:0.5b",   # llama3.2:1b, qwen2.5:0.5b
    temperature=0.7
)

# ================ Parser ======================
parser = StrOutputParser()

# ================ Chain =======================
chain = prompt1 | model1 | parser | prompt2 | model2 | parser

# ================ Invoke ======================
config = {
    'run_name': 'SequentialChai-V2',
    'tags': ['llm app', 'report generation', 'summarization'],
    'metadata': {'model1': 'llama3.2:1b', 'model1': 'qwen2.5:0.5b'}
}

result = chain.invoke({'topic': 'Generative AI'}, config=config)
print(result)


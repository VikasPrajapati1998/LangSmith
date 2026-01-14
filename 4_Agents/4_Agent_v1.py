import os
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
os.environ['LANGCHAIN_PROJECT'] = "Agent-v1"

# 1. Configure DuckDuckGo for stability
# We use a wrapper to specify max_results and a 'lite' backend if possible
wrapper = DuckDuckGoSearchAPIWrapper(max_results=5,
                                     time="y",
                                     region="wt-wt",
                                     safesearch="moderate",
                                     backend="api"  # "html", "lite", "auto"
                                    )
search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)

# 2. Setup LLM with a sane temperature (0 is best for accuracy)
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0 
)

# 3. Pull the ReAct prompt
prompt = hub.pull("hwchase17/react")

# 4. Create the ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=[search_tool],
    prompt=prompt
)

# 5. CONFIGURE EXECUTOR FOR SINGLE SEARCH
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True,
    max_iterations=2,               # 1 turn for Search, 1 turn for Final Answer
    early_stopping_method="generate", # Ensures it answers even if it hits the limit
    handle_parsing_errors=True
)

# 6. Invoke
user_input = "List some of the latest released and upcoming Bollywood movies for 2025. Perform only one search."
response = agent_executor.invoke({"input": user_input})
print("\n--- AGENT RESPONSE ---")
print(response['output'])


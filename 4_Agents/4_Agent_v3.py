import os
from langchain_ollama import ChatOllama
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
os.environ['LANGCHAIN_PROJECT'] = "Agent-v3"

# Get API key from https://serper.dev (add to your .env file)
# SERPER_API_KEY=your-api-key-here
os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

# 1. Configure GoogleSerper search
search = GoogleSerperAPIWrapper()
search_tool = Tool(
    name="search",
    func=search.run,
    description="useful for searching the internet"
)

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
    max_iterations=3,               # Allow enough iterations: 1 for search, 1 for thought, 1 for final answer
    early_stopping_method="force",  # Changed from "generate" to "force"
    handle_parsing_errors=True,
    return_intermediate_steps=False
)

# 6. Invoke
user_input = "List some of the latest released and upcoming Bollywood movies for 2025."
response = agent_executor.invoke({"input": user_input})
print("\n--- AGENT RESPONSE ---")
print(response['output'])

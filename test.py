import os
import json
import urllib.request
from dotenv import load_dotenv, find_dotenv
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

# Load environment variables
load_dotenv(find_dotenv())

# Configure LangSmith
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'Agent'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

# Initialize search tool
search_tool = DuckDuckGoSearchRun()


@tool
@traceable(name="Weather Tool", tags=["tool", "weather"])
def get_weather_data(city: str) -> str:
    """
    Fetches the current weather data for a given city using wttr.in API.
    
    Args:
        city: Name of the city to get weather for
        
    Returns:
        A formatted string with weather information
    """
    try:
        city = city.strip()
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
        
        print(f"Fetching weather for: {city}")
        
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
        
        if "current_condition" not in data or not data["current_condition"]:
            return f"No weather data available for {city}"
        
        current = data["current_condition"][0]
        
        result = (f"Weather in {city}: "
                  f"{current['weatherDesc'][0]['value']}, "
                  f"Temperature: {current['temp_C']}°C, "
                  f"Feels like: {current['FeelsLikeC']}°C, "
                  f"Humidity: {current['humidity']}%, "
                  f"Wind speed: {current['windspeedKmph']} km/h")
        
        print(f"✓ Weather data retrieved")
        return result
    
    except Exception as e:
        return f"Error fetching weather for {city}: {str(e)}"


@traceable(name="Search Tool", tags=["tool", "search"])
def search_web(query: str) -> str:
    """Search the web for information."""
    try:
        print(f"Searching web for: {query}")
        result = search_tool.run(query)
        print(f"✓ Search completed")
        return result
    except Exception as e:
        return f"Error searching: {str(e)}"


@traceable(name="Initialize LLM", tags=["setup"])
def initialize_llm(model_name: str = "llama3.2:1b", temperature: float = 0.7):
    """Initialize the language model."""
    print(f"Initializing LLM: {model_name}")
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        num_predict=512
    )


@traceable(name="Simple Agent Chain", tags=["agent"])
def create_simple_agent_chain(llm):
    """
    Creates a simple chain that acts like an agent.
    Uses a straightforward approach without complex agent frameworks.
    """
    print("Creating agent chain...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. You have access to two tools:

1. SEARCH: Use this to search the web for current information
   Format: SEARCH: your search query

2. WEATHER: Use this to get weather for a city
   Format: WEATHER: city name

When you need to use a tool, respond with EXACTLY the format above.
After getting the tool result, provide a final answer to the user.

If you don't need tools, just answer directly."""),
        ("human", "{input}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain


@traceable(name="Process Query", tags=["execution"])
def process_query(query: str, llm) -> str:
    """
    Process a query using a simple agent-like approach.
    """
    print(f"\n{'='*80}")
    print(f"Processing: {query}")
    print('='*80)
    
    chain = create_simple_agent_chain(llm)
    max_iterations = 5
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Get response from LLM
        response = chain.invoke({"input": query})
        print(f"LLM Response: {response[:200]}...")
        
        # Check if LLM wants to use a tool
        if response.startswith("SEARCH:"):
            search_query = response.replace("SEARCH:", "").strip()
            tool_result = search_web(search_query)
            query = f"Original question: {query}\n\nSearch results: {tool_result}\n\nProvide the final answer."
            
        elif response.startswith("WEATHER:"):
            city = response.replace("WEATHER:", "").strip()
            tool_result = get_weather_data(city)
            query = f"Original question: {query}\n\nWeather data: {tool_result}\n\nProvide the final answer."
            
        else:
            # No tool needed, this is the final answer
            print(f"\n✓ Final answer generated")
            return response
    
    return "Maximum iterations reached. Please try rephrasing your question."


@traceable(name="Main Execution Flow")
def main():
    """
    Main function to run the agent with example queries.
    """
    print("\n" + "="*80)
    print("SIMPLE AGENT PIPELINE STARTED")
    print("="*80)
    
    # Example queries
    queries = [
        "What is the current temp of gurgaon",
        # "What is the release date of Dhadak 2?",
        # "Identify the birthplace city of Kalpana Chawla and give its current temperature.",
    ]
    
    # Initialize LLM once
    llm = initialize_llm(model_name="llama3.2:1b", temperature=0.7)
    
    for idx, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {idx}/{len(queries)}")
        print('='*80)
        
        try:
            answer = process_query(query, llm)
            
            print("\n" + "="*80)
            print("FINAL ANSWER:")
            print("="*80)
            print(answer)
            print("="*80)
            
        except KeyboardInterrupt:
            print("\n\nExecution interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED")
    print("="*80)


if __name__ == "__main__":
    # Verify LangSmith configuration
    if not os.getenv('LANGCHAIN_API_KEY'):
        print("⚠️  Warning: LANGCHAIN_API_KEY not found.")
        print("LangSmith tracing will not work without a valid API key.\n")
    else:
        print("✓ LangSmith tracing enabled")
        print(f"✓ Project: {os.getenv('LANGCHAIN_PROJECT')}\n")
    
    main()
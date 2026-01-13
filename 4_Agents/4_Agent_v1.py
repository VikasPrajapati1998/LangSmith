import os
import json
import urllib.request
from dotenv import load_dotenv, find_dotenv
from langchain import hub
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langsmith import traceable

# Load environment variables
load_dotenv(find_dotenv())

# Configure LangSmith
# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'Agent'

# Initialize search tool
search_tool = DuckDuckGoSearchRun()


@traceable(name="Weather Tool")
@tool
def get_weather_data(city: str) -> str:
    """
    Fetches the current weather data for a given city using wttr.in API.
    
    Args:
        city: Name of the city to get weather for
        
    Returns:
        A formatted string with weather information including temperature,
        condition, humidity, and wind speed.
    """
    try:
        # Clean and encode city name
        city = city.strip()
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
        
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
        
        # Check if we got valid data
        if "current_condition" not in data or not data["current_condition"]:
            return f"No weather data available for {city}"
        
        current = data["current_condition"][0]
        
        weather_dict = {
            "city": city,
            "condition": current["weatherDesc"][0]["value"],
            "temperature_celsius": int(current["temp_C"]),
            "feels_like_celsius": int(current["FeelsLikeC"]),
            "humidity_percent": int(current["humidity"]),
            "wind_speed_kmph": int(current["windspeedKmph"])
        }
        
        # Return formatted string for better agent parsing
        return (f"Weather in {weather_dict['city']}: "
                f"{weather_dict['condition']}, "
                f"Temperature: {weather_dict['temperature_celsius']}°C, "
                f"Feels like: {weather_dict['feels_like_celsius']}°C, "
                f"Humidity: {weather_dict['humidity_percent']}%, "
                f"Wind speed: {weather_dict['wind_speed_kmph']} km/h")
    
    except urllib.error.HTTPError as e:
        return f"HTTP error fetching weather for {city}: {e.code} - {e.reason}"
    except urllib.error.URLError as e:
        return f"Network error fetching weather for {city}: {str(e.reason)}"
    except json.JSONDecodeError:
        return f"Error parsing weather data for {city}"
    except KeyError as e:
        return f"Unexpected weather data format for {city}: missing key {str(e)}"
    except Exception as e:
        return f"Error fetching weather data for {city}: {str(e)}"


@traceable(name="Setup Pipeline")
def setup_pipeline():
    """
    Initializes and configures the ReAct agent with LLM and tools.
    
    Returns:
        AgentExecutor: Configured agent executor ready to process queries
    """
    # Initialize LLM with Ollama
    llm = ChatOllama(
        model="llama3.2:1b",
        temperature=0.7,
        num_predict=512  # Limit response length
    )

    # Pull the ReAct prompt from LangChain Hub
    prompt = hub.pull("hwchase17/react")

    # Create the ReAct agent
    agent = create_react_agent(
        llm=llm,
        tools=[search_tool, get_weather_data],
        prompt=prompt
    )

    # Wrap with AgentExecutor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[search_tool, get_weather_data],
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_execution_time=60  # Timeout after 60 seconds
    )
    
    return agent_executor


@traceable(name="Agent Execution")
def run_agent(agent_executor: AgentExecutor, query: str) -> dict:
    """
    Executes the agent with the given query.
    
    Args:
        agent_executor: The configured agent executor
        query: The user's question or instruction
        
    Returns:
        dict: Response containing output and intermediate steps
    """
    return agent_executor.invoke({"input": query})


@traceable(name="Main")
def main():
    """
    Main function to run the agent with example queries.
    """
    # Example queries - uncomment to test different scenarios:
    queries = [
        # "What is the release date of Dhadak 2?",
        "What is the current temp of gurgaon",
        # "Identify the birthplace city of Kalpana Chawla and give its current temperature.",
    ]
    
    for query in queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)
        
        try:
            agent_executor = setup_pipeline()
            response = run_agent(agent_executor, query)
            
            print("\n" + "="*80)
            print("FINAL ANSWER:")
            print(response['output'])
            print("="*80)
            
            # Optional: Display intermediate steps for debugging
            if 'intermediate_steps' in response and response['intermediate_steps']:
                print("\nIntermediate Steps:")
                for i, (action, observation) in enumerate(response['intermediate_steps'], 1):
                    print(f"\nStep {i}:")
                    print(f"  Action: {action.tool} - {action.tool_input}")
                    print(f"  Observation: {observation[:200]}...")  # Truncate long observations
            
        except KeyboardInterrupt:
            print("\n\nExecution interrupted by user.")
            break
        except Exception as e:
            print(f"\nError executing agent: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Verify LangSmith configuration
    if not os.getenv('LANGCHAIN_API_KEY'):
        print("Warning: LANGCHAIN_API_KEY not found in environment variables.")
        print("LangSmith tracing will not work without a valid API key.")
        print("Set it in your .env file or export it as an environment variable.\n")
    
    main()

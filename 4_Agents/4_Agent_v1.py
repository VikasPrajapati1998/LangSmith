import os
import json
import urllib.request
from dotenv import load_dotenv, find_dotenv
from langchain import hub
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langsmith import traceable

# Load environment variables
load_dotenv(find_dotenv())
os.environ['LANGCHAIN_PROJECT'] = 'Agent'


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
        if 'city=' in city.lower():
            city = city.split('=', 1)[1]
        
        city = city.strip().strip("'\"").strip()
        
        if not city:
            return "Error: City name is empty after processing"
        
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
        
        with urllib.request.urlopen(url, timeout=15) as response:
            data = json.loads(response.read().decode())
        
        if "current_condition" not in data or not data["current_condition"]:
            return f"No weather data available for {city}"
        
        current = data["current_condition"][0]
        
        weather_dict = {
            "city": city.title(),
            "condition": current["weatherDesc"][0]["value"],
            "temperature_celsius": int(current["temp_C"]),
            "feels_like_celsius": int(current["FeelsLikeC"]),
            "humidity_percent": int(current["humidity"]),
            "wind_speed_kmph": int(current["windspeedKmph"])
        }
        
        # Return formatted string for better agent parsing
        return (f"Current temperature in {weather_dict['city']} is {weather_dict['temperature_celsius']}°C. "
                f"Weather condition: {weather_dict['condition']}. "
                f"Feels like: {weather_dict['feels_like_celsius']}°C. "
                f"Humidity: {weather_dict['humidity_percent']}%. "
                f"Wind speed: {weather_dict['wind_speed_kmph']} km/h.")
    
    except urllib.error.HTTPError as e:
        return f"HTTP error fetching weather for {city}: {e.code} - {e.reason}"
    except urllib.error.URLError as e:
        return f"Network error fetching weather for {city}. The service might be temporarily unavailable. Please try again."
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
        model="llama3.2:3b",
        temperature=0,
        num_predict=512,
        num_ctx=2048,
        top_p=0.9,
        repeat_penalty=1.1
    )
    
    # Pull the ReAct prompt template
    prompt = hub.pull("hwchase17/react")
    tools = [get_weather_data]
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_execution_time=60
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
    try:
        return agent_executor.invoke({"input": query})
    except Exception as e:
        return {
            "output": f"I encountered an issue: {str(e)}. Please try again.",
            "intermediate_steps": []
        }


@traceable(name="Main")
def main():
    """
    Main function to run the agent with example queries.
    """
    # Example queries
    queries = [
        # "What is the current temp of delhi",
        # "What is the current temp of gurgaon",
        "What is the weather in Mumbai",
        # "Tell me the temperature in Bangalore",
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
                print("\n" + "-"*80)
                print("DEBUG: Intermediate Steps")
                print("-"*80)
                for i, (action, observation) in enumerate(response['intermediate_steps'], 1):
                    print(f"\nStep {i}:")
                    print(f"  Tool: {action.tool}")
                    print(f"  Input: {action.tool_input}")
                    print(f"  Output: {observation[:200]}...")
                print("-"*80)
            
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
        print("⚠️  Warning: LANGCHAIN_API_KEY not found in environment variables.")
        print("   LangSmith tracing will not work without a valid API key.")
        print("   Set it in your .env file or export it as an environment variable.\n")
    
    print("="*80)
    print("Weather Agent - Powered by llama3.2:3b")
    print("Available tools: get_weather_data")
    print("="*80)
    
    main()


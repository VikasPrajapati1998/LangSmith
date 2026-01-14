import os
import requests
from dotenv import load_dotenv, find_dotenv
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langsmith import traceable

# Load environment variables
load_dotenv(find_dotenv())
os.environ['LANGCHAIN_PROJECT'] = 'Agent-v4'


# @tool
# def get_weather_data(city: str) -> str:
#     """
#     Fetches the current weather data for a given city using OpenWeatherMap API.
#     Use this tool when user asks about weather, temperature, or climate conditions.
    
#     Args:
#         city: Name of the city to get weather for (e.g., "Pune", "London", "New York")
        
#     Returns:
#         Weather information as a string containing all relevant data
#     """
#     try:
#         # Clean up city name
#         if 'city=' in city.lower():
#             city = city.split('=', 1)[1]
        
#         city = city.strip().strip("'\"").strip()
        
#         if not city:
#             return "Error: City name is empty after processing"
        
#         # Get API key from environment
#         api_key = os.getenv('OPENWEATHER_API_KEY')
        
#         if not api_key:
#             return "Error: OPENWEATHER_API_KEY not found in environment variables. Get free API key from https://openweathermap.org/api"
        
#         # OpenWeatherMap API endpoint
#         url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
        
#         data = response.json()
        
#         # Extract weather information
#         weather_info = (
#             f"City: {data['name']}, {data['sys']['country']}, "
#             f"Temperature: {data['main']['temp']}°C, "
#             f"Feels Like: {data['main']['feels_like']}°C, "
#             f"Condition: {data['weather'][0]['description'].title()}, "
#             f"Humidity: {data['main']['humidity']}%, "
#             f"Wind Speed: {data['wind']['speed']} m/s, "
#             f"Pressure: {data['main']['pressure']} hPa"
#         )
        
#         return weather_info
    
#     except requests.exceptions.HTTPError as e:
#         if e.response.status_code == 404:
#             return f"City '{city}' not found. Please check the city name and try again."
#         elif e.response.status_code == 401:
#             return "Invalid API key. Please check your OPENWEATHER_API_KEY."
#         else:
#             return f"HTTP error fetching weather for {city}: {e.response.status_code}"
    
#     except requests.exceptions.RequestException as e:
#         return f"Network error fetching weather for {city}. Please try again."
    
#     except KeyError as e:
#         return f"Unexpected weather data format for {city}: missing key {str(e)}"
    
#     except Exception as e:
#         return f"Error fetching weather data for {city}: {str(e)}"


@tool
def get_weather_data(city: str) -> str:
    """
    Fetches weather data using WeatherAPI.com
    """
    try:
        city = city.strip().strip("'\"").strip()
        api_key = os.getenv('WEATHERAPI_KEY')
        
        if not api_key:
            return "Error: WEATHERAPI_KEY not found. Get free key from https://www.weatherapi.com/signup.aspx"
        
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        weather_info = (
            f"City: {data['location']['name']}, {data['location']['country']}, "
            f"Temperature: {data['current']['temp_c']}°C, "
            f"Feels Like: {data['current']['feelslike_c']}°C, "
            f"Condition: {data['current']['condition']['text']}, "
            f"Humidity: {data['current']['humidity']}%, "
            f"Wind Speed: {data['current']['wind_kph']} km/h"
        )
        return weather_info
        
    except Exception as e:
        return f"Error: {str(e)}"


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
    
    # Configure GoogleSerper search
    search = GoogleSerperAPIWrapper()
    search_tool = Tool(
        name="search",
        func=search.run,
        description="useful for searching the internet for current information, news, and general knowledge"
    )
    
    # List of all tools
    tools = [get_weather_data, search_tool]
    
    # Pull the ReAct prompt from LangChain Hub
    prompt = hub.pull("hwchase17/react")
    
    # Create the ReAct agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    # Configure agent executor
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
    query = "What is the weather of Pune, India." 
    agent_executor = setup_pipeline()
    response = run_agent(agent_executor, query)
    print(response['output'])


if __name__ == "__main__":
    # Verify LangSmith configuration
    if not os.getenv('LANGCHAIN_API_KEY'):
        print("⚠️  Warning: LANGCHAIN_API_KEY not found in environment variables.")
        print("   LangSmith tracing will not work without a valid API key.")
        print("   Set it in your .env file or export it as an environment variable.\n")
        
    main()


from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv, find_dotenv
import time
import requests

load_dotenv(find_dotenv())

# Configure DuckDuckGo with better settings
ddg_wrapper = DuckDuckGoSearchAPIWrapper(
    max_results=3,
    region="wt-wt",
    safesearch="moderate",
    backend="api"
)

search_tool = DuckDuckGoSearchRun(api_wrapper=ddg_wrapper)

@tool
def get_weather_data(city: str) -> dict:
    """
    Fetches the current weather data for a given city using Open-Meteo API.
    Use this tool when the user asks about weather, temperature, or climate conditions.
    Returns temperature, wind speed, weather code, and coordinates.
    """
    # ------------------ Step 1: Geocoding ------------------
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    geo_response = requests.get(geo_url, params=geo_params, timeout=10)
    geo_data = geo_response.json()

    if "results" not in geo_data or not geo_data["results"]:
        return {
            "city": city,
            "error": "City not found",
            "source": "open-meteo"
        }

    location = geo_data["results"][0]
    latitude = location["latitude"]
    longitude = location["longitude"]

    # ------------------ Step 2: Weather ------------------
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": True
    }

    weather_response = requests.get(weather_url, params=weather_params, timeout=10)
    weather_data = weather_response.json()

    current = weather_data.get("current_weather", {})

    return {
        "city": city,
        "latitude": latitude,
        "longitude": longitude,
        "temperature_c": current.get("temperature"),
        "windspeed_kmh": current.get("windspeed"),
        "winddirection": current.get("winddirection"),
        "weathercode": current.get("weathercode"),
        "time": current.get("time"),
        "source": "open-meteo"
    }

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")

# Create the ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Wrap it with AgentExecutor with better error handling
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
    return_intermediate_steps=False
)

# Test queries
print("=" * 80)
print("Query 1: Movie Release Date")
print("=" * 80)
result1 = agent_executor.invoke({"input": "What is the release date of movie Dhadak 2?"})
print(f"\nFinal Answer: {result1['output']}")
print("\n" * 2)

# Add delay between queries to avoid rate limiting
time.sleep(5)

print("=" * 80)
print("Query 2: Current Temperature")
print("=" * 80)
result2 = agent_executor.invoke({"input": "What is the current temp of gurgaon?"})
print(f"\nFinal Answer: {result2['output']}")
print("\n" * 2)

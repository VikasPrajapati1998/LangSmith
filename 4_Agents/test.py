import os
import json
import urllib.request
from dotenv import load_dotenv, find_dotenv
from langchain import hub
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langsmith import traceable
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv(find_dotenv())
os.environ['LANGCHAIN_PROJECT'] = 'Agent-v2'


# Pydantic model for structured weather output
class WeatherResponse(BaseModel):
    """Structured weather response format"""
    summary: str = Field(description="A natural language summary of the weather")
    temperature_description: str = Field(description="Description of temperature and how it feels")
    conditions_description: str = Field(description="Description of weather conditions and additional details")


@tool
def get_weather_data(city: str) -> str:
    """
    Fetches the current weather data for a given city using wttr.in API.
    
    Args:
        city: Name of the city to get weather for
        
    Returns:
        Weather information as a string containing all relevant data
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
        
        # Return simple string with weather data
        weather_info = (
            f"City: {city.title()}, "
            f"Temperature: {current['temp_C']}°C, "
            f"Feels Like: {current['FeelsLikeC']}°C, "
            f"Condition: {current['weatherDesc'][0]['value']}, "
            f"Humidity: {current['humidity']}%, "
            f"Wind Speed: {current['windspeedKmph']} km/h"
        )
        
        return weather_info
    
    except urllib.error.HTTPError as e:
        return f"HTTP error fetching weather for {city}: {e.code} - {e.reason}"
    except urllib.error.URLError as e:
        return f"Network error fetching weather for {city}. Please try again."
    except json.JSONDecodeError:
        return f"Error parsing weather data for {city}"
    except KeyError as e:
        return f"Unexpected weather data format for {city}: missing key {str(e)}"
    except Exception as e:
        return f"Error fetching weather data for {city}: {str(e)}"


@tool
def format_weather_data(weather_data: str) -> str:
    """
    Uses AI (qwen2.5:0.5b) to format raw weather data into a natural, human-readable response.
    This provides the FINAL answer using structured output with Pydantic.
    
    Args:
        weather_data: Raw weather data string from get_weather_data
        
    Returns:
        A beautifully formatted, natural language weather report
    """
    try:
        # Check if it's an error message
        if "Error:" in weather_data or "City:" not in weather_data:
            return weather_data
        
        # Initialize the lightweight LLM for formatting
        formatting_llm = ChatOllama(
            model="qwen2.5:0.5b",
            temperature=0.7,
            format="json"
        )
        
        # Use structured output with Pydantic
        structured_llm = formatting_llm.with_structured_output(WeatherResponse)
        
        # Create prompt for the formatting LLM
        prompt = f"""You are a weather report formatter. Convert this weather data into natural, engaging language.

Weather Data: {weather_data}

Create a friendly, conversational weather report with three parts:
1. A brief summary sentence
2. A description of the temperature and how it feels
3. Details about conditions, humidity, and wind

Be natural and conversational, like a weather reporter speaking to viewers."""
        
        # Get structured response
        response = structured_llm.invoke(prompt)
        
        # Combine the structured parts into a flowing narrative
        formatted_text = f"{response.summary} {response.temperature_description} {response.conditions_description}"
        
        return formatted_text
    
    except Exception as e:
        # If AI formatting fails, fall back to simple formatting
        try:
            parts = {}
            for item in weather_data.split(", "):
                if ": " in item:
                    key, value = item.split(": ", 1)
                    parts[key.strip()] = value.strip()
            
            return (
                f"The current weather in {parts.get('City', 'the city')} is "
                f"{parts.get('Temperature', 'unknown')} with {parts.get('Condition', 'unknown conditions').lower()}. "
                f"It feels like {parts.get('Feels Like', 'unknown')}. "
                f"The humidity is {parts.get('Humidity', 'unknown')} and "
                f"wind speed is {parts.get('Wind Speed', 'unknown')}."
            )
        except:
            return weather_data


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
    
    # Now we have TWO tools
    tools = [get_weather_data, format_weather_data]
    
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=4,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_execution_time=60,
        early_stopping_method="generate"
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
    query = "What is the weather in Pune"
    agent_executor = setup_pipeline()
    response = run_agent(agent_executor, query)
    
    print("\n" + "="*80)
    print("FINAL ANSWER:")
    print(response['output'])
    print("="*80)


if __name__ == "__main__":
    # Verify LangSmith configuration
    if not os.getenv('LANGCHAIN_API_KEY'):
        print("⚠️  Warning: LANGCHAIN_API_KEY not found in environment variables.")
        print("   LangSmith tracing will not work without a valid API key.")
        print("   Set it in your .env file or export it as an environment variable.\n")
    
    print("="*80)
    print("Weather Agent - Powered by llama3.2:3b")
    print("Available tools: get_weather_data, format_weather_data (with qwen2.5:0.5b)")
    print("="*80)
    
    main()

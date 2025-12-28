import warnings
warnings.filterwarnings("ignore")

from langchain_groq import ChatGroq
from langchain_community.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import requests
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("OPENWEATHER_API_KEY")

# --- Tool Definitions ---
search_tool = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

@tool
def get_weather(city: str) -> dict:
    """
    Get current weather and AQI for ANY city using Open-Meteo (no API key)
    """

    # 1 Convert city name to latitude & longitude
    geo_url = (
        f"https://geocoding-api.open-meteo.com/v1/search"
        f"?name={city}&count=1"
    )
    geo_resp = requests.get(geo_url, timeout=5).json()

    if "results" not in geo_resp:
        return {"error": f"City '{city}' not found"}

    lat = geo_resp["results"][0]["latitude"]
    lon = geo_resp["results"][0]["longitude"]

    # 2 Get weather
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current_weather=true"
    )
    weather = requests.get(weather_url, timeout=5).json()

    # 3 Get AQI
    air_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}&current=us_aqi"
    )
    air = requests.get(air_url, timeout=5).json()

    return {
        "city": city,
        "temperature": weather["current_weather"]["temperature"],
        "windspeed": weather["current_weather"]["windspeed"],
        "aqi": air["current"]["us_aqi"]
    }

# --- LLM Setup ---
# llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0, streaming=False)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

# --- Agent Setup ---
# Pulling the prompt
prompt = hub.pull("hwchase17/react")

# Create agent
agent = create_react_agent(
    llm=llm,
    tools=[get_weather, wikipedia],
    prompt=prompt
)

# Wrap agent with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[ get_weather, wikipedia],
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,  # Stop after 3 attempts
    max_execution_time=30,  # Stop after 30 seconds
    return_intermediate_steps=False  # Don't clutter output with failed attempts
)

# --- Execution ---
response = agent_executor.invoke({"input": "when did the stree2 released in india"})
print("\nFINAL ANSWER:")
print(response['output'])
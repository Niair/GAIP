from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from typing import Annotated
from langchain_core.tools import InjectedToolArg
import requests
import os

load_dotenv()

EXCHANGE_RATE_API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")


# ---------------------------------------------------------
# TOOLS
# ---------------------------------------------------------
@tool
def get_currency_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Fetch live conversion factor for base -> target currency."""
    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_API_KEY}/pair/{base_currency}/{target_currency}"
    data = requests.get(url).json()
    return data.get("conversion_rate")


@tool
def convert(amount: float, rate: Annotated[float, InjectedToolArg]) -> float:
    """Convert amount with the conversion rate."""
    return amount * rate


tools = [get_currency_conversion_factor, convert]


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------
llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0)


# ---------------------------------------------------------
# CORRECT PROMPT (WITH agent_scratchpad)
# ---------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a currency conversion assistant. Use tools when helpful."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])


# ---------------------------------------------------------
# BUILD TOOL CALLING AGENT
# ---------------------------------------------------------
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


# ---------------------------------------------------------
# RUN AGENT
# ---------------------------------------------------------
response = agent_executor.invoke({
    "input": "What is the current USD → INR rate? Convert 10 USD to INR."
})

print("\nFINAL OUTPUT:\n", response["output"])

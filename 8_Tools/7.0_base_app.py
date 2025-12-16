from langchain_community.tools import tool
from langchain_core.tools import InjectedToolArg
from typing import Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import requests
import os
load_dotenv()

EXCHANGE_RATE_API_KEY = os.getenv('EXCHANGE_RATE_API_KEY')

# tools
@tool
def get_currency_conversion_factor(base_currency : str, target_currency : str) -> float:
      """
      this function fetches the current currency conversion factor between a given base_currency and a target_currency from an api
      """
      url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_API_KEY}/pair/{base_currency}/{target_currency}"

      response = requests.get(url)

      return response.json()

@tool
def convert(base_currency_value : int, conversion_rate : Annotated[float, InjectedToolArg]) -> float:  # this is used to reduce halucination ----> Annotated[float, InjectedToolArg]
      """
      Given a currency conversion rate this function calculate the target currency value from a given base value
      """
      return base_currency_value * conversion_rate

# cr = get_currency_conversion_factor.invoke({'base_currency' : 'USD', 'target_currency' : 'INR'})['conversion_rate']
# print(convert.invoke({'base_currency_value' : 10, 'conversion_rate' : cr}))

# model
model = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.7)

# tool binding
llm_with_tools = model.bind_tools([convert, get_currency_conversion_factor])


# chat_history
messages = [
      SystemMessage(content = "You are an helpful AI asistant that is experinced in the currency cunversion, you use tools given for that also You can call multiple tools in one response. Do NOT wait for tool results. If needed, estimate values."),
      HumanMessage(content = "what is the current cumversion factor between usd and inr, and based on that convert 10 usd to inr")
]
ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)
print(ai_message.tool_calls)
from langchain_community.tools import tool, DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import requests
import os
load_dotenv()

EXCHANGE_RATE_API_KEY = os.getenv('EXCHANGE_RATE_API_KEY')

# tools
@tool
def convert(base_currency_value : int, conversion_rate : float) -> float:
      """
      Given a currency conversion rate this function calculate the target currency value from a given base value
      """
      return base_currency_value * conversion_rate

@tool
def get_currency_conversion_factor(base_currency : str, target_currency : str) -> float:
      """
      this function fetches the current currency conversion factor between a given base_currency and a target_currency from an api
      """
      url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_API_KEY}/pair/{base_currency}/{target_currency}"

      response = requests.get(url)

      return response.json()

# cr = get_currency_conversion_factor.invoke({'base_currency' : 'USD', 'target_currency' : 'INR'})['conversion_rate']
# print(convert.invoke({'base_currency_value' : 10, 'conversion_rate' : cr}))

# model
model = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.7)

# tool binding
model_with_tools = model.bind_tools([convert, get_currency_conversion_factor])


# chat_history
chat_history = [
      SystemMessage(content = "You are an helpful AI asistant that is experinced in the currency cunversion")
]


while True:
      user_input = input("You : ")
      if user_input == "exit":
            print(chat_history)
            break
      chat_history.append(HumanMessage(content = user_input))  # append HumanMessage

      while True:
            result = model_with_tools.invoke(chat_history)

            if not result.tool_calls:
                  chat_history.append(result)  # append AIMessaage
                  print("AI : ", result.content)
                  break

            chat_history.append(result)  # append AIMessage for Tools

            if result.tool_calls[0]['name'] == 'convert':
                  tool_result = convert.invoke(result.tool_calls[0])

            elif result.tool_calls[0]['name'] == 'get_currency_conversion_factor':
                  tool_result = get_currency_conversion_factor.invoke(result.tool_calls[0])
                  
            chat_history.append(tool_result)
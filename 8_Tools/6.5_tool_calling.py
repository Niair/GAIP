from langchain_community.tools import tool, DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import requests
load_dotenv()

# tools
@tool
def multiply(a : float, b : float) -> float:
      """
      Multiply two numbers
      """
      return a * b

@tool
def web_search(query : str) -> str:
      """
      Function to search the web browser
      """
      search = DuckDuckGoSearchRun()

      results = search.invoke(query)

      return results

# model
model = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.7)

# tool binding
model_with_tools = model.bind_tools([multiply, web_search])


# chat_history
chat_history = [
      SystemMessage(content = "You are an helpful AI asistant")
]


while True:
      user_input = input("You : ")
      if user_input == "exit":
            break
      chat_history.append(HumanMessage(content = user_input))  # append HumanMessage

      while True:
            result = model_with_tools.invoke(chat_history)

            if not result.tool_calls:
                  chat_history.append(result)  # append AIMessaage
                  print("AI : ", result.content)
                  break

            chat_history.append(result)  # append AIMessage for Tools

            if result.tool_calls[0]['name'] == 'multiply':
                  tool_result = multiply.invoke(result.tool_calls[0])

            elif result.tool_calls[0]['name'] == 'web_search':
                  tool_result = web_search.invoke(result.tool_calls[0])
                  
            chat_history.append(tool_result)
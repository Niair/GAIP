from langchain_community.tools import tool, DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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


# in this we did the tool execution
while True:
      user_input = input("You : ")
      chat_history.append(HumanMessage(content = user_input))
      if user_input == "exit":
            break

      result = model_with_tools.invoke(chat_history)
      chat_history.append(AIMessage(content = result.content))

      # print("AI : ", result.content)
      # print("AI : ", result.tool_calls) --> llm do not use the tools it just suggests what to input and what tool to use when
      if result.tool_calls:
            if result.tool_calls[0]['name'] == 'multiply':
                  val = multiply.invoke(result.tool_calls[0])
                  print("AI : ", val)
                  print("AI : ", val.content)

                  chat_history.append(multiply.invoke(result.tool_calls[0]))
                  print("AI : ", result.content)

            elif result.tool_calls[0]['name'] == 'web_search':
                  val = web_search.invoke(result.tool_calls[0])
                  print("AI : ", val)
                  print("AI : ", val.content)

      else:
            print("AI : ", result.content)















# while True:
#       user_input = input("You : ")
#       chat_history.append(HumanMessage(content = user_input))
#       if user_input == "exit":
#             break
# 
#       result = model_with_tools.invoke(chat_history)
#       chat_history.append(AIMessage(content = result.content))
#       # print("AI : ", result.content)
#       # print("AI : ", result.tool_calls) --> llm do not use the tools it just suggests what to input and what tool to use when
#       if result.tool_calls:
#             if result.tool_calls[0]['name'] == 'multiply':
#                   val = multiply.invoke(result.tool_calls[0]['args'])
#                   print("AI : ", val)
# 
#             elif result.tool_calls[0]['name'] == 'web_search':
#                   val = web_search.invoke(result.tool_calls[0]['args'])
#                   print("AI : ", val)
#       else:
#             print("AI : ", result.content)
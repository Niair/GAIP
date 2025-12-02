# chatbot based on what we have learned till now
# simple chatbot using console input and output

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()


template = load_prompt('2_Prompts/template.json')

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

chat_history = [
      SystemMessage(content = "You are a helpful assistant.")
      ]

while True:
      user_input = input("You : ")
      chat_history.append(HumanMessage(content=user_input))
      if user_input in ["exit", "quit", "bye", "goodbye", "stop"]:
            print("Chatbot : Goodbye!")
            break

      result = model.invoke(chat_history)
      chat_history.append(AIMessage(content=result.content))
      print("AI : ", result.content)

print(chat_history)

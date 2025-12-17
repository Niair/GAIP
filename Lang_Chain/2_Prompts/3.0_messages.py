from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

messages = [
      SystemMessage(content="You are a helpful assistant."),
      HumanMessage(content="Tell me about langchain."),
] # chat history type of thing

result  = model.invoke(messages)

messages.append(AIMessage(content = result.content))

print(messages)
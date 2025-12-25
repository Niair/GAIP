from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver  # for taking back the conversation of the previous workflow, it save that in ram
from typing import TypedDict, Annotated, Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from dotenv import load_dotenv
load_dotenv()

import time

# state
class ChatState(TypedDict):

      messages : Annotated[list[BaseMessage], add_messages]  # We write base message as it gets inherits by other messages, so by adding this we add the flexibility that any type of message can me inserted into this list[str]
      # we use add_messages instead of operator.add as it is recommended and it is better

# model
# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature = 0.4)
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.4)

# function for node
def main_llm_function(state : ChatState) -> ChatState:

      # take user quey from state
      messages = state['messages']

      # send query to llm
      response = model.invoke(messages)

      # save response in the state
      return {"messages" : [response]}


# checkpointer
checkpointer = MemorySaver()  # for taking back the conversation of the previous workflow, it save that in ram

# create graph
graph = StateGraph(ChatState)

# add nodes
graph.add_node("main_llm_function", main_llm_function)

# add edges
graph.add_edge(START, "main_llm_function")
graph.add_edge("main_llm_function", END)

# compile
chatbot = graph.compile(checkpointer = checkpointer) # *


# # working
# initial_state = {
#       'messages' : [HumanMessage(content = "What is the capital of India")]
# }
# 
# final_state = chatbot.invoke(initial_state)['messages'][-1].content
# 
# print(final_state)

# crazy loop working
thread_id = '1'  # 1 thread is me and if others use this chatbot then there will be thread 2, 3 or more

while True:

      user_message = input("User : ")
      # print(user_message)

      if user_message.strip().lower() in ["exit", "quit", "bye"]:
            break

      config = {'configurable' : {'thread_id' : thread_id}} # *

      response = chatbot.invoke({
            'messages' : [HumanMessage(content = user_message)]
      }, config = config)

      print("AI : ", response['messages'][-1].content)

      time.sleep(2)

print(chatbot.get_state(config = config)) # it will show you the histry of chats
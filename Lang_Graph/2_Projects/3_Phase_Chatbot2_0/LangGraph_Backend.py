from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Annotated

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
load_dotenv()

import time

# state
class ChatState(TypedDict):

      messages : Annotated[list[BaseMessage], add_messages]

# model
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
checkpointer = InMemorySaver()

# create graph
graph = StateGraph(ChatState)

# add nodes
graph.add_node("main_llm_function", main_llm_function)

# add edges
graph.add_edge(START, "main_llm_function")
graph.add_edge("main_llm_function", END)

# compile
chatbot = graph.compile(checkpointer = checkpointer)

# stream = chatbot.stream(
#       {'messages' : [HumanMessage(content = "what is the recipee to make pasta")]},
#       config = {'configurable' : {'thread_id' : 'thread - 1'}},
#       stream_mode = 'messages'
# )
# 
# for message_chunk, metadata in stream:
# 
#       if message_chunk.content:
#             print(message_chunk.content, end = " ", flush = True)
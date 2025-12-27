from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
load_dotenv()
import sqlite3

import time

# -------------------------------------------------------------
# database thread extraction

# to check the current threads
def retrieve_all_threads():
      all_threads = set()
      for checkpoint in checkpointer.list(None):
            all_threads.add(checkpoint.config['configurable']['thread_id'])

      return list(all_threads)


# -------------------------------------------------------------



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


# ------ Building connection object ------
conn = sqlite3.connect(database = "chatbot.db", check_same_thread = False)  
# we are setting this to false because it might give us the errors as we are using the threads, 
# as sqlite only works on one thread and we are working with the multiple threads so that's why we set it as False
# ----------------------------------------

# checkpointer
checkpointer = SqliteSaver(conn = conn)

# create graph
graph = StateGraph(ChatState)

# add nodes
graph.add_node("main_llm_function", main_llm_function)

# add edges
graph.add_edge(START, "main_llm_function")
graph.add_edge("main_llm_function", END)

# compile
chatbot = graph.compile(checkpointer = checkpointer)

# stream = chatbot.invoke(
#       {'messages' : [HumanMessage(content = "what is the recipee to make pasta")]},
#       config = {'configurable' : {'thread_id' : 'thread - 1'}},
#       stream_mode = 'messages'
# )
# 
# 
# for message_chunk, metadata in stream:
#       if message_chunk.content:
#             print(message_chunk.content, end = " ", flush = True)


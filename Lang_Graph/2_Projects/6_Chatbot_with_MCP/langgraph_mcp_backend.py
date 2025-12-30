from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
load_dotenv()
import sqlite3
import yfinance as yf
import requests
import os

# -------------------------------------------------------------
# Helper Function - database thread extraction

# to check the current threads
def retrieve_all_threads():
      all_threads = set()
      for checkpoint in checkpointer.list(None):
            all_threads.add(checkpoint.config['configurable']['thread_id'])

      return list(all_threads)


# -------------------------------------------------------------

# Tools

search_tool = DuckDuckGoSearchRun(region = "us-en")

@tool
def calculator_tool(operation: str, numbers: str) -> float:
    """
    Perform basic math operations on any number of values.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        numbers: Comma-separated numbers like "5,10,15"
    
    Returns:
        The result of the calculation
    
    Example:
        calculator("add", "5,10,15") → 30.0
        calculator("multiply", "2,3,4") → 24.0
    """
    
    # Convert string of numbers to list of floats
    num_list = [float(x.strip()) for x in numbers.split(",")]
    
    # Perform the operation
    if operation.lower() == "add":
        result = sum(num_list)
    
    elif operation.lower() == "subtract":
        result = num_list[0]
        for num in num_list[1:]:
            result -= num
    
    elif operation.lower() == "multiply":
        result = 1
        for num in num_list:
            result *= num
    
    elif operation.lower() == "divide":
        result = num_list[0]
        for num in num_list[1:]:
            if num == 0:
                return "Error: Cannot divide by zero"
            result /= num
    
    else:
        return f"Error: Unknown operation '{operation}'. Use add, subtract, multiply, or divide."
    
    return result


@tool
def get_stock_price(ticker: str) -> str:
    """
    Get current stock price for ANY company worldwide.
    
    Args:
        ticker: Stock ticker symbol with optional exchange suffix
            Examples:
            - US stocks: AAPL, GOOGL, TSLA
            - Indian stocks: TCS.NS, INFY.NS, RELIANCE.NS
            - UK stocks: TSCO.L, BP.L
            - Japanese stocks: 7203.T (Toyota)
            - German stocks: VOW3.DE (Volkswagen)
    
    Returns:
        Current stock price with company name
    """
    
    try:
        stock = yf.Ticker(ticker.upper())
        
        # Get latest price
        hist = stock.history(period="1d")
        
        if hist.empty:
            return f"Could not find stock data for '{ticker}'. Try adding exchange suffix (e.g., .NS for India, .L for London)"
        
        price = hist['Close'].iloc[-1]
        
        # Get company info
        info = stock.info
        company_name = info.get('longName', ticker)
        currency = info.get('currency', 'USD')
        
        return f"{company_name} ({ticker.upper()}): {price:.2f} {currency}"
        
    except Exception as e:
        return f"Error: {str(e)}. Try format: TICKER.EXCHANGE (e.g., TCS.NS, AAPL)"


# -------------------------------------------------------------


# state
class ChatState(TypedDict):

      messages : Annotated[list[BaseMessage], add_messages]


tools = [search_tool, get_stock_price, calculator_tool]

# model
model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4) # llama-3.1-8b-instant

# llm with tools
llm_with_tools = model.bind_tools(tools)


# function for node
def main_llm_function(state : ChatState) -> ChatState:

      # take user quey from state
      messages = state['messages']

      # send query to llm
      response = llm_with_tools.invoke(messages)

      # save response in the state
      return {"messages" : [response]}

tool_node = ToolNode(tools)

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
graph.add_node("tools", tool_node)

# add edges
graph.add_edge(START, "main_llm_function")
graph.add_conditional_edges("main_llm_function", tools_condition)
graph.add_edge("tools", "main_llm_function")

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
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from typing import TypedDict, Annotated

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()

# import sqlite3

import asyncio
import aiosqlite
from langchain_mcp_adapters.client import MultiServerMCPClient


# ============================================================
# GLOBAL STATE - Moved outside to be accessible everywhere
# ============================================================

# conn = sqlite3.connect(database = "chatbot.db", check_same_thread = False)
# checkpointer
# checkpointer = AsyncSqliteSaver(conn = conn)

checkpointer = None


# ============================================================
# TOOLS & SERVER
# ============================================================

search_tool = DuckDuckGoSearchRun(region = "us-en")

SERVERS = {
      "math" : {
            "transport" : "stdio",
            "command" : "E:\\VS_Code\\Scripts\\uv.exe",
            "args" : [
                  "run",
                  "fastmcp",
                  "run",
                  "E:\\_Projects\\GAIP\\MCP\\chat_bot_with_mcp\\local_mcp.py"
            ],
            "env": {},
            "cwd": "E:\\_Projects\\GAIP\\MCP\\chat_bot_with_mcp"
      },

      "expense" : {
           "transport" : "streamable_http", # if not works write 'sse'
           "url" : "https://nihal-finance-server.fastmcp.app/mcp"
      }
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

# to check the current threads
def retrieve_all_threads():
      all_threads = set()
      for checkpoint in checkpointer.list(None):
            all_threads.add(checkpoint.config['configurable']['thread_id'])

      return list(all_threads)


# ============================================================
# STATE & GRAPH
# ============================================================

# state
class ChatState(TypedDict):

      messages : Annotated[list[BaseMessage], add_messages]
    

async def build_graph(client, checkpointer_param):

    mcp_tools = await client.get_tools()

    # tools = [search_tool, get_stock_price, calculator_tool]
    all_tools = [search_tool] + list(mcp_tools)

    for tool in all_tools:
        print(f"  - {tool.name}: {tool.description}")
        

    # model
    model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4) # llama-3.1-8b-instant

    # llm with tools
    llm_with_tools = model.bind_tools(all_tools)


    # function for node
    async def main_llm_function(state : ChatState) -> ChatState:

        # take user quey from state
        messages = state['messages']

        # send query to llm
        response = await llm_with_tools.ainvoke(messages)

        # save response in the state
        return {"messages" : [response]}

    tool_node = ToolNode(all_tools)

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
    chatbot = graph.compile(checkpointer = checkpointer_param)

    return chatbot


# ============================================================
# MAIN EXECUTION
# ============================================================

async def main():
    
    """
    global checkpointer

    client = MultiServerMCPClient(SERVERS)
    
    async with aiosqlite.connect("chatbot.db") as conn:

        checkpointer = AsyncSqliteSaver(conn = conn)

        chatbot = await build_graph(client)
        
        async for message_chunk, metadata in chatbot.astream(
            {'messages' : [HumanMessage(content = "what is the recipee to make pasta, add all the ingreients in weight and give me the sum, also add it as a shoping expense")]},
            config = {'configurable' : {'thread_id' : 'thread - 1'}},
            stream_mode = 'messages'
        ):
            
            if message_chunk.content:
                    print(message_chunk.content, end = " ", flush = True)
    
    """


if __name__ == "__main__":
     asyncio.run(main())
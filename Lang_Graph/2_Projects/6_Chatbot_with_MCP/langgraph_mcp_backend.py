from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from typing import TypedDict, Annotated

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()

import asyncio
import aiosqlite
import threading
from langchain_mcp_adapters.client import MultiServerMCPClient


# ============================================================
# BACKGROUND EVENT LOOP
# ============================================================

# Create a dedicated event loop in a background thread
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    """Submit a coroutine to the background event loop."""
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    """Run a coroutine on the background loop and wait for result."""
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop without waiting."""
    return _submit_async(coro)


# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "chatbot.db"

SERVERS = {
    "math": {
        "transport": "stdio",
        "command": "E:\\VS_Code\\Scripts\\uv.exe",
        "args": [
            "run",
            "fastmcp",
            "run",
            "E:\\_Projects\\GAIP\\MCP\\chat_bot_with_mcp\\local_mcp.py"
        ],
        "env": {},
        "cwd": "E:\\_Projects\\GAIP\\MCP\\chat_bot_with_mcp"
    },
    "expense": {
        "transport": "streamable_http",
        "url": "https://nihal-finance-server.fastmcp.app/mcp"
    }
}


# ============================================================
# GLOBAL STATE
# ============================================================

checkpointer = None
chatbot = None
model = None
mcp_client = None
_initialized = False


# ============================================================
# TOOLS
# ============================================================

search_tool = DuckDuckGoSearchRun(region="us-en")


# ============================================================
# STATE DEFINITION
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# GRAPH BUILDER
# ============================================================

async def build_graph(client, checkpointer_param):
    """Build the LangGraph chatbot with MCP tools"""
    global model
    
    mcp_tools = await client.get_tools()
    all_tools = [search_tool] + list(mcp_tools)

    print("Available tools:")
    for tool in all_tools:
        print(f"  - {tool.name}: {tool.description}")

    model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)
    llm_with_tools = model.bind_tools(all_tools)

    async def main_llm_function(state: ChatState) -> ChatState:
        messages = state['messages']
        
        # Filter and validate messages before sending to LLM
        validated_messages = []
        for msg in messages:
            # Skip tool messages with empty or invalid content
            if hasattr(msg, 'type') and msg.type == 'tool':
                if not msg.content or (isinstance(msg.content, list) and len(msg.content) == 0):
                    print(f"Skipping empty tool message: {msg}")
                    continue
                # Ensure content is a string
                if isinstance(msg.content, list):
                    msg.content = str(msg.content)
            validated_messages.append(msg)
        
        response = await llm_with_tools.ainvoke(validated_messages)
        return {"messages": [response]}

    # Create tool node with wrapper for error handling
    base_tool_node = ToolNode(all_tools)
    
    async def safe_tool_node(state: ChatState) -> ChatState:
        """Wrapper around tool node to ensure all tool messages have content"""
        try:
            result = await base_tool_node.ainvoke(state)
            
            # Validate and fix tool message content
            if 'messages' in result:
                fixed_messages = []
                for msg in result['messages']:
                    if hasattr(msg, 'type') and msg.type == 'tool':
                        # Ensure tool message has content
                        if not msg.content or (isinstance(msg.content, list) and len(msg.content) == 0):
                            msg.content = "Tool executed successfully with no output"
                        # Ensure content is a string
                        elif isinstance(msg.content, list):
                            msg.content = str(msg.content)
                    fixed_messages.append(msg)
                result['messages'] = fixed_messages
            
            return result
        except Exception as e:
            print(f"Error in tool execution: {e}")
            # Return error message as tool result
            from langchain_core.messages import ToolMessage
            return {"messages": [ToolMessage(
                content=f"Tool execution failed: {str(e)}",
                tool_call_id=state['messages'][-1].tool_calls[0]['id'] if state['messages'][-1].tool_calls else "error"
            )]}

    graph = StateGraph(ChatState)
    graph.add_node("main_llm_function", main_llm_function)
    graph.add_node("tools", safe_tool_node)
    
    graph.add_edge(START, "main_llm_function")
    graph.add_conditional_edges("main_llm_function", tools_condition)
    graph.add_edge("tools", "main_llm_function")

    compiled_graph = graph.compile(checkpointer=checkpointer_param)
    return compiled_graph


# ============================================================
# INITIALIZATION
# ============================================================

async def _init_async():
    """Async initialization function"""
    global checkpointer, chatbot, mcp_client
    
    print("Initializing MCP client...")
    mcp_client = MultiServerMCPClient(SERVERS)
    print("✓ MCP client created")
    
    print("Initializing database...")
    conn = await aiosqlite.connect(DB_PATH)
    checkpointer = AsyncSqliteSaver(conn=conn)
    print("✓ Checkpointer initialized")
    
    print("Building chatbot graph...")
    chatbot_instance = await build_graph(mcp_client, checkpointer)
    print("✓ Chatbot graph built successfully")
    
    return chatbot_instance


def initialize_sync():
    """Synchronously initialize all async components using background thread"""
    global chatbot, _initialized
    
    if _initialized:
        return chatbot
    
    try:
        chatbot = run_async(_init_async())
        _initialized = True
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return chatbot


def get_chatbot():
    """Get or initialize chatbot instance"""
    global chatbot
    if chatbot is None:
        initialize_sync()
    return chatbot


# ============================================================
# HELPER FUNCTIONS
# ============================================================

async def _alist_threads():
    """Async function to retrieve all thread IDs from checkpointer"""
    if checkpointer is None:
        return []
    
    all_threads = set()
    try:
        async for checkpoint in checkpointer.alist(None):
            thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
            if thread_id:
                all_threads.add(thread_id)
    except Exception as e:
        print(f"Error retrieving threads: {e}")
    
    return list(all_threads)


def retrieve_all_threads():
    """Retrieve all thread IDs from checkpointer (sync wrapper)"""
    return run_async(_alist_threads())


# ============================================================
# MAIN EXECUTION (for testing)
# ============================================================

async def main():
    """Test function"""
    client = MultiServerMCPClient(SERVERS)
    
    async with aiosqlite.connect(DB_PATH) as conn:
        global checkpointer, chatbot, model
        checkpointer = AsyncSqliteSaver(conn=conn)
        chatbot = await build_graph(client, checkpointer)
        
        print("\n" + "="*50)
        print("Testing chatbot...")
        print("="*50 + "\n")
        
        async for message_chunk, metadata in chatbot.astream(
            {'messages': [HumanMessage(content="Hello! Can you help me calculate 15 + 27?")]},
            config={'configurable': {'thread_id': 'test-thread'}},
            stream_mode='messages'
        ):
            if hasattr(message_chunk, 'content') and message_chunk.content:
                print(message_chunk.content, end="", flush=True)
        
        print("\n" + "="*50)
        print("Test completed!")
        print("="*50)


if __name__ == "__main__":
    asyncio.run(main())
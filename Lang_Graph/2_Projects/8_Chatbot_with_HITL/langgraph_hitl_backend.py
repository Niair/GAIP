import asyncio
import os
from typing import AsyncIterator, Dict, Any, List
import aiosqlite
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import TypedDict, Annotated
import logging

# Import configuration
from config import (
    LLM_MODEL, LLM_TEMPERATURE, SERVERS, RAG_CONFIG,
    DB_PATH, FEATURES, HITL_CONFIG, DEBUG, LOG_LEVEL
)

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# GLOBAL STATE
# ============================================================

_initialized = False
_loop = None
_loop_thread = None

# Chatbot components
chatbot = None
checkpointer = None
mcp_client = None

# RAG components
retriever = None
vector_store = None
current_document_info = None

# LLM
model = None

# ============================================================
# BACKGROUND ASYNC LOOP
# ============================================================

def _start_async_loop():
    """Start background event loop"""
    global _loop, _loop_thread
    
    if _loop is not None:
        return
    
    import threading
    _loop = asyncio.new_event_loop()
    _loop_thread = threading.Thread(target=_loop.run_forever, daemon=True)
    _loop_thread.start()
    logger.info("✓ Background event loop started")

def run_async(coro):
    """Run coroutine in background loop and wait for result"""
    _start_async_loop()
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result(timeout=300)

# ============================================================
# RAG TOOLS
# ============================================================

@tool
def rag_tool(query: str) -> dict:
    """
    Retrieve relevant information from uploaded PDF document.
    Use this for document Q&A and factual questions.
    """
    global retriever, current_document_info
    
    if retriever is None or current_document_info is None:
        return {
            'query': query,
            'context': [],
            'metadata': [],
            'message': 'No document loaded. Please upload a PDF first.',
            'has_document': False
        }
    
    try:
        result = retriever.invoke(query)
        context = [doc.page_content for doc in result]
        metadata = [doc.metadata for doc in result]
        
        return {
            'query': query,
            'context': context,
            'metadata': metadata,
            'document': current_document_info['filename'],
            'has_document': True
        }
    except Exception as e:
        logger.error(f"RAG Tool Error: {e}")
        return {
            'query': query,
            'context': [],
            'metadata': [],
            'error': f"Retrieval error: {str(e)}",
            'has_document': True
        }

def process_document(pdf_path: str) -> Dict[str, Any]:
    """Process PDF and create RAG system"""
    global retriever, vector_store, current_document_info
    
    try:
        logger.info(f"📄 Processing: {pdf_path}")
        
        # Load
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        logger.info(f"✅ Loaded {len(docs)} pages")
        
        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_CONFIG['chunk_size'],
            chunk_overlap=RAG_CONFIG['chunk_overlap']
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"✅ Created {len(chunks)} chunks")
        
        # Embed
        embeddings = GoogleGenerativeAIEmbeddings(
            model=RAG_CONFIG['embedding_model']
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        logger.info("✅ Vector store created")
        
        # Retrieve
        retriever = vector_store.as_retriever(
            search_type=RAG_CONFIG['search_type'],
            search_kwargs={'k': RAG_CONFIG['retriever_k']}
        )
        
        current_document_info = {
            'filename': os.path.basename(pdf_path),
            'pages': len(docs),
            'chunks': len(chunks),
            'path': pdf_path
        }
        
        logger.info("✅ RAG system ready!")
        return {'success': True, 'info': current_document_info}
    
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        return {'success': False, 'error': str(e)}

def remove_document():
    """Remove loaded document"""
    global retriever, vector_store, current_document_info
    
    retriever = None
    vector_store = None
    current_document_info = None
    logger.info("📭 Document removed")
    return {'success': True}

def get_rag_status() -> Dict[str, Any]:
    """Get RAG system status"""
    return {
        'has_document': current_document_info is not None,
        'document_info': current_document_info,
        'rag_active': retriever is not None
    }

# ============================================================
# GRAPH SETUP
# ============================================================

class ChatState(TypedDict):
    """Chat state definition"""
    messages: Annotated[list[BaseMessage], add_messages]

async def build_graph(mcp_client_param, checkpointer_param):
    """Build LangGraph chatbot"""
    global model
    
    # Get tools
    mcp_tools = await mcp_client_param.get_tools()
    
    # Build tool list
    tools = [rag_tool]
    
    if FEATURES.get('ENABLE_WEB_SEARCH'):
        search_tool = DuckDuckGoSearchRun(region="us-en")
        tools.append(search_tool)
    
    tools.extend(list(mcp_tools))
    
    logger.info("Available tools:")
    for t in tools:
        desc = t.description[:50] if t.description else "No description"
        logger.info(f"  - {t.name}: {desc}...")
    
    # Setup LLM
    model = ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    llm_with_tools = model.bind_tools(tools)
    
    # LLM Function
    async def main_llm_function(state: ChatState) -> ChatState:
        messages = state['messages']
        
        # Clean messages
        validated = []
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'tool':
                if not msg.content or (isinstance(msg.content, list) and len(msg.content) == 0):
                    logger.debug("Skipping empty tool message")
                    continue
                if isinstance(msg.content, list):
                    msg.content = str(msg.content)
            validated.append(msg)
        
        response = await llm_with_tools.ainvoke(validated)
        return {"messages": [response]}
    
    # Tool execution wrapper
    base_tool_node = ToolNode(tools)
    
    async def safe_tool_node(state: ChatState) -> ChatState:
        try:
            result = await base_tool_node.ainvoke(state)
            
            if 'messages' in result:
                fixed = []
                for msg in result['messages']:
                    if hasattr(msg, 'type') and msg.type == 'tool':
                        if not msg.content or (isinstance(msg.content, list) and len(msg.content) == 0):
                            msg.content = "Tool executed successfully"
                        elif isinstance(msg.content, list):
                            msg.content = str(msg.content)
                    fixed.append(msg)
                result['messages'] = fixed
            
            return result
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {"messages": [ToolMessage(
                content=f"Tool error: {str(e)}",
                tool_call_id="error"
            )]}
    
    # Build graph
    graph = StateGraph(ChatState)
    graph.add_node("llm", main_llm_function)
    graph.add_node("tools", safe_tool_node)
    
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", tools_condition)
    graph.add_edge("tools", "llm")
    
    # Compile with HITL
    compiled = graph.compile(
        checkpointer=checkpointer_param,
        interrupt_before=HITL_CONFIG['interrupt_before'] if FEATURES['ENABLE_HITL'] else []
    )
    
    logger.info("✓ Graph compiled successfully")
    return compiled

# ============================================================
# INITIALIZATION
# ============================================================

async def _initialize_async():
    """Async initialization"""
    global chatbot, checkpointer, mcp_client
    
    try:
        logger.info("Initializing MCP client...")
        mcp_client = MultiServerMCPClient(SERVERS)
        logger.info("✓ MCP client created")
        
        logger.info("Initializing database...")
        conn = await aiosqlite.connect(DB_PATH)
        checkpointer = AsyncSqliteSaver(conn=conn)
        logger.info("✓ Checkpointer initialized")
        
        logger.info("Building chatbot graph...")
        chatbot = await build_graph(mcp_client, checkpointer)
        logger.info("✓ Chatbot initialized")
        
        return True
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise

async def initialize():
    """Initialize chatbot (async)"""
    global _initialized
    
    if _initialized:
        return
    
    await _initialize_async()
    _initialized = True
    logger.info("🎉 System ready!")

# ============================================================
# CHAT STREAMING
# ============================================================

async def chat_stream(user_input: str, thread_id: str) -> AsyncIterator:
    """Stream chat response"""
    global chatbot
    
    if chatbot is None:
        logger.error("Chatbot not initialized")
        raise RuntimeError("Chatbot not initialized")
    
    try:
        config = {'configurable': {'thread_id': thread_id}}
        
        async for event in chatbot.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode='messages'
        ):
            yield event
    
    except Exception as e:
        logger.error(f"Chat stream error: {e}")
        raise

# ============================================================
# HITL FUNCTIONS
# ============================================================

async def get_pending_tools(thread_id: str) -> Dict[str, Any]:
    """Get pending tool calls (HITL)"""
    global chatbot
    
    if chatbot is None:
        return {'has_pending': False}
    
    try:
        config = {'configurable': {'thread_id': thread_id}}
        state = await chatbot.aget_state(config=config)
        
        if not state.next:
            return {'has_pending': False}
        
        messages = state.values.get('messages', [])
        if not messages:
            return {'has_pending': False}
        
        last_msg = messages[-1]
        
        if not hasattr(last_msg, 'tool_calls') or not last_msg.tool_calls:
            return {'has_pending': False}
        
        pending = []
        for call in last_msg.tool_calls:
            pending.append({
                'name': call['name'],
                'args': call['args'],
                'id': call['id']
            })
        
        logger.info(f"Found {len(pending)} pending tools")
        return {'has_pending': True, 'tools': pending}
    
    except Exception as e:
        logger.error(f"Error checking pending tools: {e}")
        return {'has_pending': False}

async def approve_tool_execution(thread_id: str) -> AsyncIterator:
    """Approve and execute pending tools - FIXED async generator"""
    global chatbot
    
    if chatbot is None:
        raise RuntimeError("Chatbot not initialized")
    
    try:
        config = {'configurable': {'thread_id': thread_id}}
        
        async for event in chatbot.astream(
            None,
            config=config,
            stream_mode='messages'
        ):
            yield event
            
    except Exception as e:
        logger.error(f"Tool approval error: {e}")
        raise

async def reject_tool_execution(thread_id: str, reason: str) -> Dict[str, Any]:
    """Reject pending tools"""
    global chatbot
    
    if chatbot is None:
        return {'success': False}
    
    try:
        config = {'configurable': {'thread_id': thread_id}}
        
        rejection_msg = AIMessage(content=f"I wanted to use a tool, but {reason}.")
        
        await chatbot.aupdate_state(
            config=config,
            values={'messages': [rejection_msg]}
        )
        
        logger.info("Tool rejected by user")
        return {
            'success': True,
            'message': rejection_msg.content
        }
    
    except Exception as e:
        logger.error(f"Rejection error: {e}")
        return {'success': False, 'error': str(e)}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_all_threads() -> List[str]:
    """Get all conversation threads"""
    return run_async(_list_all_threads())

async def _list_all_threads():
    """List threads (async)"""
    global checkpointer
    
    if checkpointer is None:
        return []
    
    threads = set()
    try:
        async for checkpoint in checkpointer.alist(None):
            thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
            if thread_id:
                threads.add(thread_id)
    except Exception as e:
        logger.error(f"Error retrieving threads: {e}")
    
    return list(threads)

def get_conversation_history(thread_id: str) -> List[Dict]:
    """Get conversation history"""
    return run_async(_get_history(thread_id))

async def _get_history(thread_id: str):
    """Get history (async)"""
    global chatbot
    
    if chatbot is None:
        return []
    
    try:
        state = await chatbot.aget_state(
            config={'configurable': {'thread_id': thread_id}}
        )
        
        history = []
        for msg in state.values.get('messages', []):
            if isinstance(msg, HumanMessage):
                history.append({'role': 'user', 'content': msg.content})
            elif isinstance(msg, AIMessage):
                history.append({'role': 'assistant', 'content': msg.content})
        
        return history
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return []

# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    'initialize',
    'chat_stream',  # kept for compatibility, but we won't use it anymore
    'get_pending_tools',
    'approve_tool_execution',
    'reject_tool_execution',
    'process_document',
    'remove_document',
    'get_rag_status',
    'get_all_threads',
    'get_conversation_history',
    'chatbot'  # ← NEW: allow frontend to access the graph directly
]
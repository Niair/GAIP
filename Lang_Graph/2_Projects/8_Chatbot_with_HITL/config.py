import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# DATABASE
# ============================================================
DB_PATH = "chatbot.db"

# ============================================================
# MCP SERVERS CONFIGURATION
# ============================================================
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
# LLM CONFIGURATION
# ============================================================
LLM_MODEL = "llama3-70b-8192"  # ← FIXED: valid Groq model (was invalid)
LLM_TEMPERATURE = 0.7           # slightly higher for better responses

# ============================================================
# RAG CONFIGURATION
# ============================================================
RAG_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "models/text-embedding-004",
    "retriever_k": 4,
    "search_type": "similarity",
}

# ============================================================
# SEARCH CONFIGURATION
# ============================================================
SEARCH_CONFIG = {
    "region": "us-en"
}

# ============================================================
# CHAINLIT CONFIGURATION
# ============================================================
CHAINLIT_CONFIG = {
    "title": "🤖 AI Chatbot with MCP & Human-in-the-Loop",
    "description": "Powered by LangGraph, MCP, RAG, and Chainlit",
    "theme": "dark",  # matches your screenshot
    "show_chat_history": True,
    "max_tokens": 2000
}

# ============================================================
# DEBUG
# ============================================================
DEBUG = True

# ============================================================
# FEATURE TOGGLES
# ============================================================

FEATURES = {
    "ENABLE_RAG": True,
    "ENABLE_WEB_SEARCH": True,
    "ENABLE_MCP_TOOLS": True,
    "ENABLE_HITL": True,
}

# ============================================================
# HITL CONFIGURATION
# ============================================================

HITL_CONFIG = {
    "interrupt_before": ["tools"],
}

# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = "INFO"
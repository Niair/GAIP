import streamlit as st
from langchain_community.tools import tool, DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import requests
import os
import json

load_dotenv()

# Robust API key handling
EXCHANGE_RATE_API_KEY = os.getenv('EXCHANGE_RATE_API_KEY')
try:
    secret_key = st.secrets.get("EXCHANGE_RATE_API_KEY")
    if secret_key:
        EXCHANGE_RATE_API_KEY = secret_key
except FileNotFoundError:
    pass

@tool
def convert(base_currency_value: int, conversion_rate: float) -> float:
    """Given a currency conversion rate this function calculate the target currency value from a given base value"""
    return base_currency_value * conversion_rate

@tool
def get_currency_conversion_factor(base_currency: str, target_currency: str) -> float:
    """This function fetches the current currency conversion factor between a given base_currency and a target_currency from an api"""
    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_API_KEY}/pair/{base_currency}/{target_currency}"
    response = requests.get(url)
    return response.json()

# Initialize model
model = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.7)
model_with_tools = model.bind_tools([convert, get_currency_conversion_factor])

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful AI assistant that is experienced in currency conversion. Always provide clear, formatted responses with the conversion results.")
    ]

# Page config
st.set_page_config(
    page_title="AI Currency Converter", 
    page_icon="💱", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, formal styling
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Title styling */
    h1 {
        color: #1a1a1a !important;
        text-align: center;
        font-size: 2.5em !important;
        margin-bottom: 0.3em !important;
        font-weight: 600 !important;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #4a5568;
        font-size: 1.1em;
        margin-bottom: 2em;
        font-weight: 400;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: white !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    
    /* User message */
    [data-testid="stChatMessageContent"] {
        color: #1a1a1a !important;
        font-size: 1em !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] div {
        color: #4a5568 !important;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    
    .stButton button:hover {
        background-color: #1d4ed8;
    }
    
    /* Chat input */
    .stChatInput {
        border-color: #cbd5e0 !important;
    }
    
    /* Divider */
    hr {
        margin: 1.5rem 0;
        border-color: #e2e8f0;
    }
    
    /* Welcome box */
    .welcome-box {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .welcome-box h4 {
        color: #1a1a1a;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .welcome-box p {
        color: #4a5568;
        line-height: 1.6;
    }
    
    /* Example list */
    .example-list {
        background-color: #f8f9fa;
        border-left: 3px solid #2563eb;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .example-list li {
        color: #4a5568;
        margin: 0.5rem 0;
    }
    
    /* Feature badge */
    .feature-badge {
        display: inline-block;
        background-color: #eff6ff;
        color: #2563eb;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #2563eb !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 💱 AI Currency Converter")
st.markdown('<p class="subtitle">Real-time currency conversion powered by AI</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = [
            SystemMessage(content="You are a helpful AI assistant that is experienced in currency conversion. Always provide clear, formatted responses with the conversion results.")
        ]
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    st.markdown("""
    - Convert 100 USD to EUR
    - How much is 50 GBP in JPY?
    - 1000 INR to AUD
    - What's 250 CAD in CHF?
    - Convert 75 AED to USD
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Features")
    st.markdown("""
    <div>
        <span class="feature-badge">Real-time rates</span>
        <span class="feature-badge">150+ currencies</span>
        <span class="feature-badge">Natural language</span>
        <span class="feature-badge">AI-powered</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This tool uses real-time exchange rate data and AI to help you convert between different currencies naturally.
    """)

# Welcome message for first-time users
if len(st.session_state.chat_history) == 1:
    st.markdown("""
    <div class="welcome-box">
        <h4>👋 Welcome to AI Currency Converter</h4>
        <p>I can help you convert between different currencies using real-time exchange rates.</p>
        <p><strong>Try asking:</strong></p>
        <ul style="margin-top: 0.5rem; margin-bottom: 0;">
            <li>"Convert 100 USD to EUR"</li>
            <li>"How much is 50 GBP in Japanese Yen?"</li>
            <li>"What's 1000 INR worth in Australian Dollars?"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.chat_history[1:]:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            # Display the actual response only
            if msg.content:
                st.markdown(msg.content)
    # Skip ToolMessage - don't display to users

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    if user_input.lower() == "exit":
        st.info("Conversation ended. Refresh the page to start a new one.")
        st.stop()
    
    # Add user message
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    
    # Generate and display assistant response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Processing..."):
            while True:
                result = model_with_tools.invoke(st.session_state.chat_history)
                
                # If no tool calls, display final response and break
                if not result.tool_calls:
                    st.session_state.chat_history.append(result)
                    st.markdown(result.content)
                    break
                
                # Add AI message with tool calls to history (but don't display badges)
                st.session_state.chat_history.append(result)
                
                # Execute the tool silently
                if result.tool_calls[0]['name'] == 'convert':
                    tool_output = convert.invoke(result.tool_calls[0])
                elif result.tool_calls[0]['name'] == 'get_currency_conversion_factor':
                    tool_output = get_currency_conversion_factor.invoke(result.tool_calls[0])
                
                # Add tool result to history
                tool_message = ToolMessage(
                    content=str(tool_output),
                    tool_call_id=result.tool_calls[0]['id']
                )
                st.session_state.chat_history.append(tool_message)
    
    st.rerun()
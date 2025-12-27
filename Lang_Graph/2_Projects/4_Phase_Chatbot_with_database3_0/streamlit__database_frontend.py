import streamlit as st
from LangGraph_Database_Backend import chatbot, model, retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid

# ******************************************************  Utility Functions  ******************************************************

# function to rename the each chat name
@st.dialog("Rename Chat")
def rename_dialog(thread_id):
    current_name = st.session_state['thread_titles'].get(thread_id, "New Chat")
    new_name = st.text_input("Enter new name:", value=current_name)
    
    if st.button("Save"):
        st.session_state['thread_titles'][thread_id] = new_name
        st.rerun()

def generate_thread_id():

      thread_id = uuid.uuid4()
      return thread_id

def reset_chat():

      thread_id = generate_thread_id()
      st.session_state['thread_id'] = thread_id
      # add_thread(st.session_state['thread_id'])
      st.session_state['message_history'] = []

def add_thread(thread_id):

      if thread_id not in st.session_state['chat_threads']:
            st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
      
      return chatbot.get_state(config = {'configurable' : {'thread_id' : thread_id}}).values['messages']

def model_title_generation(user_input, ai_message):
      prompt = f"Generate ONLY a short title (max 5 words, no quotes or extra text) based on this conversation:\nUser: {user_input}\nAssistant: {ai_message}"
      title_response = model.invoke(prompt)
      title = title_response.content.strip()
      return title


# ******************************************************  Session Setup  ******************************************************

if 'message_history' not in st.session_state:
      st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
      st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
      st.session_state['chat_threads'] = retrieve_all_threads()

if 'thread_titles' not in st.session_state:
      st.session_state['thread_titles'] = {}

# add_thread(st.session_state['thread_id'])


# ******************************************************  Sidebar UI  ******************************************************

st.sidebar.title("Chat")

if st.sidebar.button("New Chat"):
      reset_chat()

st.sidebar.header("Your Chats")

for thread_id in st.session_state['chat_threads'][::-1]:
    display_name = st.session_state['thread_titles'].get(thread_id, "New Chat")
    
    # Create two columns: one for chat button, one for edit button
    col1, col2 = st.sidebar.columns([5, 1])
    
    with col1:
        if st.button(display_name, key=str(thread_id)):
            st.session_state['thread_id'] = thread_id
            messages = load_conversation(thread_id)
            
            temp_message = []
            for message in messages:
                if isinstance(message, HumanMessage):
                    role = 'user'
                else:
                    role = 'ai'
                temp_message.append({'role' : role, 'content' : message.content})
            
            st.session_state['message_history'] = temp_message
    
    with col2:
        if st.button("✏️", key=f"edit_{thread_id}"):
            rename_dialog(thread_id)


# ******************************************************  Main UI  ******************************************************

for message in st.session_state['message_history']:
      with st.chat_message(message['role']):
            st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

      add_thread(st.session_state['thread_id'])

      # add the message to history
      st.session_state['message_history'].append({"role" : "user", "content" : user_input})
      with st.chat_message("user"):
            st.text(user_input)
      
      # ----- this part ----

      CONFIG = {'configurable' : {'thread_id' : st.session_state['thread_id']}}

      with st.chat_message("ai"):
            ai_message = st.write_stream(
                  message_chunk.content for message_chunk, metadata in chatbot.stream(
                        {'messages' : [HumanMessage(content = user_input)]}, 
                        config = CONFIG, 
                        stream_mode = 'messages'
                  )
            )
      
      st.session_state['message_history'].append({"role" : "ai", "content" : ai_message})

      if len(st.session_state['message_history']) == 2:
            # generate title
            title = model_title_generation(user_input, ai_message)
            st.session_state['thread_titles'][st.session_state['thread_id']] = title


      # --------------------
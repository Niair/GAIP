import chainlit as cl
from LangGraph_Backend import chatbot
from langchain_core.messages import HumanMessage


CONFIG = {'configurable' : {'thread_id' : 'thread - 1'}}



@cl.on_message
async def on_chat_start():

      await cl.Message(
            content = "Hi there! I am your personal AI"
      ).send()

@cl.on_message
async def on_message(message : cl.Message):

      user_input = message.content
      
      # ----- this part ----
      response = chatbot.invoke({'messages' : [HumanMessage(content = user_input)]}, config = CONFIG)
      ai_message = response['messages'][-1].content
      # --------------------

      await cl.Message(content = ai_message).send()
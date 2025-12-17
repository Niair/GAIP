from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

st.header("Text Symmirization model using Google Gemini-2.5")

user_input = st.text_input("Ask anything")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

if st.button("Summarize"):
      st.text("Generating response...")
      result = model.invoke(user_input)
      st.write(result.content)
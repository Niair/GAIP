from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model_name="gpt-4o")  
response = llm.invoke("What is the capital of India?")

print(response)
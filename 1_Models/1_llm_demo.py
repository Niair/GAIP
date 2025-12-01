from langchain_openai import OpenAI
from devenv import load_env
load_env()

llm = OpenAI(model_name="gpt-4")
response = llm("What is the capital of India?")
print(response)

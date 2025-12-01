# This one works but not working for other models from Hugging Face using api keys as their endpoints responce is canseled, so only option
# is to use it locally, or using the paid plan.

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-0.6B", # Or "HuggingFaceH4/zephyr-7b"
    task="text-generation",
    temperature=0.7,
    max_new_tokens=200
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of India?")
print(result.content)
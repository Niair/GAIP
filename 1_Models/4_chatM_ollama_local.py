from langchain_ollama import OllamaLLM

# Replace 'mistral' with any model name listed in `ollama list`
llm = OllamaLLM(
    model="llama3.2:latest",  # ollama list
    base_url="http://localhost:11434"
)

result = llm.invoke("What is the capital of India?")
print(result)
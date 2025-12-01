from dotenv import load_dotenv

load_dotenv()


class ChatModel:
    def __init__(self):
        pass
    
    def openai(self, messages):
        from langchain_openai import ChatOpenAI

        chat_model = ChatOpenAI(model="gpt-4.1")
        return chat_model.invoke(messages)
    
    def anthropic(self, messages):
        from langchain_anthropic import ChatAnthropic

        chat_model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        return chat_model.invoke(messages)
    
    def google(self, messages):
        from langchain_google_genai import ChatGoogleGenerativeAI

        chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        return chat_model.invoke(messages)
    
    def groq(self, messages):
        from langchain_groq import ChatGroq

        chat_model = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.7) 
        # llama-3.3-70b-versatile, moonshotai/kimi-k2-instruct-0905, meta-llama/llama-4-maverick-17b-128e-instruct, meta-llama/llama-4-scout-17b-16e-instruct
        # groq models: openai/gpt-oss-safeguard-20b, openai/gpt-oss-20b, openai/gpt-oss-120b, 
        # qwen/qwen3-32b
        return chat_model.invoke(messages)
    


if __name__ == "__main__":
    chat_model = ChatModel()
    response = chat_model.groq("Tell me the poem on cricket in 50 words.")

    print(response.content)
    

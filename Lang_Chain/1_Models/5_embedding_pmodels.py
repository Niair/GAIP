from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

class EmbeddingModels:
      def __init__(self):
            pass

      def openai_embeddings_normalquery(self):
            embedding = OpenAIEmbeddings(model = "text-embedding-3-small", dimentions = 32) # larger the dimentions more context can be captured.
            return str(embedding.embed_query("Delhi is the capital of India.")) # we are using the str to print the embedding as a string.
      
      def openai_embeddings_docs(self):
            embedding = OpenAIEmbeddings(model = "text-embedding-3-small", dimentions = 32)
            documents = [
                  "Delhi is the capital of India.",
                  "Mumbai is the financial capital of India.",
                  "Bangalore is the IT hub of India."
            ]
            return str(embedding.embed_documents(documents))
      
if __name__ == "__main__":
      emodel = EmbeddingModels()
      print(emodel.openai_embeddings_normalquery())
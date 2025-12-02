from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "Delhi is the capital of India."

result = embedding.embed_query(text)

print(str(result))

class EmbeddingModels:
      def __init__(self):
            pass

      def hugface_embeddings_normalquery(self):
            embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            text = "Delhi is the capital of India."
            
            result = embedding.embed_query(text)

            return str(result)
      
      def hugface_embeddings_docs(self):
            embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            documents = [
                  "Delhi is the capital of India.",
                  "Mumbai is the financial capital of India.",
                  "Bangalore is the IT hub of India."
            ]
            
            result = embedding.embed_documents(documents)
            
            return str(result)

if __name__ == "__main__":
      emodel = EmbeddingModels()
      print(emodel.hugface_embeddings_docs())
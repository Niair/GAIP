from langchain_community.vectorstores import Chroma, FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

# embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")

# vector store
vector_store = Chroma.from_documents(
      documents = docs,
      embedding = embedding_model
)

vector_store1 = FAISS.from_documents(
      documents = docs,
      embedding = embedding_model
)

# retriever - using vectore
retrievers = vector_store1.as_retriever(
      search_type = 'mmr',
      search_kwargs = {'k':3, "lambda_mult" : 0.4} # lambda_mult - relavence diversity tuker/nob {1 : works normally as no mmr, 0.5/0.4 : works as mmr}
)

# query
query = "what is langchain"

# results
results = retrievers.invoke(query)

for i, doc in enumerate(results):
      print(f"------------------- Retrieved result document {i+1} -------------------")
      print(f"Content : \n\n{doc.page_content}\n")
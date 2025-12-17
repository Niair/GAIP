from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

# sorce document
documents = [
      Document(page_content = "LangChain helps developers build LLM applications easily."),
      Document(page_content = "Chroma is a vector database optimized for LLM-based search."),
      Document(page_content = "Embeddings convert text into high-dimensional vectors."),
      Document(page_content = "OpenAI provides powerful embedding models.")
]

# initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")

# create croma vector store in memory
vector_store = Chroma.from_documents(
      documents = documents,
      embedding = embeddings,
      collection_name = 'my_collection' 
)

# retriever
retriever = vector_store.as_retriever(search_kwargs = {'k':2})

# query
query = "what is Chroma used for"


# results
results = retriever.invoke(query)

# print(results)
# print(len(results))
# print(type(results))

for i, doc in enumerate(results):
      print(f"Retrieved result document {i+1}")
      print(f"Content : \n\n {doc.page_content}")


# Here comes the questions if we can just do it using the vector store that we did in the vector store component code, we can still get the same results
# so why we need this retriever?
# ---->   results = vector_store.similarity_search(query, k = 2)

# Ans --> Yes it's true that we can do it using the searching using vector store but we can only do it using only one algorithm/strategy 
# but if we use the retriever then we can use multiple algorithms/strategy to retrieve the answer, what we use above is just the simple strategy


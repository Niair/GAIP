from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain.retrievers import MultiQueryRetriever
from dotenv import load_dotenv
load_dotenv()
# pip3 install pinecone

# Relevant health & wellness documents
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# model
# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# embeddings
embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")

# vector store
vector_store = Chroma.from_documents(
      documents = all_docs,
      embedding = embeddings
)

# retriever
retriever = vector_store.as_retriever(
      search_type = "similarity",
      search_kwargs = {"k" : 5}
)

multi_query_retriever = MultiQueryRetriever.from_llm(
      retriever = vector_store.as_retriever(search_kwargs = {"k" : 5}),
      llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
)

# query
query = "how to improve energy levels and mintain balance"

# results
similarity_results = retriever.invoke(query)

multiquery_results = multi_query_retriever.invoke(query)

for i, doc in enumerate(similarity_results):
      print(f"------------------- Retrieved (similarity_results) result document {i+1} -------------------")
      print(f"Content : \n\n{doc.page_content}\n")

print("-----------------------------------------------------------------------------------------")

for i, doc in enumerate(multiquery_results):
      print(f"------------------- Retrieved (MultiQueryRetriever) result document {i+1} -------------------")
      print(f"Content : \n\n{doc.page_content}\n")
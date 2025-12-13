from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv
load_dotenv()

# Recreate the document objects from the previous data
docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

# embeddings
embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")

# vector store
vector_store = Chroma.from_documents(
      embedding = embeddings,
      documents = docs
)

# retriever
base_retriever = vector_store.as_retriever(
      search_kwargs = {"k" : 5}
)

# compresser using llm
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

compressor = LLMChainExtractor.from_llm(model)


# contexual compression retriever
compression_retriever = ContextualCompressionRetriever(
      base_retriever = base_retriever,
      base_compressor = compressor
)

# query
query = "what is photosynthesis"

# compression result
compression_results = compression_retriever.invoke(query)


for i, doc in enumerate(compression_results):
      print(f"------------------- Retrieved (similarity_results) result document {i+1} -------------------")
      print(f"Content : \n\n{doc.page_content}\n")
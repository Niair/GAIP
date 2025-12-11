from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# text loader
loader = PyPDFLoader("7_RAG/attention_is_all_you_need.pdf")

docs = loader.load()

# text splitter
splitter = CharacterTextSplitter(
      chunk_size = 500,
      chunk_overlap = 0,
      separator = ''
)

chunks = splitter.split_documents(docs)

print(chunks[0])  # the chunk we get is also an document object
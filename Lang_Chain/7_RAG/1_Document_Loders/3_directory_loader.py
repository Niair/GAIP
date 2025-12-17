from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
      path = "7_RAG/docs",
      glob = "*.pdf",
      loader_cls = PyPDFLoader
)

docs = loader.load()

# print(len(docs))
# print(docs)
# print(type(docs))
print(docs[326].page_content)
print(docs[326].metadata)

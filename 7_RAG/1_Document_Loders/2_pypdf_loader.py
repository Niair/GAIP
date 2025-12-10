from langchain_community.document_loaders import PyPDFLoader
# pip install PyPF

loader = PyPDFLoader("7_RAG//2.0_attention_is_all_you_need.pdf")

docs = loader.load()


# print(docs)
# print(type(docs))
print(docs[0].page_content)
print(docs[0].metadata)

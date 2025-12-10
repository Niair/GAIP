from langchain_community.document_loaders import UnstructuredPDFLoader

# pip install unstructured[pdf]
# pip install unstructured
# pip install pdfminer.six
# pip install unstructured-inference
# pip install pypdf

loader = UnstructuredPDFLoader("7_RAG//2.0_attention_is_all_you_need.pdf")

docs = loader.load()

print(docs[3].page_content)
print(docs[3].metadata)
# print(len(docs))
# print(type(docs))
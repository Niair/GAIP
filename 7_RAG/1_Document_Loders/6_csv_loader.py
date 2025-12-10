from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path = "7_RAG/Social_Network_Ads.csv")

docs = loader.load()

# print(len(docs))
# print(docs)
# print(type(docs))
print(docs[0].page_content)
print(docs[0].metadata)
# every row is the page here or you could say every row is langchain_object in the list
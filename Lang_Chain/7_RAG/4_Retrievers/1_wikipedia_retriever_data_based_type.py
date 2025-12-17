from langchain_community.retrievers import WikipediaRetriever
# pip install wikipedia

# initializing retrievers
retriever = WikipediaRetriever(
      top_k_results = 2,
      lang = 'en'
)


# query
query = "Explain me about the geographical history of India and Pakistan from the perspective of Chinese"

# get relivant wikipedia document
docs = retriever.invoke(query)

# print(len(docs))
# print(type(docs))

for i, doc in enumerate(docs):
      print(f"Retrieved result document {i+1}")
      print(f"Content : \n\n {doc.page_content}")
import os
import numpy as np
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Set HUGGINGFACEHUB_API_TOKEN in your .env or env")

# Use the 'model' param per newer versions
embedder = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-small-en-v1.5",
    huggingfacehub_api_token=HF_TOKEN,
    task="feature-extraction"   # ensure embeddings
)

documents = [
    "Virat Kohli: An Indian cricketer known for his aggressive batting and leadership and he is the last cricket team captain.",
    "MS Dhoni: A former Indian cricket team captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar: Known as the 'God of Cricket' with many batting records.",
    "Rohit Sharma: Known for elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah: Fast bowler known for his unorthodox action and yorkers."
]

query = "Tell me about the cricket team captain."

document_embeddings = np.array(embedder.embed_documents(documents))
query_embeddings = np.array(embedder.embed_query(query))

scores = cosine_similarity([query_embeddings], document_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key = lambda x : x[1])[-1]

print(f"Query: {query}\n")
print(documents[index])
print(f"\nsimilarity score : {score}")


# or


# # fix shapes & normalize
# if qry_emb.ndim == 1:
#     qry_emb = qry_emb / np.linalg.norm(qry_emb)
# else:
#     qry_emb = qry_emb.reshape(-1) / np.linalg.norm(qry_emb)
# 
# doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
# 
# sims = doc_embs.dot(qry_emb)
# top_idx = int(np.argmax(sims))
# print(f"Top match (score={sims[top_idx]:.4f}):\n{documents[top_idx]}")

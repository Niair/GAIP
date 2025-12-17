from langchain_google_genai import GoogleGenerativeAIEmbeddings
# pip install semantic-chunker-langchain

from semantic_chunker_langchain.chunker import SemanticChunker, SimpleSemanticChunker
# from semantic_chunker_langchain.extractors.pdf import extract_pdf
# from semantic_chunker_langchain.outputs.formatter import write_to_txt

from dotenv import load_dotenv
load_dotenv()

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.


Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

# # embedings
# embeddings = GoogleGenerativeAIEmbeddings(
#       model = "models/embedding-001"
# )

# text splitter
splitter = SemanticChunker(model_name="models/embedding-001")

chunks = splitter.split_text(sample)

print(len(chunks))



# Extract
# docs = extract_pdf("sample.pdf")

# Save to file
# write_to_txt(chunks, "output.txt")

# Using SimpleSemanticChunker
# simple_chunker = SimpleSemanticChunker(model_name="gpt-3.5-turbo")
# simple_chunks = simple_chunker.split_documents(docs)


# -----------------

from langchain_text_splitters import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Sample text
sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

# Initialize Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create splitter (official version)
splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

# Split text
chunks = splitter.split_text(sample)

print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks, 1):
    print(f"\n--- Chunk {i} ---")
    print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
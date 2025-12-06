from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings

from dotenv import load_dotenv
load_dotenv()

# text loader / document loader
# loader = TextLoader("sample.txt")
loader = PyPDFLoader("6_Runnables/2.0_attention_is_all_you_need.pdf")
documents = loader.load()

# Text Splitter - split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
chunks = text_splitter.split_documents(documents)

# Hugging Face Embedings object
emb = HuggingFaceEndpointEmbeddings(repo_id="BAAI/bge-base-en-v1.5")

# Storing the embeddings in FAISS/chroma vector database
vdb = Chroma.from_documents(chunks, emb)

# retriever (used to fetch the relevant chunks)
retriever = vdb.as_retriever()

# manually writing the query
query = "What is the encoding and decoding component of the transformer and how it is different from the encoder-decodder model."
retrieved_chunks = retriever.invoke(query)

# Combining them in an single prompt
relevent_test = "\n".join([docs.page_content for docs in retrieved_chunks])

# model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# Prompt template
template = f"Based on the following text, answer the question: {query} \n\n {relevent_test}"
answer = model.invoke(template)

# parser
parser = StrOutputParser()


print(parser.parse(answer.content))
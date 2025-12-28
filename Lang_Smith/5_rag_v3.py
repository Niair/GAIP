# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langsmith import traceable  # <---- important factor to trace all of the things in this llm project

load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = "RAG_LLM_v3"

PDF_PATH = "Lang_Smith\\docs\\islr.pdf"


# ----------------- helpers (not traced individually) -----------------

# 1) Load PDF
@traceable(name = 'load_pdf', tags = ['pdf_loader'], metadata = {'loader_name' : 'PyPDFLoader'})
def load_pdf(path : str):
      loader = PyPDFLoader(path)
      docs = loader.load()  # one Document per page
      return docs


# 2) Chunk  ## traced
@traceable(name = 'sptit_documents')
def split_documents(docs, chunk_size = 1000, chunk_overlap = 150):
      splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
      splits = splitter.split_documents(docs)
      return splits


# 3) Embed + index  ## traced
@traceable(name = 'build_vectorstore', tags = ['embedding', 'vectorstore'], metadata = {'embeddings_name' : 'GoogleGenerativeAIEmbeddings'})
def build_vectorstore(splits):
      emb = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")
      vs = FAISS.from_documents(splits, emb)
      return vs


# ----------------- parent setup function (traced) -----------------

# 4) setup pipeline  ## traced
@traceable(name="setup_pipeline", tags = ['setup'])
def setup_pipeline(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vs = build_vectorstore(splits)
    return vs


# ----------------- model, prompt, and run -----------------

# 5) ---- Pipeline ----
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# model
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

def format_docs(docs): return "\n\n".join(d.page_content for d in docs)


# ----------------- one top-level (root) run -----------------

@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_quey(pdf_path:str, question : str):

      vestor_store = setup_pipeline(pdf_path)
      retriever = vestor_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


      parallel = RunnableParallel({
      "context": retriever | RunnableLambda(format_docs),
      "question": RunnablePassthrough()
      })

      chain = parallel | prompt | llm | StrOutputParser()

      # Give the visible run name + tags/metadata so it’s easy to find:
      config = {
      "run_name": "pdf_rag_query"
      }

      result = chain.invoke(question, config = config)
      return result

# 6) Ask questions
if __name__ == "__main__":
      print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
      q = input("\nQ: ").strip()
      ans = setup_pipeline_and_quey(PDF_PATH, q)
      print("\nA:", ans)
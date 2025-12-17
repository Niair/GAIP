from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()


# loading youtube transcript
video_id = "Gfr50f6ZBvo" # only the ID, not full URL
try:
    ytt = YouTubeTranscriptApi()
    fetched = ytt.fetch(video_id, languages=["en"])
    transcript_list= fetched.to_raw_data()

    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    # print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")
except Exception as e:
    print(f"Error fetching transcript: {e}")


# text splitter and chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = text_splitter.create_documents([transcript])

# print(len(chunks)) / print(chunks[100])


# embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# vector store
vector_store = FAISS.from_documents(
    documents = chunks,
    embedding = embeddings
)


# retriever
retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs = {'k' : 5})


# model
model = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.7)

# prompt template
template = PromptTemplate(
    template = """
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.
        
        {context}
        Question: {question}
    """,
    input_variables = ['context', 'question']
)

# retrieved documents based on the query
question = "is the topic of aliens discussed in this video? if yes then what is discussed"

# retrived_docs = retriever.invoke(question)

# retrieved docs combined text
def concatinate(retrived_docs):
    context = "\n\n".join(doc.page_content for doc in retrived_docs)
    return context

# final_prompt
# final_prompt = template.invoke({"context":context, "question" : question})

# output parser
parser = StrOutputParser()

# answer generation
# answer = model.invoke(final_prompt)
# print(answer.content)

# chain

parallel_chain = RunnableParallel({
    "context" : retriever | RunnableLambda(concatinate),
    "question" : RunnablePassthrough()
})

chain = parallel_chain | template | model | parser

print(chain.invoke(question))
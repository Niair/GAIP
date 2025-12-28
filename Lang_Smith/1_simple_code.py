from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

prompt = PromptTemplate.from_template("Answer the following question - {question}")

model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.4)
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"question" : "What is the capital of India?"})

print(result)
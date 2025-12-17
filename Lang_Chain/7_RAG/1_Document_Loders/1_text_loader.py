from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

template = PromptTemplate(
      template = "Write a summry for the following poem - \n {poem}",
      input_variables = ['poem']
)

parser = StrOutputParser()

loader = TextLoader("7_RAG\cricket.txt", encoding = 'utf-8')

docs = loader.load()

# print(docs)
# print(type(docs))  # List

# print(len(docs))  # there is only 1 docs right now
# print(docs[0])
# print(type(docs[0]))  # this shows that the langchain document object is loaded in the list, which we can call it as multiple documents

# print(docs[0].page_content)
# print(docs[0].metadata)

chain = template | model | parser

result = chain.invoke({'poem':docs[0].page_content})

print(result)
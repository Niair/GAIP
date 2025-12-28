from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
import os

os.environ['LANGCHAIN_PROJECT'] = "Sequential_LLM_App2"

prompt1 = PromptTemplate(
      template = "Write an report on the the topic - {topic}",
      input_variables = ['topic']
)

prompt2 = PromptTemplate(
      template = "Give the 5 points summary of the following text \n {text}",
      input_variables = ['text']
)

model1 = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
model2 = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)
parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
      'run_name' : 'sequential chain',
      'tags' : ['llm app', 'report generation', 'summarizaton'],
      'metadata' : {'model1' : "llama-3.1-8b-instant", "model1_tep" : 0.7, "model2" : "llama-3.3-70b-versatile", "model2_tep" : 0.4, "parser" : "StrOutputParser"}
}

result = chain.invoke({"topic" : "what to learn in ai"}, config = config)

print(result)
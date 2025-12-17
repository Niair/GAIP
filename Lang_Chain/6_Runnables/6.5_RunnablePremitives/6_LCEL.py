from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableLambda, RunnableParallel, RunnableBranch
from dotenv import load_dotenv
load_dotenv()

# function to count the number of words
def word_count(text):

      return len(text.split())

# prompt template
template1 = PromptTemplate(
      template = "Write a detailed report about {topic}",
      input_variables = ['topic']
)

template2 = PromptTemplate(
      template = "Summarize the following text \n {text}",
      input_variables = ['text']
)

# model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# parser
parser = StrOutputParser()

# chain

report_gen_chain = template1 | model | parser

branch_chain = RunnableBranch(
      ( lambda x : len(x.split()) > 500, template2 | model | parser),
      RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

result = final_chain.invoke({"topic" : "Black Hole"})
print(result)
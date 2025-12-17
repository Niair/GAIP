from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
load_dotenv()

# prompt template
template1 = PromptTemplate(
      template = "Whrite a joke about {topic}",
      input_variables = ["topic"]
)

template2 = PromptTemplate(
      template = "Explain me the following joke - {joke}",
      input_variables = ["joke"]
)

# model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# paser
parser = StrOutputParser()

# chain
chain = RunnableSequence(template1, model, parser, template2, model, parser)

result = chain.invoke({"topic":"AI"})

print(result)
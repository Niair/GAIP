from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
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
joke_gen_chain = RunnableSequence(template1, model, parser)

parallel_chain = RunnableParallel({
      "joke" : RunnablePassthrough(),
      "explanation" : RunnableSequence(template2, model, parser)
})

chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = chain.invoke({"topic":"AI"})

print(result)
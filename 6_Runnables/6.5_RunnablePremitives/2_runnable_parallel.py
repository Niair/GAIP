from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv
load_dotenv()

# prompt template
template1 = PromptTemplate(
      template = "Generate a tweet about the {topic}",
      input_variables = ["topic"]
)

template2 = PromptTemplate(
      template = "Generate a linkedin post about {topic}",
      input_variables = ["topic"]
)

# model - ca take two models bot i am taking just one for now
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# parser
parser = StrOutputParser()

# runnable parllel - It is used for the parallel models for single input
parallel_chains = RunnableParallel({
      "tweet" : RunnableSequence(template1, model, parser),
      "linkedin" : RunnableSequence(template2, model, parser)
})

result = parallel_chains.invoke({"topic":"AI"})

print(result['tweet'])
print("-----------------------------------------------------------------------------")
print(result['linkedin'])
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableLambda, RunnableParallel
from dotenv import load_dotenv
load_dotenv()

# function to count the number of words
def word_count(text):

      return len(text.split())

# prompt template
template1 = PromptTemplate(
      template = "Write an joke about {topic}",
      input_variables = ['topic']
)

# model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# parser
parser = StrOutputParser()

# chain

joke_gen_chain = RunnableSequence(template1, model, parser)

parallel_chain = RunnableParallel({
      "joke" : RunnablePassthrough(),
      "word_count" : RunnableLambda(lambda x : len(x.split()))  # or --> RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({"topic" : "AI"})

# print(result)

print(f"{result['joke']} ----> Word Count : {result['word_count']}")
# why use the str output parser. because it become easy then the result.context

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

# model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)


# 1st prompt -> detailed response

template1 = PromptTemplate(
      template = "Write an detailed report on {topic}.", # <|im_start|>user\nWrite an detailed report on {topic}.<|im_end|>\n<|im_start|>assistant\n
      input_variables = ['topic']
)


# 2nd prompt -> summary response

template2 = PromptTemplate(
      template = "Write an 5 pointer summary on the fillowing text. \n {text}",
      input_variables = ['text']
)

# String Parser
parser = StrOutputParser()

# pipeline
chain = template1 | model | parser | template2 | model | parser
# template1 will give the prompt to model then using parser we only take the valuable data then, 
# we pass thet output to the template2 as input then to model again then only take useable things using parser

result = chain.invoke({"topic":"black hole"})


print(result)
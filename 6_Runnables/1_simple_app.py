from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

# model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# Prompt template
template = PromptTemplate(
      template = "Suggest me the catchy blog title about {topic}",
      input_variables = ["topic"]
)

# output parser
parser = StrOutputParser()

topic = input("Enter the title : ")

prompt = template.invoke({"topic":topic})

result = model.invoke(prompt)

print(parser.parse(result.content))
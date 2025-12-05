from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

# model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# Schema
class Output(BaseModel):

      written_by : str = Field("Auther name")
      content : str = Field("Detail report about the topic")
      refrence : str = Field("Refrence from wrere it is taken from.")

# parser
parser = PydanticOutputParser(pydantic_object=Output)


# Prompt template
template1 = PromptTemplate(
      template = "Generate an detail report on {topic} \n {format_instructions}",
      input_variables=['topic'],
      partial_variables = {'format_instructions' : parser.get_format_instructions()}
)

template2 = PromptTemplate(
      template = "Generate an 5 small pointer summary on {report} \n {format_instructions}",
      input_variables=["report"],
      partial_variables = {'format_instructions' : parser.get_format_instructions()}
)

chains = template1 | model | parser | template2 | model | parser

result = chains.invoke({"topic":"Agentic AI"})

print(result)

chains.get_graph().print_ascii()
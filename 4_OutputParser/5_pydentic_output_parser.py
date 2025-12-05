from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field 
from dotenv import load_dotenv

load_dotenv()

# Model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

# Schema
class Person(BaseModel):

      name : str = Field(description = "Name of the person.")
      age : int = Field(gt = 18, description = "Age of the person.")
      city : str = Field(description = "Name of the city in which this person belongs to.")

# parser
parser = PydanticOutputParser(pydantic_object = Person)

# prompt template
template = PromptTemplate(
    template="give me the name, age and city of a fictional person from {country} \n {format_instructions}",
    input_variables=["country"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# prompt = template.invoke({'country':"Japan"})
# 
# response = model.invoke(prompt)
# 
# final_result = parser.parse(response.content)
# 
# print(final_result)
# print(response.content)


# or

chain = template | model | parser

result = chain.invoke({'country':"Japan"})

print(result)
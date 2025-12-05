from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

# Model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

# Schema
structure = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

# parser
parser = StructuredOutputParser.from_response_schemas(structure)

# prompt template
template = PromptTemplate(
    template="Give 3 facts about the {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

prompt = template.format(topic="Black hole")

response = model.invoke(prompt)

final_result = parser.parse(response.content)

print(final_result)

# or

chain = template | model | parser

result = chain.invoke({'topic':"Black hole"})
print(result)
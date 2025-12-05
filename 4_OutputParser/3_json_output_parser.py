# even though we get the ouput in json but the structure is not in our hands as it is desided by the llm

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# parser
parser = JsonOutputParser()

# prompt
template = PromptTemplate(
    template='give me the name, age and city of a fictional person from {country} \n {format_instruction}',
    input_variables=['country'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)


# -------------------------------------------------------------

# -> prompt = template.invoke({'country':'Japan'}) # you could use the template.format if there is no input lke country

# -> result = model.invoke(prompt)

# print(result.content)

# ```json
# {
#   "name": "Ayame Sakura",
#   "age": 24,
#   "city": "Tokyo"
# }
# ```

# or

# -> output = parser.parse(result.content)
# -> print(output)
# print(output['name'])

# {'name': 'Sakura Ito', 'age': 28, 'city': 'Kyoto'}


# ----------------------------------------------------------------
# or ( using chains )

chain = template | model | parser

result = chain.invoke({'country':'Japan'})

print(result)

# {'name': 'Aiko Sato', 'age': 26, 'city': 'Kyoto'}
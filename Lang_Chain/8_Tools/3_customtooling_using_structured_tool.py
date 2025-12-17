from langchain_community.tools import StructuredTool
from pydantic import BaseModel, Field

class input_multiply(BaseModel):
      a : float = Field(..., description = "Variable 1")  # ... -- means --> required = true
      b : float = Field(..., description = "Variable 2")

def multiply(a : float, b : float) -> float:
      return a * b

multiply_tool = StructuredTool.from_function(
      func = multiply,
      name = "multiplication",
      description = "Multiplication of two numbers",
      args_schema = input_multiply
)

result = multiply_tool.invoke({"a" : 3, "b" : 5.5})
print(result)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args_schema)
from langchain_community.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class MyltiplyFunction(BaseModel):
      a : float = Field(..., description = "variable 1")
      b : float = Field(..., description = "variable 2")

class MyltiplyTool(BaseTool):

      name : str = "Multiplication"
      description : str = "Multiplication of two numbers"

      args_schema : Type[BaseModel] = MyltiplyFunction

      def _run(self, a : float, b : float) -> int:
            return a * b

multiply_tool = MyltiplyTool()
result = multiply_tool.invoke({"a" : 3, "b" : 5.5})
print(result)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args_schema)
from langchain_community.tools import tool

class ToolsForLLM:
      def __init__(self):
            pass
      
      @staticmethod
      @tool
      def multiplication(a : float, b : float) -> float:
            """
            Multiply two numbers
            """
            return a*b
      
obj = ToolsForLLM()
result = obj.multiplication.invoke({"a" : 3, "b" : 5.5})
print(result)

# NOTE - Better to use the function as stand alone function not with the class and another way as you can see in the screen.

print(obj.multiplication.name)
print(obj.multiplication.description)
print(obj.multiplication.args)

# what llm see
print(obj.multiplication.args_schema.model_json_schema())
from langchain_community.tools import tool

@tool
def add(a : float, b : float):
      """
      Add two numbers
      """
      return a + b

@tool
def multiply(a : float, b : float):
      """
      Multiply two numbers
      """
      return a * b

class MathClass:  # toolkit
      def get_tools(self):
            return [add, multiply]
      
      def get_tool_by_name(self, name: str):
        """Get a specific tool by its name"""
        tool_dict = {tool.name: tool for tool in self.get_tools()}
        return tool_dict.get(name)

# anoter way to create toolkit
math_tool_kit = {
    "multiply": multiply
}

toolkit = MathClass()
tools = toolkit.get_tools()

for tool in tools:
      print(f"--------- Tool Name : {tool.name} and Tool Description {tool.description} ---------")
      print(tool.invoke({"a" : 3, "b" : 5}))

print("-"*20)
addnum = toolkit.get_tool_by_name("add")
print(addnum.invoke({"a": 10, "b": 20}))

# more simple way
print("-"*20)
print(math_tool_kit["multiply"].invoke({"a": 6, "b": 8}))
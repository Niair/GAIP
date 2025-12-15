class Tools:
      def __init__(self):
            pass

      def searching_tool(self, query : str) -> str:
            from langchain_community.tools import DuckDuckGoSearchRun

            search_tool = DuckDuckGoSearchRun()

            results = search_tool.invoke(query)

            print(results)
      
      def shell_tool(self, query : str):
            from langchain_community.tools import ShellTool

            shell_command = ShellTool()

            results = shell_command.invoke(query)

            print(results)
      

obj = Tools()
# obj.searching_tool("recent news of Messi visiting India")
obj.shell_tool("ls")

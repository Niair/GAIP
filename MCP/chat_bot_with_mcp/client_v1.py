import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
import json
from dotenv import load_dotenv
load_dotenv()

SERVERS = {
      "math" : {
            "transport" : "stdio",
            "command" : "E:\\VS_Code\\Scripts\\uv.exe",
            "args" : [
                  "run",
                  "fastmcp",
                  "run",
                  "E:\\_Projects\\GAIP\\MCP\\chat_bot_with_mcp\\local_mcp.py"
            ],
            "env": {},
            "cwd": "E:\\_Projects\\GAIP\\MCP\\chat_bot_with_mcp"
      },

      "expense" : {
           "transport" : "streamable_http", # if not works write 'sse'
           "url" : "https://nihal-finance-server.fastmcp.app/mcp"
      },

      "manim-server": {
           "transport" : "stdio",
           "command": "E:\\VS_Code\\python.exe",
            "args": [
                  "E:\\_Projects\\manim-mcp-server\\src\\manim_server.py"
            ],
            "env": {
                  "MANIM_EXECUTABLE": "E:\\VS_Code\\Scripts\\manim.exe"
            },
            "cwd": "E:\\_Projects\\manim-mcp-server"
      }
}


async def main():
      
      client = MultiServerMCPClient(SERVERS)
      tools = await client.get_tools()

      named_tools = {}
      for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
        named_tools[tool.name] = tool
      
      model = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.7)

      llm_with_tools = model.bind_tools(tools)
      
      # prompt = "what is the product of 12 and 15 using the math tool?"
      # prompt = "what is the capital of japan"
      # prompt = "Add an expense - Rs 800 for grocries on 2nd Jan 2026"
      prompt = "Draw a triangle rotating in place using the manim tool"
      response = await llm_with_tools.ainvoke(prompt)

      if not getattr(response, "tool_calls", None):
           print(response.content)
           return
      
      tools_message = []
      for tc in response.tool_calls:
            selected_tool = tc['name']
            selected_tool_args = tc.get('args') or {}
            selected_tool_id = tc['id']

            print(f"executing remote tools : {selected_tool} \n with args: {selected_tool_args}")

            tool_result = await named_tools[selected_tool].ainvoke(selected_tool_args)
            tools_message.append(ToolMessage(content = json.dumps(tool_result), tool_name = selected_tool, tool_call_id = selected_tool_id))

      final_result = await llm_with_tools.ainvoke([prompt, response, *tools_message])
      print(final_result.content)


if __name__ == "__main__":
      asyncio.run(main())
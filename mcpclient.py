# server.py
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("mcpclient")


# Add an addition tool
@mcp.tool()
def add(a: int, b: int):
    """Add two numbers"""
    return a+b

# 添加這行代碼以確保服務器可以運行
if __name__ == "__main__":
    mcp.run()
# mcp_server_sample.py
from fastmcp import FastMCP

mcp = FastMCP("MyServer")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    # 啟動 HTTP 伺服器
    mcp.run(transport="http", host="127.0.0.1", port=8000)
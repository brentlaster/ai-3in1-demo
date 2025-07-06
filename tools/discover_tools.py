# discover_tools.py
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://127.0.0.1:8000/mcp/") as mcp:
        for tool in await mcp.list_tools():
            print(f"{tool.name}: {tool.description}")

asyncio.run(main())

from mcp.server.fastmcp import FastMCP

import continuum_client

mcp = FastMCP("Continuum")


@mcp.tool()
def store_memory(content: str, source: str):
    return continuum_client.store_memory(content, source)


@mcp.tool()
def search_memory(query: str):
    return continuum_client.search_memory(query)


@mcp.tool()
def get_memories():
    return continuum_client.get_memories()


@mcp.tool()
def get_memory(memory_id: int):
    return continuum_client.get_memory(memory_id)


if __name__ == "__main__":
    mcp.run()
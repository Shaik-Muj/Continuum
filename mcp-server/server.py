from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("Continuum")


@mcp.tool()
def search_memory(query: str):
    response = requests.get(
        "http://127.0.0.1:8000/search",
        params={"q": query}
    )

    return response.json()


if __name__ == "__main__":
    mcp.run()
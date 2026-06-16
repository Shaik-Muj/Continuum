from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("Continuum")

BASE_URL = "http://127.0.0.1:8000"


@mcp.tool()
def search_memory(query: str):
    response = requests.get(
        f"{BASE_URL}/search",
        params={"q": query}
    )

    return response.json()


@mcp.tool()
def store_memory(content: str, source: str):
    response = requests.post(
        f"{BASE_URL}/memory",
        json={
            "user_id": "demo-id",
            "content": content,
            "source": source
        }
    )

    return response.json()


if __name__ == "__main__":
    mcp.run()
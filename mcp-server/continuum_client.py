import requests

BASE_URL = "http://127.0.0.1:8000"


def _handle_response(response):
    """
    Returns a consistent response for both success and failure.
    """
    if response.ok:
        return response.json()

    return {
        "status_code": response.status_code,
        "error": response.json()
    }


def store_memory(content: str, source: str):
    response = requests.post(
        f"{BASE_URL}/memory",
        json={
            "user_id": "demo-id",
            "content": content,
            "source": source
        }
    )

    return _handle_response(response)


def search_memory(query: str):
    response = requests.get(
        f"{BASE_URL}/search",
        params={"q": query}
    )

    return _handle_response(response)


def get_memories():
    response = requests.get(
        f"{BASE_URL}/memory"
    )

    return _handle_response(response)


def get_memory(memory_id: int):
    response = requests.get(
        f"{BASE_URL}/memory/{memory_id}"
    )

    return _handle_response(response)


def delete_memory(memory_id: int):
    response = requests.delete(
        f"{BASE_URL}/memory/{memory_id}"
    )

    return _handle_response(response)
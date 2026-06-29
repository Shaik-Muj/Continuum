import requests

BASE_URL = "http://127.0.0.1:8000"


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


def search_memory(query: str):
    response = requests.get(
        f"{BASE_URL}/search",
        params={"q": query}
    )

    return response.json()


def get_memories():
    response = requests.get(
        f"{BASE_URL}/memory"
    )

    return response.json()


def get_memory(memory_id: int):
    response = requests.get(
        f"{BASE_URL}/memory/{memory_id}"
    )

    return response.json()

def delete_memory(memory_id: int):
    response = requests.delete(
        f"{BASE_URL}/memory/{memory_id}"
    )

    return response.json()
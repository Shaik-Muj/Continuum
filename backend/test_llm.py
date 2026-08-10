from core.huggingface_llm import HuggingFaceLLM


llm = HuggingFaceLLM()

response = llm.generate(
    "Explain what a knowledge graph is in two sentences."
)

print("\nLLM RESPONSE:\n")
print(response)
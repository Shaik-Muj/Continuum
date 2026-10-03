from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer to provide a clean interface
    for generating embeddings throughout the application.
    """

    def __init__(self):
        self.model_name = "BAAI/bge-small-en-v1.5"

        # Prefix the BGE v1.5 model card recommends for short queries
        # searched against passages. Stored memories are embedded without it.
        self.query_instruction = (
            "Represent this sentence for searching relevant passages: "
        )

        self.model = SentenceTransformer(
            self.model_name
        )

    def embed(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        Generate normalized embeddings for a batch of texts.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors.
        """

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return vectors.tolist()

    def embed_query(
        self,
        query: str
    ) -> list[float]:
        """
        Generate a normalized embedding for a search query.

        Args:
            query: Search query.

        Returns:
            Query embedding vector, computed with the BGE query instruction.
        """

        return self.embed(
            [self.query_instruction + query]
        )[0]

    def dimension(self) -> int:
        """
        Returns the embedding dimension of the loaded model.
        """

        return self.model.get_sentence_embedding_dimension()


embedding_model = EmbeddingModel()
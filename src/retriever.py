"""
Módulo responsável por consultar o banco vetorial.
"""

from langchain_community.vectorstores import Chroma

from embeddings import EmbeddingGenerator


class Retriever:

    def __init__(self):

        generator = EmbeddingGenerator()

        self.vector_store = Chroma(
            persist_directory="vector_db",
            embedding_function=generator.get_embedding_model(),
        )

    def search(self, question: str, k: int = 3):
        """
        Busca os k documentos mais relevantes.
        """
        return self.vector_store.similarity_search(question, k=k)


if __name__ == "__main__":

    retriever = Retriever()

    pergunta = input("Digite sua pergunta: ")

    resultados = retriever.search(pergunta)

    print("\n" + "=" * 60)

    for i, doc in enumerate(resultados, start=1):

        print(f"\nResultado {i}")
        print("-" * 60)

        print(doc.page_content[:500])

        print("\nDocumento:",
              doc.metadata.get("source", "Desconhecido"))

        print("Página:",
              doc.metadata.get("page", "?"))
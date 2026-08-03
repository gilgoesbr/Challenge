"""
Módulo responsável por criar e persistir o banco vetorial.
"""

from langchain_community.vectorstores import Chroma

from ingest import PDFIngestor
from embeddings import EmbeddingGenerator


class VectorStoreManager:

    def __init__(self):

        self.persist_directory = "vector_db"

    def create(self):

        ingestor = PDFIngestor("docs/pdfs")

        documents = ingestor.load_documents()

        generator = EmbeddingGenerator()

        chunks = generator.split_documents(documents)

        embeddings = generator.get_embedding_model()

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_directory,
        )

        return vector_store


if __name__ == "__main__":

    manager = VectorStoreManager()

    vector_db = manager.create()

    print("=" * 60)
    print("Banco vetorial criado com sucesso.")
    print("=" * 60)
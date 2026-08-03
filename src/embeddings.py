"""
Módulo responsável por preparar documentos para geração de embeddings.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from ingest import PDFIngestor

class EmbeddingGenerator:
    """
    Responsável por criar chunks e disponibilizar
    o modelo de embeddings.
    """

    def __init__(self):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def split_documents(self, documents):            
        """
        Divide os documentos em pequenos chunks.
        """
        return self.splitter.split_documents(documents)

    def get_embedding_model(self):
        """
        Retorna o modelo de embeddings.
        """
        return self.embedding_model
    
if __name__ == "__main__":

    ingestor = PDFIngestor("docs/pdfs")

    documents = ingestor.load_documents()

    generator = EmbeddingGenerator()

    chunks = generator.split_documents(documents)

    print("=" * 60)
    print(f"Documentos carregados : {len(documents)}")
    print(f"Chunks criados        : {len(chunks)}")
    print("=" * 60)

    print("\nPrimeiro chunk:\n")
    print(chunks[0].page_content[:500])
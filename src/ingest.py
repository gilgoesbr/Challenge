"""
Módulo responsável pela ingestão de documentos PDF.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document


class PDFIngestor:
    """
    Responsável por carregar documentos PDF de um diretório.
    """

    def __init__(self, pdf_directory: str | Path):
        self.pdf_directory = Path(pdf_directory)

    def load_documents(self) -> list[Document]:
        """
        Carrega todos os PDFs encontrados no diretório.

        Returns:
            list[Document]: Lista de documentos carregados.
        """

        if not self.pdf_directory.exists():
            raise FileNotFoundError(
                f"Diretório não encontrado: {self.pdf_directory}"
            )

        loader = PyPDFDirectoryLoader(str(self.pdf_directory))
        return loader.load()
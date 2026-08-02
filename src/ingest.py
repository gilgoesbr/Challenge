from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader


class PDFIngestor:
    def __init__(self, pdf_directory: str):
        self.pdf_directory = Path(pdf_directory)

    def load_documents(self):
        loader = PyPDFDirectoryLoader(str(self.pdf_directory))
        documents = loader.load()

        print(f"{len(documents)} páginas carregadas.")

        return documents


if __name__ == "__main__":

    pdf_path = Path(__file__).parent.parent / "docs" / "pdfs"

    ingestor = PDFIngestor(pdf_path)

    documents = ingestor.load_documents()

    print("=" * 60)
    print(f"Total de páginas: {len(documents)}")
    print("=" * 60)

    for i, doc in enumerate(documents[:3], start=1):
        print(f"\nPágina {i}")
        print("-" * 40)
        print(doc.page_content[:300])
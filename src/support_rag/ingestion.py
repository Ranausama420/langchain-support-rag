import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def load_chunks(settings) -> list[Document]:
    header = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")], strip_headers=False
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = []
    for path in sorted(settings.manuals_directory.glob("*.md")):
        sections = header.split_text(path.read_text(encoding="utf-8"))
        for number, doc in enumerate(splitter.split_documents(sections)):
            doc.metadata = {
                "source": path.name,
                "section": doc.metadata.get("section", doc.metadata.get("title", "Manual")),
            }
            fingerprint = f"{path.name}:{number}:{doc.page_content}"
            doc.metadata["id"] = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
            chunks.append(doc)
    if not chunks:
        raise ValueError("No Markdown content found in MANUALS_DIRECTORY")
    return chunks

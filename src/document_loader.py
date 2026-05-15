import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree


WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def clean_text(text):
    """Normalize line endings and collapse noisy whitespace."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def read_text_file(file_path):
    """Read a plain-text source with a few encoding fallbacks."""
    for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return file_path.read_text(encoding="utf-8", errors="replace")


def read_docx_file(file_path):
    """Extract readable paragraph text from a DOCX file without extra dependencies."""
    paragraphs = []

    with zipfile.ZipFile(file_path) as archive:
        xml_bytes = archive.read("word/document.xml")

    root = ElementTree.fromstring(xml_bytes)
    for paragraph in root.iter(f"{WORD_NAMESPACE}p"):
        paragraph_text_parts = []
        for text_node in paragraph.iter(f"{WORD_NAMESPACE}t"):
            if text_node.text:
                paragraph_text_parts.append(text_node.text)
        paragraph_text = "".join(paragraph_text_parts).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)

    return "\n\n".join(paragraphs)


def read_pdf_file(file_path):
    """Extract text from a text-based PDF when PyPDF is available."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF ingestion requires the optional 'pypdf' dependency. "
            "Install requirements and rebuild the processed assets."
        ) from exc

    reader = PdfReader(str(file_path))
    page_text = []
    for page in reader.pages:
        extracted_text = page.extract_text() or ""
        if extracted_text.strip():
            page_text.append(extracted_text.strip())

    return "\n\n".join(page_text)


def read_document_text(file_path):
    """Read a supported assignment-brief document into normalized plain text."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return clean_text(read_text_file(path))
    if suffix == ".docx":
        return clean_text(read_docx_file(path))
    if suffix == ".pdf":
        return clean_text(read_pdf_file(path))

    raise ValueError(
        f"Unsupported brief format for {path.name}. "
        "Supported formats are .txt, .md, .docx, and .pdf."
    )

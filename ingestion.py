"""
ingestion.py — PDF extraction, chunking, embedding
Works with ANY uploaded PDF — no hardcoded content.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    text: str
    document_type: str        # "codebook" | "specification"
    discipline: str
    project_id: Optional[str]
    source_doc: str
    page: int = 1
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def metadata(self) -> dict:
        return {
            "chunk_id":      self.chunk_id,
            "document_type": self.document_type,
            "discipline":    self.discipline,
            "project_id":    self.project_id,
            "source_doc":    self.source_doc,
            "page":          self.page,
        }


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> List[dict]:
    """
    Returns list of {page: int, text: str} dicts.
    Uses PyMuPDF (fitz). Falls back to raw byte decode for plain .txt files.
    """
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": i, "text": text})
        return pages
    except Exception:
        # Plain text fallback
        text = file_bytes.decode("utf-8", errors="ignore")
        return [{"page": 1, "text": text}]


def extract_text_from_upload(uploaded_file) -> List[dict]:
    """Accept a Streamlit UploadedFile object."""
    return extract_text_from_pdf(uploaded_file.read())


def _extract_project_id_from_content(pages: List[dict]) -> str:
    """Try to extract a consistent project ID from document content"""
    import re
    
    for page in pages[:5]:  # Check first 5 pages
        text = page["text"]
        
        # Look for project reference patterns (case-insensitive)
        patterns = [
            r'project\s+reference[:\s]*([0-9-]+)',
            r'project\s+ref[:\s]*([0-9-]+)', 
            r'reference[:\s]*([0-9-]+)',
            r'ref[:\s]*([0-9-]+)',
            r'([0-9]{4}-[0-9]{3,4})',  # Pattern like 2024-0026
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                project_id = match.group(1)
                # Validate it looks like a project ID
                if re.match(r'^[0-9]{4}-[0-9]{3,4}', project_id):
                    return project_id
    
    return None


def _detect_document_type_from_content(pages: List[dict], filename: str) -> str:
    """Detect if this is a specification, codebook, or other document type"""
    import re
    
    # Check filename first
    filename_lower = filename.lower()
    if any(word in filename_lower for word in ['spec', 'specification']):
        return "specification"
    elif any(word in filename_lower for word in ['codebook', 'code']):
        return "codebook"
    
    # Check first few pages content
    combined_text = " ".join(page["text"] for page in pages[:3]).lower()
    
    if any(phrase in combined_text for phrase in [
        'codebook', 'inspection reference', 'hold points', 'witness points'
    ]):
        return "codebook"
    elif any(phrase in combined_text for phrase in [
        'specification', 'scope of works', 'codes and standards'
    ]):
        return "specification"
    
    # Default classification
    return "specification"


# ---------------------------------------------------------------------------
# Discipline detection
# ---------------------------------------------------------------------------

_DISCIPLINE_MAP = {
    "fire_safety":  ["fire", "exit", "escape", "smoke", "flame", "evacuation", "sprinkler"],
    "structural":   ["concrete", "stair", "structural", "riser", "tread", "foundation",
                     "reinforced", "cover", "beam", "column", "slab", "load"],
    "ventilation":  ["ventilation", "air change", "fresh air", "hvac", "mechanical",
                     "exhaust", "supply air"],
    "electrical":   ["electrical", "wiring", "circuit", "voltage", "cable", "earthing"],
    "plumbing":     ["plumbing", "drainage", "water supply", "pipe", "sanitary"],
}

def _discipline_from_text(text: str) -> str:
    text_lower = text.lower()
    for discipline, keywords in _DISCIPLINE_MAP.items():
        if any(k in text_lower for k in keywords):
            return discipline
    return "general"


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def _make_id(source: str, page: int, index: int) -> str:
    raw = f"{source}_{page}_{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def chunk_pages(
    pages: List[dict],
    document_type: str,
    source_doc: str,
    project_id: Optional[str] = None,
    chunk_size: int = 400,
    overlap: int = 80,
) -> List[Chunk]:
    """
    Paragraph-aware sliding window chunker.
    Each page is split on double newlines first, then windowed if needed.
    """
    chunks: List[Chunk] = []
    index = 0

    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]

        # Split into natural paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        buffer = ""
        for para in paragraphs:
            if len(buffer) + len(para) <= chunk_size:
                buffer = (buffer + "\n\n" + para).strip()
            else:
                if buffer:
                    cid = _make_id(source_doc, page_num, index)
                    chunks.append(Chunk(
                        chunk_id=cid,
                        text=buffer,
                        document_type=document_type,
                        discipline=_discipline_from_text(buffer),
                        project_id=project_id,
                        source_doc=source_doc,
                        page=page_num,
                    ))
                    index += 1
                # Start new buffer with overlap from previous
                overlap_text = buffer[-overlap:] if len(buffer) > overlap else buffer
                buffer = (overlap_text + "\n\n" + para).strip()

        if buffer:
            cid = _make_id(source_doc, page_num, index)
            chunks.append(Chunk(
                chunk_id=cid,
                text=buffer,
                document_type=document_type,
                discipline=_discipline_from_text(buffer),
                project_id=project_id,
                source_doc=source_doc,
                page=page_num,
            ))
            index += 1

    return chunks


# ---------------------------------------------------------------------------
# Embedding — sentence-transformers required (no fallback)
# ---------------------------------------------------------------------------

_embedder = None
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 80 MB, downloads once, cached by HuggingFace


def _get_embedder():
    """
    Loads the sentence-transformer model on first call.
    Raises a clear error if sentence-transformers is not installed.
    """
    global _embedder
    if _embedder is not None:
        return _embedder
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required.\n"
            "Run:  pip install sentence-transformers"
        )
    _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def embed_chunks(chunks: List[Chunk], status_callback=None) -> List[Chunk]:
    """
    Embed all chunks using the real sentence-transformer model.
    status_callback: optional callable(str) for progress messages (e.g. st.status).
    """
    if status_callback:
        status_callback(f"Loading embedding model ({EMBEDDING_MODEL})…")

    embedder = _get_embedder()

    if status_callback:
        status_callback(f"Embedding {len(chunks)} chunks…")

    texts   = [c.text for c in chunks]
    vectors = embedder.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
        batch_size=64,
    )
    for chunk, vec in zip(chunks, vectors):
        chunk.embedding = vec.astype(np.float32)

    return chunks


# ---------------------------------------------------------------------------
# Public: ingest from uploaded files
# ---------------------------------------------------------------------------

def ingest_uploaded(
    code_file,           # Streamlit UploadedFile
    project_file,        # Streamlit UploadedFile
) -> List[Chunk]:
    """
    Extract → chunk → embed both uploaded documents.
    Returns merged list of Chunks ready for indexing.
    """
    code_pages    = extract_text_from_upload(code_file)
    project_pages = extract_text_from_upload(project_file)
    
    # Try to extract project ID from document content
    project_id = _extract_project_id_from_content(code_pages + project_pages)
    if not project_id:
        project_id = "uploaded-project"

    # Detect document types properly
    code_doc_type = _detect_document_type_from_content(code_pages, code_file.name)
    project_doc_type = _detect_document_type_from_content(project_pages, project_file.name)

    code_chunks = chunk_pages(
        code_pages,
        document_type=code_doc_type,
        source_doc=code_file.name,
        project_id=project_id,  # Same project ID for both documents
    )
    project_chunks = chunk_pages(
        project_pages,
        document_type=project_doc_type,
        source_doc=project_file.name,
        project_id=project_id,  # Same project ID for both documents
    )

    all_chunks = code_chunks + project_chunks
    embed_chunks(all_chunks)
    return all_chunks
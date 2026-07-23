"""
loaders.py
==========
Document loaders for resume / job-description input.

Supports three file types so you can feed REAL resumes, not just plain text:
  - .txt   plain text
  - .pdf   extracted with pypdf
  - .docx  extracted with python-docx

Also provides auto-discovery: drop files into the data/ folder named like
`resume_*.pdf` (or .docx/.txt) and they are picked up automatically.
"""

import os
import glob

# File extensions we know how to read.
SUPPORTED = (".txt", ".pdf", ".docx")


def read_document(path: str) -> str:
    """Read a .txt, .pdf, or .docx file and return its plain text.

    Raises a clear error for unsupported types or missing files.
    """
    if not os.path.exists(path):
        raise FileNotFoundError("File not found: %s" % path)

    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()

    if ext == ".pdf":
        # Lazy import so .txt-only users don't need pypdf installed.
        from pypdf import PdfReader
        reader = PdfReader(path)
        # Some pages can return None for extract_text(); guard with "or ''".
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        if not text.strip():
            raise ValueError(
                "No text could be extracted from %s. If it is a scanned/image "
                "PDF, OCR is required (out of scope here)." % path)
        return text

    if ext == ".docx":
        import docx  # from the python-docx package
        document = docx.Document(path)
        return "\n".join(p.text for p in document.paragraphs)

    raise ValueError(
        "Unsupported file type '%s'. Supported: %s" % (ext, ", ".join(SUPPORTED)))


def _infer_band(filename: str) -> str:
    """Best-effort band label from the filename (resume_strong.pdf -> 'strong')."""
    name = os.path.splitext(os.path.basename(filename))[0].lower()
    for band in ("strong", "average", "weak"):
        if band in name:
            return band
    return "unknown"


def discover_resumes(data_dir: str):
    """Find all resume files in data_dir, any supported extension.

    A 'resume file' is any file whose name starts with 'resume'.
    Returns a list of dicts: {"name", "band", "path"}.
    """
    found = []
    for ext in SUPPORTED:
        found.extend(glob.glob(os.path.join(data_dir, "resume*" + ext)))
    found = sorted(set(found))

    candidates = []
    for path in found:
        base = os.path.splitext(os.path.basename(path))[0]
        candidates.append({
            "name": base,                 # e.g. "resume_strong"
            "band": _infer_band(path),    # strong / average / weak / unknown
            "path": path,
        })
    return candidates


def find_job_description(data_dir: str) -> str:
    """Locate the job-description file (job_description.txt/.pdf/.docx)."""
    for ext in SUPPORTED:
        path = os.path.join(data_dir, "job_description" + ext)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No job_description.(txt|pdf|docx) found in %s" % data_dir)

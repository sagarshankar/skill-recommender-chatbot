"""CLI utility to ingest protocol documents (PDF + text) into a FAISS index.

Usage:
    python -m clinical_trial_agent.src.ingest --docs-dir ./protocols/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-small"


def load_documents(docs_dir: Path) -> list:
    documents = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            documents.extend(PyPDFLoader(str(path)).load())
        elif suffix in {".txt", ".md", ".markdown"}:
            documents.extend(TextLoader(str(path)).load())
    return documents


def build_index(docs_dir: Path, index_dir: Path) -> None:
    docs = load_documents(docs_dir)
    if not docs:
        print(f"No PDF or text files found in {docs_dir}", file=sys.stderr)
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    print(f"FAISS index saved to {index_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest clinical trial protocol documents into a FAISS index"
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        required=True,
        help="Directory containing protocol PDFs and/or text files",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory to save the FAISS index",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required", file=sys.stderr)
        sys.exit(1)

    if not args.docs_dir.is_dir():
        print(f"Error: {args.docs_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    build_index(args.docs_dir, args.index_dir)


if __name__ == "__main__":
    main()

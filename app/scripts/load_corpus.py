"""Load + chunk the scikit-learn documentation corpus.

The corpus covers the entire Classic ML cycle of the course (weeks 7-10):
linear models, decision trees, ensembles, cross-validation, metrics,
preprocessing, pipelines, grid search, missing values, feature selection.

Local markdown files in data/local/*.md are also indexed — they hold
the service self-description ("about.md") so the bot can answer
meta-questions like "what do you know about?".
"""

import json
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# scikit-learn sections covering the entire Classic ML cycle of the course
SEED_URLS = [
    # Models
    "https://scikit-learn.org/stable/modules/linear_model.html",
    "https://scikit-learn.org/stable/modules/tree.html",
    "https://scikit-learn.org/stable/modules/ensemble.html",
    # Validation + metrics
    "https://scikit-learn.org/stable/modules/cross_validation.html",
    "https://scikit-learn.org/stable/modules/model_evaluation.html",
    # Data handling
    "https://scikit-learn.org/stable/modules/preprocessing.html",
    "https://scikit-learn.org/stable/modules/impute.html",
    "https://scikit-learn.org/stable/modules/feature_selection.html",
    # Workflow
    "https://scikit-learn.org/stable/modules/compose.html",
    "https://scikit-learn.org/stable/modules/grid_search.html",
]

LOCAL_DIR = Path("data/local")
OUTPUT_PATH = Path("data/corpus_chunks.jsonl")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def clean_html(html: str) -> str:
    """Strip scripts, styles, navigation; return plain text."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def load_url_corpus() -> list[Document]:
    """Download the scikit-learn documentation pages."""
    all_docs: list[Document] = []
    for url in SEED_URLS:
        print(f"Loading {url} ...")
        loader = RecursiveUrlLoader(url=url, max_depth=1, extractor=clean_html)
        docs = loader.load()
        all_docs.extend(docs)
        print(f"  -> {len(docs)} pages")
    return all_docs


def load_local_corpus() -> list[Document]:
    """Read any *.md files from data/local/ — these are service-internal docs."""
    if not LOCAL_DIR.exists():
        return []
    docs: list[Document] = []
    for path in sorted(LOCAL_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": f"local://{path.name}",
                    "title": path.stem.replace("_", " ").title(),
                },
            )
        )
        print(f"Local file: {path.name} ({len(text)} chars)")
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(docs)


def save_chunks(chunks: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            row = {"content": chunk.page_content, "metadata": dict(chunk.metadata)}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} chunks to {path}")


def main() -> None:
    url_docs = load_url_corpus()
    local_docs = load_local_corpus()
    all_docs = url_docs + local_docs

    chunks = chunk_documents(all_docs)
    print(f"\nTotal: {len(all_docs)} pages -> {len(chunks)} chunks")
    save_chunks(chunks, OUTPUT_PATH)


if __name__ == "__main__":
    main()

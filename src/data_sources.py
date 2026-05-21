from __future__ import annotations

import logging
from datetime import date
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from src import helper
from src.config import Config
from src.curated_data import CURATED_DOCS

logger = logging.getLogger(__name__)

WEB_SOURCES = [
    {"label": "WHO: Hypertension", "url": "https://www.who.int/news-room/fact-sheets/detail/hypertension"},
    {"label": "WHO: Diabetes", "url": "https://www.who.int/news-room/fact-sheets/detail/diabetes"},
    {"label": "WHO: Cardiovascular diseases", "url": "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)"},
    {"label": "MedlinePlus: Stroke", "url": "https://medlineplus.gov/stroke.html"},
    {"label": "MedlinePlus: Asthma", "url": "https://medlineplus.gov/asthma.html"},
]

REQUEST_TIMEOUT = 20
USER_AGENT = "medical-chatbot-ingest/1.0 (educational use)"
_STRIP_TAGS = ["script", "style", "noscript", "header", "footer", "nav", "form"]


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def fetch_web_sources(sources: list | None = None) -> List[Document]:
    """Fetch and clean each web source. Fail-soft: skip on any error."""
    sources = WEB_SOURCES if sources is None else sources
    today = date.today().isoformat()
    docs: List[Document] = []
    for src in sources:
        try:
            resp = requests.get(
                src["url"], timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}
            )
            resp.raise_for_status()
            text = html_to_text(resp.text)
            if not text.strip():
                logger.warning("Empty content from %s; skipping", src["url"])
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": src["label"],
                        "source_type": "web",
                        "url": src["url"],
                        "retrieved_at": today,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001 - fail-soft by design
            logger.warning("Failed to fetch %s: %s", src["url"], exc)
    return docs


def gather_documents(config: Config | None = None) -> List[Document]:
    """Aggregate documents from every enabled source."""
    config = config or Config.from_env()
    docs: List[Document] = []
    if config.enable_curated:
        docs.extend(CURATED_DOCS)
    if config.enable_dropin:
        file_docs = helper.load_files(config.data_dir)
        for doc in file_docs:
            doc.metadata.setdefault("source_type", "file")
        docs.extend(file_docs)
    if config.enable_web:
        docs.extend(fetch_web_sources())
    return docs

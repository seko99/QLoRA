#!/usr/bin/env python3
import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from openai import OpenAI

try:
    from llama_index.core import Document, Settings, VectorStoreIndex
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.node_parser import SentenceSplitter
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(
        "Не удалось импортировать llama_index.core. Установите llama-index-core."
    ) from exc

class OpenAICompatibleEmbedding(BaseEmbedding):
    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str,
        timeout: float,
        embed_batch_size: int = 64,
    ) -> None:
        super().__init__(model_name=model_name, embed_batch_size=embed_batch_size)
        self._model_name = model_name
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def _embed_many(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(model=self._model_name, input=texts)
        return [item.embedding for item in resp.data]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed_many([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_many([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed_many(texts)


def read_markdown_documents(docs_dir: Path, recursive: bool) -> List[Document]:
    if not docs_dir.exists():
        raise FileNotFoundError(f"Папка не найдена: {docs_dir}")
    if not docs_dir.is_dir():
        raise NotADirectoryError(f"Ожидалась папка: {docs_dir}")

    pattern = "**/*.md" if recursive else "*.md"
    files = sorted(docs_dir.glob(pattern))
    if not files:
        raise RuntimeError(f"В папке {docs_dir} не найдено markdown-файлов")

    docs: List[Document] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        docs.append(
            Document(
                text=text,
                metadata={
                    "file_name": path.name,
                    "file_path": str(path.resolve()),
                    "source": str(path.resolve()),
                },
                doc_id=path.stem,
            )
        )
    return docs


def build_openai_embedding(
    base_url: str,
    api_key: str,
    embed_model: str,
    timeout: float,
) -> OpenAICompatibleEmbedding:
    # Быстрая проверка доступности embedding endpoint до построения индекса.
    probe_client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    probe_client.embeddings.create(model=embed_model, input=["healthcheck"])

    return OpenAICompatibleEmbedding(
        model_name=embed_model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Построение и локальное сохранение LlamaIndex индекса по markdown-документам. "
            "Индекс можно использовать для hybrid retrieval (BM25 + vector)."
        )
    )
    parser.add_argument("--docs-dir", type=Path, required=True, help="Папка с .md документами")
    parser.add_argument("--persist-dir", type=Path, required=True, help="Куда сохранить индекс")
    parser.add_argument("--recursive", action="store_true", help="Искать markdown в подпапках")

    parser.add_argument("--chunk-size", type=int, default=900, help="Размер чанка в токенах")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Перекрытие чанков в токенах")

    parser.add_argument("--base-url", default="http://localhost:1234/v1", help="OpenAI-совместимый URL")
    parser.add_argument("--api-key", default="lm-studio", help="API key (для LM Studio обычно любое значение)")
    parser.add_argument("--embed-model", default="text-embedding-nomic-embed-text-v1.5", help="Embedding model")
    parser.add_argument("--timeout", type=float, default=120.0, help="Таймаут запроса к API (сек)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size должен быть > 0")
    if args.chunk_overlap < 0:
        raise ValueError("--chunk-overlap должен быть >= 0")
    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("--chunk-overlap должен быть меньше --chunk-size")

    print(f"[index] loading docs from {args.docs_dir}")
    docs = read_markdown_documents(args.docs_dir, recursive=args.recursive)
    print(f"[index] loaded {len(docs)} documents")

    print(f"[index] initializing embedding model: {args.embed_model}")
    embed_model = build_openai_embedding(
        base_url=args.base_url,
        api_key=args.api_key,
        embed_model=args.embed_model,
        timeout=args.timeout,
    )
    Settings.embed_model = embed_model
    Settings.llm = None

    splitter = SentenceSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(docs)
    print(f"[index] built {len(nodes)} nodes")

    index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)

    args.persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(args.persist_dir))
    print(f"[done] index persisted to {args.persist_dir}")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "docs_dir": str(args.docs_dir.resolve()),
        "persist_dir": str(args.persist_dir.resolve()),
        "documents": len(docs),
        "nodes": len(nodes),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "embed_model": args.embed_model,
        "base_url": args.base_url,
    }
    manifest_path = args.persist_dir / "index_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] manifest: {manifest_path}")


if __name__ == "__main__":
    main()

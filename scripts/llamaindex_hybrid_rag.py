#!/usr/bin/env python3
import argparse
import asyncio
import json
import locale
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever

try:
    import readline  # type: ignore
except Exception:  # noqa: BLE001
    readline = None

try:
    from prompt_toolkit import PromptSession  # type: ignore
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory  # type: ignore
    from prompt_toolkit.completion import WordCompleter  # type: ignore
    from prompt_toolkit.history import FileHistory  # type: ignore
except Exception:  # noqa: BLE001
    PromptSession = None
    FileHistory = None
    WordCompleter = None
    AutoSuggestFromHistory = None

try:
    from rich.console import Console  # type: ignore
    from rich.panel import Panel  # type: ignore
    from rich.table import Table  # type: ignore
except Exception:  # noqa: BLE001
    Console = None
    Panel = None
    Table = None


SYSTEM_PROMPT = (
    "Ты ассистент техподдержки торговой компании. "
    "Отвечай строго по предоставленному контексту, кратко и по делу. "
    "Если данных недостаточно, задай один уточняющий вопрос. "
    "Не выдумывай факты. В конце добавь блок 'Источники:'."
)

SYSTEM_PROMPT_NO_RAG = (
    "Ты ассистент техподдержки торговой компании. "
    "RAG отключен: отвечай кратко и аккуратно, без выдумок. "
    "Если уверенности нет, явно скажи об этом и предложи следующий безопасный шаг."
)

CHAT_COMMANDS = ["/exit", "/reset", "/save"]


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


def load_manifest(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:  # noqa: BLE001
        return {}
    return {}


def build_retrievers(
    index_dir: Path,
    embed_model_name: str,
    base_url: str,
    api_key: str,
    timeout: float,
    vector_top_k: int,
    bm25_top_k: int,
    bm25_language: str,
) -> Tuple[BaseRetriever, BM25Retriever]:
    embed_model = OpenAICompatibleEmbedding(
        model_name=embed_model_name,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )
    Settings.embed_model = embed_model
    Settings.llm = None

    storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
    index = load_index_from_storage(storage_context=storage_context, embed_model=embed_model)
    vector_retriever = index.as_retriever(similarity_top_k=vector_top_k)
    bm25_retriever = BM25Retriever.from_defaults(
        index=index,
        similarity_top_k=bm25_top_k,
        language=bm25_language,
    )
    return vector_retriever, bm25_retriever


def hybrid_retrieve(
    query: str,
    vector_retriever: BaseRetriever,
    bm25_retriever: BM25Retriever,
    top_k: int,
    vector_weight: float,
    bm25_weight: float,
    rrf_k: int = 60,
) -> List[NodeWithScore]:
    vec_nodes = vector_retriever.retrieve(query)
    bm_nodes = bm25_retriever.retrieve(query)

    merged: Dict[str, Dict[str, object]] = {}
    for rank, nws in enumerate(vec_nodes, start=1):
        key = nws.node.node_id
        merged.setdefault(key, {"node": nws.node, "score": 0.0})
        merged[key]["score"] = float(merged[key]["score"]) + vector_weight * (1.0 / (rrf_k + rank))
    for rank, nws in enumerate(bm_nodes, start=1):
        key = nws.node.node_id
        merged.setdefault(key, {"node": nws.node, "score": 0.0})
        merged[key]["score"] = float(merged[key]["score"]) + bm25_weight * (1.0 / (rrf_k + rank))

    fused = [
        NodeWithScore(node=item["node"], score=float(item["score"]))  # type: ignore[arg-type]
        for item in merged.values()
    ]
    fused.sort(key=lambda x: float(x.score or 0.0), reverse=True)
    return fused[:top_k]


def snippet(text: str, limit: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def format_source(node_with_score: NodeWithScore) -> str:
    md = node_with_score.node.metadata or {}
    file_name = str(md.get("file_name") or "unknown.md")
    file_path = str(md.get("file_path") or md.get("source") or file_name)
    return f"[{file_name}]({file_path})"


def build_messages(
    question: str,
    retrieved: Sequence[NodeWithScore],
    history_turns: Sequence[Dict[str, str]],
    rag_enabled: bool,
) -> List[Dict[str, str]]:
    blocks: List[str] = []
    for i, nws in enumerate(retrieved, start=1):
        md = nws.node.metadata or {}
        file_name = str(md.get("file_name") or "unknown.md")
        blocks.append(f"Источник {i}: {file_name}\n{nws.node.get_content()}")

    history_blocks: List[str] = []
    for i, turn in enumerate(history_turns, start=1):
        uq = str(turn.get("user", "")).strip()
        aa = str(turn.get("assistant", "")).strip()
        if not uq and not aa:
            continue
        history_blocks.append(f"Ход {i}:\nПользователь: {uq}\nАссистент: {aa}")

    if rag_enabled:
        user = (
            f"Вопрос пользователя: {question}\n\n"
            + (
                "Предыдущие ходы диалога:\n"
                f"{chr(10).join(history_blocks)}\n\n"
                if history_blocks
                else ""
            )
            +
            "Контекст документов:\n"
            f"{chr(10).join(blocks)}\n\n"
            "Правила ответа:\n"
            "1) Ответ короткий и конкретный.\n"
            "2) Только факты из контекста.\n"
            "3) Если данных недостаточно, задай один уточняющий вопрос.\n"
            "4) В конце выведи блок 'Источники:' с markdown-ссылками."
        )
    else:
        user = (
            f"Вопрос пользователя: {question}\n\n"
            + (
                "Предыдущие ходы диалога:\n"
                f"{chr(10).join(history_blocks)}\n\n"
                if history_blocks
                else ""
            )
            +
            "RAG отключен: отвечай без контекста документов.\n"
            "Правила ответа:\n"
            "1) Кратко и по делу.\n"
            "2) Если уверенности нет, прямо скажи об этом.\n"
            "3) Не выдумывай конкретные регламентные значения."
        )
    system_prompt = SYSTEM_PROMPT if rag_enabled else SYSTEM_PROMPT_NO_RAG
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def ensure_sources_block(answer: str, retrieved: Sequence[NodeWithScore], rag_enabled: bool) -> str:
    if not rag_enabled:
        return answer.strip()
    if "источники:" in answer.lower():
        return answer
    lines = [answer.strip(), "", "Источники:"]
    for nws in retrieved:
        lines.append(f"- {format_source(nws)}")
    return "\n".join(lines)


def llm_rerank(
    *,
    client: OpenAI,
    chat_model: str,
    question: str,
    candidates: Sequence[NodeWithScore],
    top_n: int,
    temperature: float,
) -> List[NodeWithScore]:
    if not candidates:
        return []
    if top_n >= len(candidates):
        return list(candidates)

    blocks: List[str] = []
    for i, nws in enumerate(candidates, start=1):
        md = nws.node.metadata or {}
        file_name = str(md.get("file_name") or "unknown.md")
        blocks.append(
            f"[{i}] {file_name}\n"
            f"Текст: {snippet(nws.node.get_content(), limit=700)}\n"
            f"Базовый score: {float(nws.score or 0.0):.6f}"
        )

    prompt = (
        "Ты ранжируешь фрагменты документа для ответа саппорт-ассистента.\n"
        f"Вопрос: {question}\n\n"
        "Кандидаты:\n"
        f"{chr(10).join(blocks)}\n\n"
        f"Выбери {top_n} наиболее релевантных кандидатов.\n"
        "Верни только JSON-массив номеров кандидатов в порядке релевантности, например: [3,1,5].\n"
        "Без пояснений."
    )
    resp = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": "Ты rerank-модуль. Возвращай только валидный JSON-массив индексов."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=220,
    )
    raw = (resp.choices[0].message.content or "").strip()

    chosen: List[int] = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            chosen = [int(x) for x in parsed if isinstance(x, int | float)]
    except Exception:  # noqa: BLE001
        nums = re.findall(r"\d+", raw)
        chosen = [int(x) for x in nums]

    selected: List[NodeWithScore] = []
    used: set[int] = set()
    for idx in chosen:
        if idx < 1 or idx > len(candidates):
            continue
        pos = idx - 1
        if pos in used:
            continue
        used.add(pos)
        selected.append(candidates[pos])
        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        for i, nws in enumerate(candidates):
            if i in used:
                continue
            selected.append(nws)
            if len(selected) >= top_n:
                break
    return selected[:top_n]


def ask_llm(
    client: OpenAI,
    model: str,
    question: str,
    retrieved: Sequence[NodeWithScore],
    history_turns: Sequence[Dict[str, str]],
    rag_enabled: bool,
    temperature: float,
    max_tokens: int,
) -> str:
    messages = build_messages(question, retrieved, history_turns, rag_enabled)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    return ensure_sources_block(text, retrieved, rag_enabled)


def run_single_question(
    *,
    client: OpenAI,
    chat_model: str,
    question: str,
    vector_retriever: Optional[BaseRetriever],
    bm25_retriever: Optional[BM25Retriever],
    top_k: int,
    vector_weight: float,
    bm25_weight: float,
    rerank_mode: str,
    rerank_model: str,
    rerank_candidates: int,
    history_turns: Sequence[Dict[str, str]],
    temperature: float,
    rerank_temperature: float,
    max_tokens: int,
    show_retrieval: bool,
) -> Tuple[str, List[NodeWithScore]]:
    rag_enabled = vector_retriever is not None and bm25_retriever is not None
    if rag_enabled:
        retrieve_top = max(top_k, rerank_candidates if rerank_mode == "llm" else top_k)
        retrieved = hybrid_retrieve(
            question,
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            top_k=retrieve_top,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )
        if rerank_mode == "llm":
            retrieved = llm_rerank(
                client=client,
                chat_model=rerank_model,
                question=question,
                candidates=retrieved,
                top_n=top_k,
                temperature=rerank_temperature,
            )
        else:
            retrieved = retrieved[:top_k]
    else:
        retrieved = []

    answer = ask_llm(
        client=client,
        model=chat_model,
        question=question,
        retrieved=retrieved,
        history_turns=history_turns,
        rag_enabled=rag_enabled,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if show_retrieval:
        pass
    return answer, list(retrieved)


def load_session(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(rec, dict):
                rows.append(
                    {
                        "user": str(rec.get("user", "")),
                        "assistant": str(rec.get("assistant", "")),
                    }
                )
    return rows


def save_session(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {"user": row.get("user", ""), "assistant": row.get("assistant", "")},
                    ensure_ascii=False,
                )
                + "\n"
            )


def init_prompt_history(path: Path) -> None:
    if readline is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            readline.read_history_file(str(path))
        except Exception:  # noqa: BLE001
            pass
    readline.set_history_length(2000)


def persist_prompt_history(path: Path) -> None:
    if readline is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        readline.write_history_file(str(path))
    except Exception:  # noqa: BLE001
        pass


def create_prompt_session(prompt_history_path: Path) -> Any:
    if PromptSession is None or FileHistory is None:
        return None
    prompt_history_path.parent.mkdir(parents=True, exist_ok=True)
    completer = WordCompleter(CHAT_COMMANDS, ignore_case=True, sentence=True) if WordCompleter else None
    auto_suggest = AutoSuggestFromHistory() if AutoSuggestFromHistory else None
    return PromptSession(history=FileHistory(str(prompt_history_path)), completer=completer, auto_suggest=auto_suggest)


def safe_stdin_input(prompt: str) -> str:
    sys.stdout.write(prompt)
    sys.stdout.flush()

    if not hasattr(sys.stdin, "buffer"):
        return input("").strip()

    raw = sys.stdin.buffer.readline()
    if not raw:
        return ""

    encodings: List[str] = []
    if sys.stdin.encoding:
        encodings.append(sys.stdin.encoding)
    preferred = locale.getpreferredencoding(False)
    if preferred:
        encodings.append(preferred)
    encodings.extend(["utf-8", "cp1251"])

    tried: set[str] = set()
    for enc in encodings:
        key = enc.lower()
        if key in tried:
            continue
        tried.add(key)
        try:
            return raw.decode(enc).strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").strip()


def render_chat_output(
    console: Any,
    answer: str,
    retrieved: Sequence[NodeWithScore],
    show_retrieval: bool,
    models_info: Dict[str, str],
) -> None:
    models_line = (
        f"Модели: chat={models_info.get('chat', '-')}, "
        f"retrieval={models_info.get('retrieval', '-')}, "
        f"rerank={models_info.get('rerank', '-')}"
    )
    if console is None or Panel is None or Table is None:
        print(models_line)
        print(f"\nБот> {answer}\n")
        if show_retrieval:
            print("[retrieval]")
            if not retrieved:
                print("RAG отключен: retrieval-пул пуст.")
            else:
                for i, nws in enumerate(retrieved, start=1):
                    print(f"{i}. score={float(nws.score or 0.0):.4f} {format_source(nws)}")
                    print(f"   {snippet(nws.node.get_content())}")
        return

    console.print(f"[bold] {models_line}")
    console.print(Panel(answer, title="Бот", border_style="cyan"))
    if not show_retrieval:
        return
    table = Table(title="Источники (retrieval)")
    table.add_column("#", justify="right", style="bold")
    table.add_column("Источник")
    table.add_column("Score", justify="right")
    table.add_column("Preview")
    if not retrieved:
        table.add_row("-", "RAG отключен", "-", "-")
    else:
        for i, nws in enumerate(retrieved, start=1):
            table.add_row(
                str(i),
                format_source(nws),
                f"{float(nws.score or 0.0):.4f}",
                snippet(nws.node.get_content(), limit=140),
            )
    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Инференс с hybrid retrieval (BM25 + vector) по сохраненному LlamaIndex."
    )
    parser.add_argument("--index-dir", type=Path, default=None, help="Папка persist-dir с индексом LlamaIndex")
    parser.add_argument("--question", default=None, help="Одиночный вопрос (если не указан, запускается интерактивный режим)")
    parser.add_argument("--disable-rag", action="store_true", help="Отключить retrieval и отвечать напрямую моделью")

    parser.add_argument("--base-url", default="http://localhost:1234/v1", help="OpenAI-совместимый URL")
    parser.add_argument("--api-key", default="lm-studio", help="API key (для LM Studio обычно любое значение)")
    parser.add_argument("--chat-model", default="local-model", help="Модель для ответа")
    parser.add_argument(
        "--embed-model",
        default=None,
        help="Embedding модель для retrieval. По умолчанию берется из index_manifest.json",
    )
    parser.add_argument(
        "--retrieval-model",
        default=None,
        help="Явная retrieval embedding модель (приоритетнее --embed-model и manifest)",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="Таймаут API-запроса (сек)")

    parser.add_argument("--top-k", type=int, default=8, help="Сколько итоговых чанков передавать в LLM")
    parser.add_argument("--vector-top-k", type=int, default=16, help="Сколько брать кандидатов из vector retriever")
    parser.add_argument("--bm25-top-k", type=int, default=16, help="Сколько брать кандидатов из BM25 retriever")
    parser.add_argument("--vector-weight", type=float, default=1.0, help="Вес vector-ретривера в fusion")
    parser.add_argument("--bm25-weight", type=float, default=1.0, help="Вес BM25-ретривера в fusion")
    parser.add_argument("--bm25-language", default="russian", help="Язык BM25 stemmer (например: russian, english)")

    parser.add_argument("--temperature", type=float, default=0.1, help="Температура генерации ответа")
    parser.add_argument(
        "--rerank",
        choices=["off", "llm"],
        default="off",
        help="Режим rerank после hybrid retrieval",
    )
    parser.add_argument(
        "--rerank-model",
        default=None,
        help="Модель для rerank (по умолчанию равна --chat-model)",
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=24,
        help="Размер пула кандидатов до rerank (используется при --rerank llm)",
    )
    parser.add_argument("--rerank-temperature", type=float, default=0.0, help="Температура LLM-rerank")
    parser.add_argument("--max-tokens", type=int, default=900, help="Максимум токенов ответа")
    parser.add_argument("--show-retrieval", action="store_true", help="Печатать выбранные чанки retrieval")
    parser.set_defaults(load_session=True, save_session=True)
    parser.add_argument("--load-session", dest="load_session", action="store_true", help="Загружать память сессии")
    parser.add_argument("--no-load-session", dest="load_session", action="store_false", help="Не загружать память сессии")
    parser.add_argument("--save-session", dest="save_session", action="store_true", help="Сохранять память сессии")
    parser.add_argument("--no-save-session", dest="save_session", action="store_false", help="Не сохранять память сессии")
    parser.add_argument("--memory-turns", type=int, default=4, help="Сколько последних ходов включать в prompt")
    parser.add_argument(
        "--session-file",
        default="data/trade_support_rag/chat_session.jsonl",
        help="Файл памяти диалога для /save и автосохранения",
    )
    parser.add_argument(
        "--prompt-history-file",
        default="data/trade_support_rag/chat_prompt_history.txt",
        help="Файл истории prompt_toolkit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k должен быть > 0")
    if args.vector_top_k <= 0:
        raise ValueError("--vector-top-k должен быть > 0")
    if args.bm25_top_k <= 0:
        raise ValueError("--bm25-top-k должен быть > 0")
    if args.vector_weight < 0 or args.bm25_weight < 0:
        raise ValueError("--vector-weight и --bm25-weight должны быть >= 0")
    if args.vector_weight == 0 and args.bm25_weight == 0:
        raise ValueError("Хотя бы один из весов должен быть > 0")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens должен быть > 0")
    if args.memory_turns < 0:
        raise ValueError("--memory-turns должен быть >= 0")
    if args.rerank_candidates <= 0:
        raise ValueError("--rerank-candidates должен быть > 0")
    if not (0.0 <= args.rerank_temperature <= 1.0):
        raise ValueError("--rerank-temperature должен быть в диапазоне [0.0, 1.0]")

    vector_retriever: Optional[BaseRetriever] = None
    bm25_retriever: Optional[BM25Retriever] = None
    retrieval_model_name = "off"
    rerank_model_name = "off"
    rerank_model = args.rerank_model or args.chat_model
    if args.disable_rag:
        print("[init] RAG disabled: retrieval/index loading skipped")
    else:
        if args.index_dir is None:
            raise ValueError("--index-dir обязателен, если не указан --disable-rag")
        if not args.index_dir.exists():
            raise FileNotFoundError(f"index-dir не найден: {args.index_dir}")
        manifest = load_manifest(args.index_dir / "index_manifest.json")
        embed_model = args.retrieval_model or args.embed_model or manifest.get("embed_model") or "text-embedding-3-small"
        retrieval_model_name = embed_model
        rerank_model_name = rerank_model if args.rerank == "llm" else "off"
        print(f"[init] index-dir: {args.index_dir}")
        print(f"[init] embed-model: {embed_model}")
        print(f"[init] rerank: {args.rerank}")
        if args.rerank == "llm":
            print(f"[init] rerank-model: {rerank_model}")
        print("[init] loading retrievers")
        vector_retriever, bm25_retriever = build_retrievers(
            index_dir=args.index_dir,
            embed_model_name=embed_model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=args.timeout,
            vector_top_k=args.vector_top_k,
            bm25_top_k=args.bm25_top_k,
            bm25_language=args.bm25_language,
        )
        print("[init] retrievers ready")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    if args.question:
        answer, retrieved = run_single_question(
            client=client,
            chat_model=args.chat_model,
            question=args.question,
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            top_k=args.top_k,
            vector_weight=args.vector_weight,
            bm25_weight=args.bm25_weight,
            rerank_mode=args.rerank,
            rerank_model=rerank_model,
            rerank_candidates=args.rerank_candidates,
            history_turns=[],
            temperature=args.temperature,
            rerank_temperature=args.rerank_temperature,
            max_tokens=args.max_tokens,
            show_retrieval=args.show_retrieval,
        )
        render_chat_output(
            None,
            answer,
            retrieved,
            args.show_retrieval,
            {
                "chat": args.chat_model,
                "retrieval": retrieval_model_name,
                "rerank": rerank_model_name,
            },
        )
        return

    console = Console() if Console else None
    prompt_history_path = Path(args.prompt_history_file)
    session_path = Path(args.session_file)
    init_prompt_history(prompt_history_path)
    pt_session = create_prompt_session(prompt_history_path)
    session_rows = load_session(session_path) if args.load_session else []
    if console:
        console.print("[chat] interactive mode. Для выхода: /exit")
        console.print("[chat] команды: /reset (очистить память), /save (сохранить память)")
        if pt_session is None:
            console.print("[chat] prompt_toolkit не найден, используется стандартный ввод.")
    else:
        print("[chat] interactive mode. Для выхода: /exit")
        print("[chat] команды: /reset (очистить память), /save (сохранить память)")

    while True:
        try:
            if pt_session is not None:
                q = pt_session.prompt("Вы> ").strip()
            else:
                q = safe_stdin_input("Вы> ")
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not q:
            continue

        lowered = q.lower()
        if lowered in {"/exit", "exit", "quit"}:
            if args.save_session:
                save_session(session_path, session_rows)
            persist_prompt_history(prompt_history_path)
            if console:
                console.print(f"[chat] session saved: {session_path}")
            else:
                print(f"[chat] session saved: {session_path}")
            break

        if lowered == "/reset":
            session_rows = []
            if args.save_session:
                save_session(session_path, session_rows)
            if console:
                console.print("[chat] session memory cleared")
            else:
                print("[chat] session memory cleared")
            continue

        if lowered == "/save":
            save_session(session_path, session_rows)
            if console:
                console.print(f"[chat] session saved: {session_path}")
            else:
                print(f"[chat] session saved: {session_path}")
            continue

        history_slice = session_rows[-args.memory_turns :] if args.memory_turns > 0 else []
        answer, retrieved = run_single_question(
            client=client,
            chat_model=args.chat_model,
            question=q,
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            top_k=args.top_k,
            vector_weight=args.vector_weight,
            bm25_weight=args.bm25_weight,
            rerank_mode=args.rerank,
            rerank_model=rerank_model,
            rerank_candidates=args.rerank_candidates,
            history_turns=history_slice,
            temperature=args.temperature,
            rerank_temperature=args.rerank_temperature,
            max_tokens=args.max_tokens,
            show_retrieval=args.show_retrieval,
        )
        session_rows.append({"user": q, "assistant": answer})
        if args.save_session:
            save_session(session_path, session_rows)
        render_chat_output(
            console,
            answer,
            retrieved,
            args.show_retrieval,
            {
                "chat": args.chat_model,
                "retrieval": retrieval_model_name,
                "rerank": rerank_model_name,
            },
        )
        print()


if __name__ == "__main__":
    main()

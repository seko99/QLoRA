#!/usr/bin/env python3
import argparse
import json
import locale
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from openai import OpenAI

try:
    import faiss  # type: ignore
except Exception:  # noqa: BLE001
    faiss = None

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


DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"
DEFAULT_MODEL = "model-identifier"
CHAT_COMMANDS = ["/exit", "/reset", "/save"]


@dataclass
class Chunk:
    chunk_id: int
    file_name: str
    file_path: str
    section: str
    anchor: str
    text: str


class LMStudioClient:
    def __init__(self, base_url: str, api_key: str, timeout: float):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def embed(self, model: str, texts: Sequence[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=model, input=list(texts))
        return [item.embedding for item in response.data]

    def chat(self, model: str, messages: Sequence[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return (content or "").strip()


def ensure_faiss() -> None:
    if faiss is None:
        raise RuntimeError("Не найден модуль faiss. Установите faiss-cpu или faiss-gpu.")


def slugify(text: str) -> str:
    txt = text.lower().strip()
    txt = re.sub(r"[^a-zа-я0-9\s-]", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\s+", "-", txt)
    return txt.strip("-") or "section"


def split_with_overlap(text: str, max_chars: int, overlap: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]
        if end < len(text):
            split_idx = chunk.rfind("\n\n")
            if split_idx > max_chars // 2:
                end = start + split_idx
                chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]


def parse_markdown_chunks(md_text: str, file_name: str, file_path: str, chunk_size: int, overlap: int) -> List[Chunk]:
    sections: List[Tuple[str, List[str]]] = []
    current_title = "Документ"
    current_lines: List[str] = []

    for line in md_text.splitlines():
        if line.startswith("#"):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line.lstrip("#").strip() or "Раздел"
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))

    chunks: List[Chunk] = []
    cid = 0
    for section, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        for part in split_with_overlap(body, max_chars=chunk_size, overlap=overlap):
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    file_name=file_name,
                    file_path=file_path,
                    section=section,
                    anchor=slugify(section),
                    text=part,
                )
            )
            cid += 1
    return chunks


def build_chunks(docs_dir: Path, chunk_size: int, overlap: int) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    chunk_id = 0
    paths = sorted(docs_dir.glob("*.md"))
    if not paths:
        raise RuntimeError(f"В папке {docs_dir} не найдено markdown-документов")

    for path in paths:
        raw = path.read_text(encoding="utf-8")
        per_file = parse_markdown_chunks(
            raw,
            file_name=path.name,
            file_path=str(path),
            chunk_size=chunk_size,
            overlap=overlap,
        )
        for chunk in per_file:
            chunk.chunk_id = chunk_id
            all_chunks.append(chunk)
            chunk_id += 1
    return all_chunks


def batched(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def to_float32_matrix(vectors: Sequence[Sequence[float]]) -> "Any":
    import numpy as np

    mat = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat /= norms
    return mat


def save_metadata(metadata_path: Path, chunks: Sequence[Chunk]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            rec = {
                "chunk_id": ch.chunk_id,
                "file_name": ch.file_name,
                "file_path": ch.file_path,
                "section": ch.section,
                "anchor": ch.anchor,
                "text": ch.text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_metadata(metadata_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def cosine_overlap_score(answer: str, contexts: Sequence[str]) -> float:
    answer_tokens = set(re.findall(r"[a-zа-я0-9]{3,}", answer.lower(), flags=re.IGNORECASE))
    if not answer_tokens:
        return 0.0
    context_tokens: set[str] = set()
    for text in contexts:
        context_tokens.update(re.findall(r"[a-zа-я0-9]{3,}", text.lower(), flags=re.IGNORECASE))
    if not context_tokens:
        return 0.0
    inter = len(answer_tokens.intersection(context_tokens))
    return inter / max(1, len(answer_tokens))


def format_source_link(item: Dict[str, Any]) -> str:
    file_path = item["file_path"]
    label = f"{item['file_name']}#{item['section']}"
    return f"[{label}]({file_path}#{item['anchor']})"


def preview_text(text: str, limit: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "..."


def retrieve(
    client: LMStudioClient,
    index: Any,
    metadata: Sequence[Dict[str, Any]],
    embed_model: str,
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    import numpy as np

    q_vec = client.embed(embed_model, [query])[0]
    q_mat = np.asarray([q_vec], dtype="float32")
    norms = np.linalg.norm(q_mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    q_mat /= norms

    pool_k = min(max(top_k * 4, top_k), len(metadata))
    scores, ids = index.search(q_mat, pool_k)
    found: List[Dict[str, Any]] = []
    q_tokens = set(re.findall(r"[a-zа-я0-9]{3,}", query.lower(), flags=re.IGNORECASE))
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = dict(metadata[idx])
        text = f"{item.get('section', '')} {item.get('text', '')}".lower()
        c_tokens = set(re.findall(r"[a-zа-я0-9]{3,}", text, flags=re.IGNORECASE))
        overlap = len(q_tokens.intersection(c_tokens)) / max(1, len(q_tokens))

        section_bonus = 0.0
        section_name = str(item.get("section", "")).lower()
        if any(key in section_name for key in ["требован", "процесс", "параметр", "контроль"]):
            section_bonus = 0.03

        hybrid_score = 0.78 * float(score) + 0.19 * overlap + section_bonus
        item["score"] = float(score)
        item["hybrid_score"] = float(hybrid_score)
        found.append(item)
    found.sort(key=lambda x: x.get("hybrid_score", x.get("score", 0.0)), reverse=True)
    return found[:top_k]


def build_prompt(question: str, retrieved: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    return build_prompt_with_memory(question, retrieved, history_turns=[])


def build_prompt_with_memory(
    question: str,
    retrieved: Sequence[Dict[str, Any]],
    history_turns: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    context_blocks = []
    for i, item in enumerate(retrieved, start=1):
        context_blocks.append(
            f"Источник {i}: {item['file_name']} | Раздел: {item['section']}\n"
            f"Текст:\n{item['text']}"
        )

    history_blocks: List[str] = []
    for i, turn in enumerate(history_turns, start=1):
        uq = str(turn.get("user", "")).strip()
        aa = str(turn.get("assistant", "")).strip()
        if not uq and not aa:
            continue
        history_blocks.append(f"Ход {i}:\nПользователь: {uq}\nАссистент: {aa}")

    system = (
        "Ты производственный ассистент. Отвечай только по предоставленному контексту документов. "
        "Если в контексте нет ответа, явно сообщи об этом без выдумок. "
        "Если найдено несколько разных значений в разных документах, не отказывай: перечисли значения по каждому источнику, "
        "отметь расхождение и попроси пользователя уточнить документ (код/версию), по которому нужен ответ. "
        "В конце обязательно добавь блок 'Источники:' со ссылками на использованные документы в формате markdown."
    )
    user = (
        f"Вопрос: {question}\n\n"
        + (
            "Предыдущие ходы диалога:\n"
            f"{chr(10).join(history_blocks)}\n\n"
            if history_blocks
            else ""
        )
        +
        "Контекст:\n"
        f"{'\n\n'.join(context_blocks)}\n\n"
        "Требования к ответу:\n"
        "1) Краткий точный ответ только по контексту.\n"
        "2) Если в контексте есть нужные числа/факты, обязательно дай ответ и укажи их.\n"
        "3) Если есть противоречия между документами, перечисли варианты по источникам и задай 1 уточняющий вопрос.\n"
        "4) Если данных недостаточно, ответи в свободной форме с смыслом отказа (нет данных в документах).\n"
        "5) После ответа выведи 'Источники:' и список ссылок markdown на файлы и разделы, которые использовал."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def ensure_sources_block(answer: str, retrieved: Sequence[Dict[str, Any]]) -> str:
    if "источники:" in answer.lower():
        return answer
    lines = [answer.rstrip(), "", "Источники:"]
    for item in retrieved:
        lines.append(f"- {format_source_link(item)}")
    return "\n".join(lines).strip()


def cmd_index(args: argparse.Namespace) -> None:
    ensure_faiss()

    docs_dir = Path(args.docs_dir)
    out_dir = Path(args.index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = LMStudioClient(args.base_url, args.api_key, args.timeout)

    print(f"[index] loading docs from {docs_dir}")
    chunks = build_chunks(docs_dir, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
    print(f"[index] built {len(chunks)} chunks")

    texts = [c.text for c in chunks]
    vectors: List[List[float]] = []
    for batch in batched(texts, args.embed_batch_size):
        vectors.extend(client.embed(args.embed_model, batch))
        print(f"[index] embedded {len(vectors)}/{len(texts)}")

    mat = to_float32_matrix(vectors)
    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(mat)

    faiss.write_index(index, str(out_dir / "index.faiss"))
    save_metadata(out_dir / "metadata.jsonl", chunks)

    config = {
        "docs_dir": str(docs_dir),
        "embed_model": args.embed_model,
        "vector_dim": int(dim),
        "chunks": len(chunks),
    }
    (out_dir / "index_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[done] index saved in {out_dir}")


def load_index_bundle(index_dir: Path) -> Tuple[Any, List[Dict[str, Any]]]:
    ensure_faiss()
    index_path = index_dir / "index.faiss"
    meta_path = index_dir / "metadata.jsonl"
    if not index_path.exists() or not meta_path.exists():
        raise RuntimeError(f"Не найден индекс в {index_dir}. Сначала запустите команду index.")
    index = faiss.read_index(str(index_path))
    metadata = load_metadata(meta_path)
    return index, metadata


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


def create_chat_prompt_session(prompt_history_path: Path) -> Any:
    if PromptSession is None or FileHistory is None:
        return None
    prompt_history_path.parent.mkdir(parents=True, exist_ok=True)
    completer = WordCompleter(CHAT_COMMANDS, ignore_case=True, sentence=True) if WordCompleter else None
    auto_suggest = AutoSuggestFromHistory() if AutoSuggestFromHistory else None
    return PromptSession(history=FileHistory(str(prompt_history_path)), completer=completer, auto_suggest=auto_suggest)


def render_chat_output(console: Any, answer: str, retrieved: Sequence[Dict[str, Any]], debug: bool) -> None:
    if console is None or Panel is None or Table is None:
        print(f"\nБот> {answer}\n")
        if debug:
            print("[debug] top-k retrieval:")
            for i, item in enumerate(retrieved, start=1):
                print(
                    f"{i}. score={item['score']:.4f} hybrid={item.get('hybrid_score', item['score']):.4f} "
                    f"| {item['file_name']}#{item['section']}\n"
                    f"   {preview_text(item['text'])}"
                )
        print("Источники (retrieval):")
        for item in retrieved:
            print(
                f"- {format_source_link(item)} "
                f"(score={item['score']:.3f}, hybrid={item.get('hybrid_score', item['score']):.3f})"
            )
        return

    console.print(Panel(answer, title="Бот", border_style="cyan"))
    table = Table(title="Источники (retrieval)")
    table.add_column("#", justify="right", style="bold")
    table.add_column("Источник")
    table.add_column("Score", justify="right")
    table.add_column("Hybrid", justify="right")
    for i, item in enumerate(retrieved, start=1):
        table.add_row(
            str(i),
            format_source_link(item),
            f"{item['score']:.3f}",
            f"{item.get('hybrid_score', item['score']):.3f}",
        )
    console.print(table)

    if debug:
        debug_table = Table(title="Debug top-k chunks")
        debug_table.add_column("#", justify="right", style="bold")
        debug_table.add_column("File#Section")
        debug_table.add_column("Preview")
        for i, item in enumerate(retrieved, start=1):
            debug_table.add_row(
                str(i),
                f"{item['file_name']}#{item['section']}",
                preview_text(item["text"], limit=180),
            )
        console.print(debug_table)


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


def cmd_chat(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir)
    index, metadata = load_index_bundle(index_dir)
    client = LMStudioClient(args.base_url, args.api_key, args.timeout)
    prompt_history_path = Path(args.prompt_history_file)
    session_path = Path(args.session_file)
    console = Console() if Console else None

    init_prompt_history(prompt_history_path)
    pt_session = create_chat_prompt_session(prompt_history_path)
    session_rows = load_session(session_path) if args.load_session else []
    if session_rows:
        msg = f"[chat] loaded session turns: {len(session_rows)} from {session_path}"
        if console:
            console.print(msg)
        else:
            print(msg)
    if console:
        console.print("[chat] interactive mode. Для выхода: /exit")
        console.print("[chat] команды: /reset (очистить память сессии), /save (сохранить сессию)")
        if pt_session is None:
            console.print("[chat] prompt_toolkit не найден, используется стандартный ввод.")
    else:
        print("[chat] interactive mode. Для выхода: /exit")
        print("[chat] команды: /reset (очистить память сессии), /save (сохранить сессию)")
    while True:
        if pt_session is not None:
            try:
                question = pt_session.prompt("Вы> ").strip()
            except (EOFError, KeyboardInterrupt):
                question = "/exit"
        else:
            try:
                question = input("\nВы> ").strip()
            except UnicodeDecodeError:
                question = safe_stdin_input("\nВы> ").strip()
            except (EOFError, KeyboardInterrupt):
                question = "/exit"
        if not question:
            continue
        if readline is not None and pt_session is None:
            readline.add_history(question)
        if question in {"/exit", "exit", "quit"}:
            if args.save_session:
                save_session(session_path, session_rows)
            persist_prompt_history(prompt_history_path)
            if console:
                console.print("[chat] bye")
            else:
                print("[chat] bye")
            break
        if question == "/reset":
            session_rows = []
            if args.save_session:
                save_session(session_path, session_rows)
            if console:
                console.print("[chat] session memory cleared")
            else:
                print("[chat] session memory cleared")
            continue
        if question == "/save":
            save_session(session_path, session_rows)
            persist_prompt_history(prompt_history_path)
            if console:
                console.print(f"[chat] session saved: {session_path}")
            else:
                print(f"[chat] session saved: {session_path}")
            continue

        retrieved = retrieve(client, index, metadata, args.embed_model, question, args.top_k)
        history_for_prompt = session_rows[-args.history_turns :] if args.history_turns > 0 else []
        messages = build_prompt_with_memory(question, retrieved, history_for_prompt)
        answer = client.chat(args.chat_model, messages, temperature=args.temperature, max_tokens=args.max_tokens)
        answer = normalize_conflict_answer(answer, retrieved)
        answer = ensure_sources_block(answer, retrieved)
        session_rows.append({"user": question, "assistant": answer})
        if args.save_session:
            save_session(session_path, session_rows)
        persist_prompt_history(prompt_history_path)
        render_chat_output(console, answer, retrieved, args.debug)


def detect_refusal(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "не могу ответить",
        "нет информации",
        "недостаточно данных",
        "не содерж",
        "документах нет",
    ]
    return any(m in lowered for m in markers)


def extract_temperature_facts(retrieved: Sequence[Dict[str, Any]], max_docs: int = 3) -> List[str]:
    facts: List[str] = []
    seen_docs: set[str] = set()
    pattern = re.compile(r"\d+(?:[.,]\d+)?\s*(?:±\s*\d+(?:[.,]\d+)?\s*)?°\s*C", re.IGNORECASE)
    for item in retrieved:
        doc = str(item.get("file_name", ""))
        if not doc or doc in seen_docs:
            continue
        matches = pattern.findall(str(item.get("text", "")))
        if not matches:
            continue
        seen_docs.add(doc)
        uniq: List[str] = []
        for m in matches:
            norm = re.sub(r"\s+", " ", m).strip()
            if norm not in uniq:
                uniq.append(norm)
            if len(uniq) >= 2:
                break
        facts.append(f"- {doc}: {', '.join(uniq)}")
        if len(facts) >= max_docs:
            break
    return facts


def normalize_conflict_answer(answer: str, retrieved: Sequence[Dict[str, Any]]) -> str:
    lowered = answer.lower()
    has_refusal = detect_refusal(answer)
    has_conflict = any(x in lowered for x in ["разн", "противореч", "не единого", "расхожд"])
    if not (has_refusal and has_conflict):
        return answer

    text = answer.strip()
    text = re.sub(r"(?i)^не могу ответить:\s*", "", text, count=1).strip()
    text = re.sub(r"(?i)^не могу ответить\s*", "", text, count=1).strip()
    text = re.sub(r"(?i)^в предоставленных документах нет[^.]*\.?\s*", "", text, count=1).strip()
    text = f"По документам обнаружено расхождение значений.\n\n{text}".strip()

    has_temp = re.search(r"\d+(?:[.,]\d+)?\s*(?:±\s*\d+(?:[.,]\d+)?\s*)?°\s*C", text, flags=re.IGNORECASE)
    if not has_temp:
        facts = extract_temperature_facts(retrieved)
        if facts:
            text += "\n\nНайденные значения в извлеченных источниках:\n" + "\n".join(facts)

    if "уточните" not in text.lower() and "какой документ" not in text.lower():
        text += (
            "\n\nУточните, пожалуйста, какой документ (код/версия) считать приоритетным, "
            "и я дам точный ответ по нему."
        )
    return text


def extract_eval_samples(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if 0 < limit <= len(rows):
                break
    return rows


def load_questions(path: Path, limit: int) -> List[str]:
    questions: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if not q:
                continue
            questions.append(q)
            if 0 < limit <= len(questions):
                break
    return questions


def cmd_diagnose(args: argparse.Namespace) -> None:
    index, metadata = load_index_bundle(Path(args.index_dir))
    client = LMStudioClient(args.base_url, args.api_key, args.timeout)

    questions: List[str] = []
    if args.question:
        questions.append(args.question)
    if args.questions_file:
        questions.extend(load_questions(Path(args.questions_file), args.max_samples))

    if not questions:
        raise RuntimeError("Укажите --question или --questions-file")

    out_rows: List[Dict[str, Any]] = []
    for i, question in enumerate(questions, start=1):
        retrieved = retrieve(client, index, metadata, args.embed_model, question, args.top_k)
        top_scores = [x["score"] for x in retrieved]
        avg_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
        max_score = max(top_scores) if top_scores else 0.0
        min_score = min(top_scores) if top_scores else 0.0

        print(f"\n[diagnose] Q{i}: {question}")
        print(f"score_max={max_score:.4f} score_avg={avg_score:.4f} score_min={min_score:.4f}")
        for rank, item in enumerate(retrieved, start=1):
            print(
                f"  {rank}. score={item['score']:.4f} hybrid={item.get('hybrid_score', item['score']):.4f} "
                f"| {item['file_name']}#{item['section']}\n"
                f"     {preview_text(item['text'], limit=180)}"
            )

        out_rows.append(
            {
                "question": question,
                "score_max": round(max_score, 6),
                "score_avg": round(avg_score, 6),
                "score_min": round(min_score, 6),
                "top_k": [
                    {
                        "rank": rank,
                        "score": round(item["score"], 6),
                        "hybrid_score": round(item.get("hybrid_score", item["score"]), 6),
                        "file_name": item["file_name"],
                        "section": item["section"],
                        "anchor": item["anchor"],
                        "text_preview": preview_text(item["text"]),
                    }
                    for rank, item in enumerate(retrieved, start=1)
                ],
            }
        )

    if args.report_out:
        out_path = Path(args.report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n[diagnose] saved: {out_path}")


def cmd_eval(args: argparse.Namespace) -> None:
    index, metadata = load_index_bundle(Path(args.index_dir))
    client = LMStudioClient(args.base_url, args.api_key, args.timeout)

    samples = extract_eval_samples(Path(args.eval_path), args.max_samples)
    if not samples:
        raise RuntimeError("Eval dataset пуст")

    known_total = 0
    known_hit = 0
    unknown_total = 0
    unknown_refusal_ok = 0
    grounded_scores: List[float] = []

    for i, sample in enumerate(samples, start=1):
        messages = sample.get("messages", [])
        user_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        if not user_msg:
            continue

        stype = sample.get("meta", {}).get("sample_type", "known")
        expected_doc = sample.get("meta", {}).get("source_doc")

        retrieved = retrieve(client, index, metadata, args.embed_model, user_msg, args.top_k)
        reply = client.chat(
            args.chat_model,
            build_prompt(user_msg, retrieved),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        grounded_scores.append(cosine_overlap_score(reply, [x["text"] for x in retrieved]))

        if stype == "known":
            known_total += 1
            if expected_doc and any(x["file_name"] == expected_doc for x in retrieved):
                known_hit += 1
        else:
            unknown_total += 1
            if detect_refusal(reply):
                unknown_refusal_ok += 1

        if i % 10 == 0:
            print(f"[eval] processed {i}/{len(samples)}")

    hit_k = known_hit / known_total if known_total else 0.0
    refusal_acc = unknown_refusal_ok / unknown_total if unknown_total else 0.0
    grounded_avg = sum(grounded_scores) / len(grounded_scores) if grounded_scores else 0.0

    report = {
        "samples_total": len(samples),
        "known_total": known_total,
        "unknown_total": unknown_total,
        "hit_at_k": round(hit_k, 4),
        "refusal_accuracy": round(refusal_acc, 4),
        "groundedness_avg": round(grounded_avg, 4),
        "top_k": args.top_k,
    }

    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("[eval] report")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[eval] saved: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI RAG-бот на локальном LM Studio + FAISS")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="URL OpenAI-совместимого API LM Studio")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key для LM Studio")
    parser.add_argument("--timeout", type=float, default=120.0, help="Таймаут API-запроса, сек")
    parser.add_argument("--chat-model", default=DEFAULT_MODEL, help="Модель чата (дообученная)")
    parser.add_argument("--embed-model", default=DEFAULT_MODEL, help="Модель embeddings")

    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Построить FAISS-индекс по markdown-документам")
    p_index.add_argument("--docs-dir", default="data/factory_rag/documents", help="Папка с markdown-документами")
    p_index.add_argument("--index-dir", default="data/factory_rag/index", help="Папка для индекса")
    p_index.add_argument("--chunk-size", type=int, default=1400, help="Размер чанка в символах")
    p_index.add_argument("--chunk-overlap", type=int, default=200, help="Перекрытие чанков")
    p_index.add_argument("--embed-batch-size", type=int, default=16, help="Батч для embedding-запроса")
    p_index.set_defaults(func=cmd_index)

    p_chat = sub.add_parser("chat", help="Интерактивный CLI-чат по RAG")
    p_chat.add_argument("--index-dir", default="data/factory_rag/index", help="Папка с индексом")
    p_chat.add_argument("--top-k", type=int, default=5, help="Количество извлекаемых чанков")
    p_chat.add_argument("--temperature", type=float, default=0.1, help="Температура генерации")
    p_chat.add_argument("--max-tokens", type=int, default=700, help="Максимум токенов ответа")
    p_chat.add_argument("--debug", action="store_true", help="Показать score и превью извлеченных чанков")
    p_chat.add_argument(
        "--session-file",
        default="data/factory_rag/chat_session.jsonl",
        help="Файл памяти сессии (jsonl: user/assistant)",
    )
    p_chat.add_argument(
        "--history-turns",
        type=int,
        default=6,
        help="Сколько последних ходов сессии включать в prompt",
    )
    p_chat.add_argument(
        "--prompt-history-file",
        default="data/factory_rag/.prompt_history",
        help="Файл истории ввода для стрелок вверх/вниз",
    )
    p_chat.set_defaults(load_session=True, save_session=True)
    p_chat.add_argument("--load-session", dest="load_session", action="store_true", help="Загружать память сессии при старте")
    p_chat.add_argument("--no-load-session", dest="load_session", action="store_false", help="Не загружать память сессии")
    p_chat.add_argument("--save-session", dest="save_session", action="store_true", help="Сохранять память сессии после каждого ответа")
    p_chat.add_argument("--no-save-session", dest="save_session", action="store_false", help="Не сохранять память сессии")
    p_chat.set_defaults(func=cmd_chat)

    p_diag = sub.add_parser("diagnose", help="Диагностика retrieval-качества без генерации ответа")
    p_diag.add_argument("--index-dir", default="data/factory_rag/index", help="Папка с индексом")
    p_diag.add_argument("--question", default="", help="Один вопрос для диагностики")
    p_diag.add_argument("--questions-file", default="", help="Файл с вопросами (по одному на строку)")
    p_diag.add_argument("--max-samples", type=int, default=50, help="Лимит вопросов из --questions-file")
    p_diag.add_argument("--top-k", type=int, default=5, help="Количество извлекаемых чанков")
    p_diag.add_argument(
        "--report-out",
        default="data/factory_rag/retrieval_diagnose.json",
        help="Куда сохранить JSON-отчет (пусто - не сохранять)",
    )
    p_diag.set_defaults(func=cmd_diagnose)

    p_eval = sub.add_parser("eval", help="Оценка hit@k, refusal accuracy и groundedness")
    p_eval.add_argument("--index-dir", default="data/factory_rag/index", help="Папка с индексом")
    p_eval.add_argument("--eval-path", default="data/factory_rag/training_dataset.jsonl", help="JSONL eval-набор")
    p_eval.add_argument("--max-samples", type=int, default=120, help="Лимит сэмплов для оценки")
    p_eval.add_argument("--top-k", type=int, default=5, help="Количество извлекаемых чанков")
    p_eval.add_argument("--temperature", type=float, default=0.1, help="Температура генерации")
    p_eval.add_argument("--max-tokens", type=int, default=500, help="Максимум токенов ответа")
    p_eval.add_argument("--report-out", default="data/factory_rag/rag_eval_report.json", help="Куда сохранить отчет")
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

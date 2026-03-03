#!/usr/bin/env python3
import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from openai import OpenAI

DOC_TYPES = [
    "Регламент",
    "Норматив",
    "Технологическая карта",
    "Инструкция",
    "Стандарт операции",
]

PROCESSES = [
    "термообработка сплава КТ-47",
    "контроль чистоты технологического воздуха",
    "подготовка пресс-форм серии ПФ-9",
    "входной контроль реагента Р-12",
    "калибровка дозатора линии L-3",
    "обслуживание узла вакуумной сушки",
    "контроль герметичности контейнера ТК-11",
    "маркировка партии полуфабрикатов",
    "упаковка модулей серии А-104",
    "утилизация отходов класса Т2",
    "санитарная подготовка участка смешения",
    "приемка сырья по спецификации SX-88",
]

SYSTEM_DOC = (
    "Ты технический писатель вымышленного производственного предприятия. "
    "Пиши реалистично, но все названия и коды должны быть вымышленными. "
    "Верни только Markdown документ на русском языке."
)

SYSTEM_QA = (
    "Ты готовишь датасет для SFT модели в RAG-сценарии. "
    "Вопрос и ответ должны опираться только на входной документ. "
    "Не добавляй факты, которых нет в документе."
)

DATASET_SYSTEM_PROMPT = (
    "Ты ассистент предприятия. Отвечай только на основе предоставленных документов. "
    "Если информации в документах нет или она неоднозначна, отвечай: "
    "'Не могу ответить: в предоставленных документах нет этой информации.' "
    "Не выдумывай детали и не используй внешние знания."
)

UNKNOWN_QUESTION_TEMPLATES = [
    "Какой размер годового бонуса для руководителя смены?",
    "Какая формула расчета экспортной пошлины на продукцию?",
    "Какой пароль от Wi-Fi в административном корпусе?",
    "Какая политика компании по удаленной работе инженеров?",
    "Какой план маркетинговой кампании на следующий квартал?",
    "Сколько стоит сервисное обслуживание станка у внешнего подрядчика?",
    "Кто утверждает бюджет на корпоративные мероприятия?",
    "Какие условия ДМС для сотрудников отдела логистики?",
    "Какая ставка налога на дивиденды учредителей?",
    "Какой официальный слоган компании для рекламных материалов?",
]


def slugify(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-zа-я0-9]+", "-", lowered, flags=re.IGNORECASE)
    return lowered.strip("-")[:80] or "doc"


class LMStudioGenerator:
    def __init__(self, base_url: str, model: str, api_key: str, timeout: float):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        retries: int = 3,
    ) -> str:
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=list(messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                if not content:
                    raise RuntimeError("Пустой ответ модели")
                return content.strip()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt == retries:
                    break
                time.sleep(0.7 * attempt)
        raise RuntimeError(f"Ошибка LM Studio после {retries} попыток: {last_err}")


def build_doc_prompt(doc_idx: int, doc_type: str, process: str) -> str:
    code = f"FF-{doc_type[:3].upper()}-{doc_idx:03d}"
    return (
        f"Сгенерируй документ типа '{doc_type}' для процесса '{process}'.\\n"
        "Требования:\\n"
        f"- Код документа: {code}.\\n"
        "- Предприятие вымышленное (например, ООО 'ФерроПоток').\\n"
        "- Добавь конкретные нормы: температуры/время/допуски/SLA, минимум 6 числовых параметров.\\n"
        "- Добавь роли и ответственность (минимум 4 роли).\\n"
        "- Структура markdown: заголовок, область применения, термины, требования, пошаговый процесс, контроль качества, "
        "исключения и эскалация, журналирование/артефакты, пересмотр документа.\\n"
        "- Объем: 700-1200 слов.\\n"
        "- Без упоминаний реальных компаний и законов."
    )


def extract_qa_pairs(
    llm: LMStudioGenerator,
    doc_markdown: str,
    doc_name: str,
    max_pairs: int,
    temperature: float,
) -> List[Dict[str, str]]:
    prompt = (
        f"На основе документа '{doc_name}' создай до {max_pairs} обучающих пар вопрос-ответ.\\n"
        "Верни только JSON-массив объектов формата: "
        "[{\"question\":\"...\",\"answer\":\"...\",\"source_excerpt\":\"...\"}].\\n"
        "Правила:\\n"
        "- Вопросы должны быть прикладными, по регламентам/нормативам/техкартам.\\n"
        "- Ответы только по данным из документа, без домысливаний.\\n"
        "- source_excerpt: короткая цитата (до 25 слов) для проверки, что ответ grounded.\\n"
        "- Не добавляй markdown-ограждения и комментарии.\\n\\n"
        f"Документ:\\n{doc_markdown}"
    )
    raw = llm.chat(
        [
            {"role": "system", "content": SYSTEM_QA},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=2500,
    )

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\\n", "", cleaned)
        cleaned = re.sub(r"\\n```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Не удалось распарсить JSON для {doc_name}: {exc}") from exc

    if not isinstance(parsed, list):
        raise RuntimeError(f"Ответ QA для {doc_name} не является списком")

    pairs: List[Dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer:
            pairs.append({"question": question, "answer": answer})
    return pairs


def generate_documents(
    llm: LMStudioGenerator,
    docs_count: int,
    out_dir: Path,
    temperature: float,
) -> List[Dict[str, str]]:
    docs_dir = out_dir / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    generated: List[Dict[str, str]] = []
    for idx in range(1, docs_count + 1):
        doc_type = random.choice(DOC_TYPES)
        process = random.choice(PROCESSES)
        prompt = build_doc_prompt(idx, doc_type, process)
        markdown = llm.chat(
            [
                {"role": "system", "content": SYSTEM_DOC},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=3500,
        )

        filename = f"{idx:03d}_{slugify(doc_type + '_' + process)}.md"
        path = docs_dir / filename
        path.write_text(markdown + "\n", encoding="utf-8")

        generated.append(
            {
                "id": f"doc-{idx:03d}",
                "type": doc_type,
                "process": process,
                "filename": filename,
                "content": markdown,
            }
        )
        print(f"[docs] {idx}/{docs_count}: {filename}")
    return generated


def build_training_dataset(
    llm: LMStudioGenerator,
    docs: List[Dict[str, str]],
    dataset_size: int,
    out_dir: Path,
    qa_per_doc_attempt: int,
    temperature: float,
    known_ratio: float,
) -> Path:
    if not docs:
        raise ValueError("Невозможно собрать датасет без документов")

    known_target = max(1, min(dataset_size, int(round(dataset_size * known_ratio))))
    unknown_target = max(0, dataset_size - known_target)

    known_samples: List[Dict[str, Any]] = []
    for doc in docs:
        if len(known_samples) >= known_target:
            break
        try:
            pairs = extract_qa_pairs(
                llm,
                doc_markdown=doc["content"],
                doc_name=doc["filename"],
                max_pairs=qa_per_doc_attempt,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] QA extraction failed for {doc['filename']}: {exc}")
            continue

        for pair in pairs:
            if len(known_samples) >= known_target:
                break
            rec = {
                "messages": [
                    {"role": "system", "content": DATASET_SYSTEM_PROMPT},
                    {"role": "user", "content": pair["question"]},
                    {"role": "assistant", "content": pair["answer"]},
                ],
                "meta": {
                    "sample_type": "known",
                    "source_doc": doc["filename"],
                },
            }
            known_samples.append(rec)

    if len(known_samples) < known_target:
        print(
            f"[warn] Не удалось набрать целевой объем grounded-сэмплов: "
            f"{len(known_samples)}/{known_target}."
        )

    unknown_samples: List[Dict[str, Any]] = []
    refusal = "Не могу ответить: в предоставленных документах нет этой информации."
    for idx in range(unknown_target):
        question = random.choice(UNKNOWN_QUESTION_TEMPLATES)
        question = f"{question} (запрос {idx + 1})"
        unknown_samples.append(
            {
                "messages": [
                    {"role": "system", "content": DATASET_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": refusal},
                ],
                "meta": {
                    "sample_type": "unknown",
                    "source_doc": None,
                },
            }
        )

    all_samples = known_samples + unknown_samples
    random.shuffle(all_samples)

    if len(all_samples) > dataset_size:
        all_samples = all_samples[:dataset_size]

    out_path = out_dir / "training_dataset.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    stats = {
        "dataset_size_requested": dataset_size,
        "dataset_size_written": len(all_samples),
        "known_samples": sum(1 for x in all_samples if x.get("meta", {}).get("sample_type") == "known"),
        "unknown_samples": sum(1 for x in all_samples if x.get("meta", {}).get("sample_type") == "unknown"),
    }
    (out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"[dataset] written {len(all_samples)} samples "
        f"(known={stats['known_samples']}, unknown={stats['unknown_samples']})"
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Генерация вымышленных производственных markdown-документов и "
            "обучающего JSONL-датасета через локальную LM Studio."
        )
    )
    parser.add_argument("--docs-count", type=int, default=30, help="Количество генерируемых документов")
    parser.add_argument("--dataset-size", type=int, default=600, help="Размер обучающего датасета (JSONL)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/factory_rag"), help="Папка вывода")

    parser.add_argument("--base-url", default="http://localhost:1234/v1", help="OpenAI-совместимый URL LM Studio")
    parser.add_argument("--model", default="local-model", help="Имя модели в LM Studio")
    parser.add_argument("--api-key", default="lm-studio", help="API key (для LM Studio обычно любое значение)")
    parser.add_argument("--timeout", type=float, default=120.0, help="Таймаут запроса к LM Studio (сек)")

    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно")
    parser.add_argument("--doc-temperature", type=float, default=0.8, help="Температура для генерации документов")
    parser.add_argument("--qa-temperature", type=float, default=0.2, help="Температура для генерации Q/A")
    parser.add_argument(
        "--known-ratio",
        type=float,
        default=0.7,
        help="Доля сэмплов с ответами по документам (остальное - отказ при отсутствии информации)",
    )
    parser.add_argument(
        "--qa-per-doc-attempt",
        type=int,
        default=8,
        help="Сколько Q/A пар пытаться извлечь из каждого документа",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.docs_count <= 0:
        raise ValueError("--docs-count должен быть > 0")
    if args.dataset_size <= 0:
        raise ValueError("--dataset-size должен быть > 0")
    if not (0.1 <= args.known_ratio <= 0.95):
        raise ValueError("--known-ratio должен быть в диапазоне [0.1, 0.95]")

    random.seed(args.seed)

    llm = LMStudioGenerator(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
    )

    print("[start] generating markdown documents")
    docs = generate_documents(
        llm=llm,
        docs_count=args.docs_count,
        out_dir=args.out_dir,
        temperature=args.doc_temperature,
    )

    print("[start] building training dataset")
    dataset_path = build_training_dataset(
        llm=llm,
        docs=docs,
        dataset_size=args.dataset_size,
        out_dir=args.out_dir,
        qa_per_doc_attempt=args.qa_per_doc_attempt,
        temperature=args.qa_temperature,
        known_ratio=args.known_ratio,
    )

    print(f"[done] documents: {args.out_dir / 'documents'}")
    print(f"[done] dataset: {dataset_path}")
    print(f"[done] stats: {args.out_dir / 'dataset_stats.json'}")


if __name__ == "__main__":
    main()

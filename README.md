# QLoRA + Support RAG (LlamaIndex Hybrid)

Проект для:

- генерации вымышленных документов техподдержки торговой компании;
- генерации SFT-датасета и demo-вопросов;
- hybrid RAG-инференса на LlamaIndex (BM25 + vector + optional rerank);
- QLoRA-обучения, merge и экспорта в GGUF.

## Что есть в репозитории

- `scripts/gen_support_rag_docs.py` — генерация support-документов, SFT-датасета и demo-вопросов.
- `scripts/build_llamaindex_index.py` — построение и локальное сохранение LlamaIndex индекса.
- `scripts/llamaindex_hybrid_rag.py` — инференс с hybrid retrieval, rerank, интерактивным CLI.
- `scripts/train_qlora_unsloth.py` — обучение QLoRA.
- `scripts/merge_lora.py` — merge LoRA в базовую модель.
- `scripts/convert_to_gguf.sh` — конвертация merged модели в GGUF.
- `scripts/quantize.sh` — квантизация GGUF.

## Требования

- Linux + Python 3.12.
- Для обучения: NVIDIA GPU + совместимый CUDA стек.
- Виртуальное окружение (пример):

```bash
source ~/venvs/QLoRA/bin/activate
```

Установка зависимостей:

```bash
pip install -U -r requirements.txt
```

## Быстрый старт: Support RAG

### 1) Сгенерировать документы + датасет + demo-вопросы

```bash
source ~/venvs/QLoRA/bin/activate

python scripts/gen_support_rag_docs.py \
  --mode all \
  --docs-count 60 \
  --dataset-size 1500 \
  --questions-count 50 \
  --out-dir data/trade_support_rag \
  --model qwen3-coder-30b-a3b-instruct \
  --qa-per-doc-attempt 15 \
  --grounded-ratio 0.6 \
  --sensitive-ratio 0.2 \
  --clarify-ratio 0.1 \
  --enforce-consistency \
  --doc-retries 3 \
  --unique-processes
```

### 2) Построить LlamaIndex

```bash
source ~/venvs/QLoRA/bin/activate

python scripts/build_llamaindex_index.py \
  --docs-dir data/trade_support_rag/documents \
  --persist-dir data/trade_support_rag/llamaindex_store \
  --base-url http://localhost:1234/v1 \
  --embed-model text-embedding-nomic-embed-text-v1.5 \
  --chunk-size 900 \
  --chunk-overlap 150
```

### 3) Задать вопрос (RAG on)

```bash
source ~/venvs/QLoRA/bin/activate

python scripts/llamaindex_hybrid_rag.py \
  --index-dir data/trade_support_rag/llamaindex_store \
  --chat-model local-model \
  --base-url http://localhost:1234/v1 \
  --question "Почему заказ подтверждается с задержкой после оплаты?" \
  --top-k 8 \
  --vector-top-k 16 \
  --bm25-top-k 16 \
  --rerank llm \
  --rerank-model local-model \
  --rerank-candidates 24 \
  --show-retrieval
```

### 4) Сравнение без RAG (RAG off)

```bash
source ~/venvs/QLoRA/bin/activate

python scripts/llamaindex_hybrid_rag.py \
  --disable-rag \
  --chat-model local-model \
  --base-url http://localhost:1234/v1 \
  --question "Почему заказ подтверждается с задержкой после оплаты?"
```

## `gen_support_rag_docs.py`

Назначение: единый генератор документов, SFT-датасета и demo-вопросов.

### Режимы

- `--mode docs` — только документы (`documents/*.md`).
- `--mode dataset` — только `training_dataset.jsonl` (из существующих документов).
- `--mode questions` — только demo-вопросы (`demo_questions.txt/json`).
- `--mode all` — документы + датасет + вопросы.

### Примеры

Только документы:

```bash
python scripts/gen_support_rag_docs.py \
  --mode docs \
  --docs-count 60 \
  --out-dir data/trade_support_rag \
  --model qwen3-coder-30b-a3b-instruct
```

Только датасет:

```bash
python scripts/gen_support_rag_docs.py \
  --mode dataset \
  --dataset-size 1500 \
  --docs-dir data/trade_support_rag/documents \
  --out-dir data/trade_support_rag \
  --model qwen3-coder-30b-a3b-instruct \
  --qa-per-doc-attempt 15 \
  --grounded-ratio 0.6 \
  --sensitive-ratio 0.2 \
  --clarify-ratio 0.1
```

Только вопросы:

```bash
python scripts/gen_support_rag_docs.py \
  --mode questions \
  --questions-count 50 \
  --docs-dir data/trade_support_rag/documents \
  --out-dir data/trade_support_rag \
  --model qwen3-coder-30b-a3b-instruct
```

### Ключевые аргументы

- `--docs-count` — число генерируемых документов.
- `--dataset-size` — размер SFT-датасета.
- `--questions-count` — число demo-вопросов.
- `--qa-per-doc-attempt` — максимум grounded Q/A на документ.
- `--grounded-ratio`, `--sensitive-ratio`, `--clarify-ratio` — баланс типов сэмплов.
- `--doc-temperature`, `--qa-temperature`, `--questions-temperature` — temperature по этапам.
- `--docs-dir` — источник документов для `dataset/questions`.
- `--out-dir` — папка вывода.
- `--enforce-consistency` / `--no-enforce-consistency` — включить/выключить проверку на конфликты.
- `--doc-retries` — число перегенераций документа при провале consistency-check.
- `--unique-processes` / `--allow-duplicate-processes` — управление повторами процессов.

### Выходные файлы

- `data/trade_support_rag/documents/*.md`
- `data/trade_support_rag/canonical_facts.json`
- `data/trade_support_rag/training_dataset.jsonl`
- `data/trade_support_rag/dataset_stats.json`
- `data/trade_support_rag/demo_questions.txt`
- `data/trade_support_rag/demo_questions.json`

`demo_questions.json` включает поля для сравнения:

- `question`
- `doc_number`
- `doc_filename`
- `answer_line`
- `source_excerpt`

## `build_llamaindex_index.py`

Назначение: build + persist индекса LlamaIndex для последующего retrieval.

### Пример

```bash
python scripts/build_llamaindex_index.py \
  --docs-dir data/trade_support_rag/documents \
  --persist-dir data/trade_support_rag/llamaindex_store \
  --base-url http://localhost:1234/v1 \
  --embed-model text-embedding-nomic-embed-text-v1.5
```

### Ключевые аргументы

- `--docs-dir` — папка с markdown.
- `--persist-dir` — куда сохранить индекс.
- `--recursive` — искать `.md` в подпапках.
- `--chunk-size`, `--chunk-overlap` — параметры чанкинга.
- `--embed-model`, `--base-url`, `--api-key` — embedding endpoint.

## `llamaindex_hybrid_rag.py`

Назначение: инференс по сохраненному индексу с hybrid retrieval и CLI-чатом.

### Режимы запуска

Одиночный вопрос:

```bash
python scripts/llamaindex_hybrid_rag.py \
  --index-dir data/trade_support_rag/llamaindex_store \
  --chat-model local-model \
  --question "Какие поля обязательны перед эскалацией тикета?"
```

Интерактивный чат:

```bash
python scripts/llamaindex_hybrid_rag.py \
  --index-dir data/trade_support_rag/llamaindex_store \
  --chat-model local-model \
  --rerank llm
```

Без retrieval (для демо-сравнения):

```bash
python scripts/llamaindex_hybrid_rag.py \
  --disable-rag \
  --chat-model local-model
```

### Ключевые аргументы

- Retrieval:
- `--top-k`, `--vector-top-k`, `--bm25-top-k`
- `--vector-weight`, `--bm25-weight`
- `--bm25-language`
- Rerank:
- `--rerank {off,llm}`
- `--rerank-model` (модель для rerank, по умолчанию = `--chat-model`)
- `--rerank-candidates`
- `--rerank-temperature`
- LLM:
- `--chat-model`, `--temperature`, `--max-tokens`
- Demo/no-RAG:
- `--disable-rag`
- CLI memory:
- `--load-session/--no-load-session`
- `--save-session/--no-save-session`
- `--memory-turns`
- `--session-file`
- `--prompt-history-file`

Интерактивные команды в чате:

- `/exit` — выход
- `/reset` — очистить память сессии
- `/save` — сохранить сессию

## Обучение QLoRA

```bash
source ~/venvs/QLoRA/bin/activate

python scripts/train_qlora_unsloth.py \
  --model-dir base_model \
  --data-path data/trade_support_rag/training_dataset.jsonl \
  --out-dir adapter/qlora_support_rag \
  --num-epochs 2 \
  --learning-rate 2e-4 \
  --batch-size 1 \
  --grad-accum 16
```

## Merge, GGUF, Quantize

Merge LoRA:

```bash
python scripts/merge_lora.py \
  --base-model base_model \
  --adapter adapter/qlora_support_rag \
  --out-dir merged_model
```

Конвертация в GGUF:

```bash
bash scripts/convert_to_gguf.sh \
  --input-dir merged_model \
  --outtype f16 \
  --outfile gguf/model.f16.gguf
```

Квантизация:

```bash
bash scripts/quantize.sh \
  --input gguf/model.f16.gguf \
  --output gguf/qlora.Q4_K_M.gguf \
  --quant Q4_K_M
```

## Полезные проверки

```bash
source ~/venvs/QLoRA/bin/activate

python scripts/gen_support_rag_docs.py --help
python scripts/build_llamaindex_index.py --help
python scripts/llamaindex_hybrid_rag.py --help
python scripts/train_qlora_unsloth.py --help
python scripts/merge_lora.py --help
bash scripts/convert_to_gguf.sh --help
bash scripts/quantize.sh --help
```

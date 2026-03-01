# QLoRA Fine-Tuning (Unsloth) + GGUF Pipeline

Проект для дообучения локальной LLM на доменных данных (пример: бизнес-процессы пиццерии), последующего merge LoRA-адаптера и экспорта в GGUF.

## Что в проекте

- `scripts/get_dataset_pizza.py` - генерация синтетического train/eval датасета с уникальными фактами.
- `scripts/gen_dataset.py` - генерация датасета через локальную LLM (LM Studio/OpenAI-совместимый API).
- `scripts/train_qlora_unsloth.py` - обучение QLoRA через Unsloth + TRL `SFTTrainer`.
- `scripts/merge_lora.py` - merge LoRA-адаптера в базовую модель.
- `scripts/convert_to_gguf.sh` - конвертация merged HF-модели в GGUF.
- `scripts/quantize.sh` - квантизация GGUF (например, `Q4_K_M`).

## Требования

- Linux + NVIDIA GPU (для Unsloth training).
- Python 3.12 и окружение с пакетами `unsloth`, `trl`, `transformers`, `peft`, `datasets`, `torch`.
- `llama.cpp` в директории `../llama.cpp` (рекомендуемо; есть fallback на `./llama.cpp`).
- Собранный `llama-quantize` по пути `../llama.cpp/build/bin/llama-quantize`.

Сборка `llama.cpp`:

```bash
cd ..
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build -j
cd ../QLoRA
```

Пример активации окружения:

```bash
source ~/venvs/QLoRA/bin/activate
```

Установка Python-зависимостей:

```bash
pip install -U "torch" "transformers" "datasets" "accelerate" "peft" "trl" "bitsandbytes"
```

Скачивание базовой модели с Hugging Face:

```bash
pip install -U "huggingface_hub[cli]" git-lfs
git lfs install
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir base_model
```

## Структура данных

Train-файл ожидается в формате JSONL, по одному объекту на строку:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

## 1) Генерация датасета

Синтетический датасет (без вызовов LLM):

```bash
python scripts/get_dataset_pizza.py --n 1200 --eval-n 120 \
  --out data/pizza_dataset_1200.jsonl \
  --eval-out data/pizza_eval_set.jsonl
```

Генератор делает mixed-task train:

- полный регламент (структура 1..9),
- извлечение фактов в JSON (`code`, `sla_min`, `escalation`, `log_key`),
- извлечение в формате `key=value`,
- disambiguation-кейсы с похожими названиями процессов.

Датасет через локальную LLM (LM Studio):

```bash
python scripts/gen_dataset.py
```

## 2) Обучение QLoRA

Базовый запуск:

```bash
python scripts/train_qlora_unsloth.py \
  --model-dir base_model \
  --data-path data/pizza_dataset_1200.jsonl \
  --out-dir adapter/qlora_pizza \
  --num-epochs 2 \
  --learning-rate 2e-4 \
  --batch-size 1 \
  --grad-accum 16
```

Полезные параметры:

- `--max-seq-length`
- `--load-in-4bit / --no-load-in-4bit`
- `--lora-r`, `--lora-alpha`, `--lora-dropout`
- `--target-modules` (через запятую)
- `--dataset-num-proc`
- `--optim`, `--lr-scheduler-type`

## 3) Merge LoRA в базовую модель

```bash
python scripts/merge_lora.py \
  --base-model base_model \
  --adapter adapter/qlora_pizza \
  --out-dir merged_model
```

Дополнительно:

- `--dtype {float16,bfloat16,float32}`
- `--device-map`
- `--trust-remote-code / --no-trust-remote-code`
- `--safe-serialization / --no-safe-serialization`

## 4) Конвертация в GGUF

```bash
bash scripts/convert_to_gguf.sh \
  --input-dir merged_model \
  --outtype f16 \
  --outfile gguf/model.f16.gguf
```

## 5) Квантизация GGUF

```bash
bash scripts/quantize.sh \
  --input gguf/model.f16.gguf \
  --output gguf/qlora.Q4_K_M.gguf \
  --quant Q4_K_M
```

## 6) Подключение модели в LM Studio

```bash
mkdir -p ~/.cache/lm-studio/models/my/qlora
cp gguf/qlora.Q4_K_M.gguf ~/.cache/lm-studio/models/my/qlora/
```

## 7) Оценка качества дообучения (base vs fine-tuned)

Скрипт сравнивает `base_model` и `merged_model` на eval-наборе и считает:

- `hit_all_rate` - доля ответов, где найдены все маркеры из `expected_contains`
- `partial_match_rate` - средняя доля найденных маркеров
- дельту tuned против base
- итоговый verdict: `SUCCESS` / `NOT SUCCESS`

Быстрый тест:

```bash
python scripts/eval_finetune_vs_base.py --max-samples 10
```

Полный прогон:

```bash
python scripts/eval_finetune_vs_base.py \
  --base-model base_model \
  --tuned-model merged_model \
  --eval-path data/pizza_eval_set.jsonl \
  --max-new-tokens 180 \
  --report-out data/eval_report.json
```

Пороговые критерии (можно менять):

- `--min-tuned-hit-all` (по умолчанию `0.70`)
- `--min-delta-hit-all` (по умолчанию `0.20`)

## Проверка параметров скриптов

```bash
python scripts/train_qlora_unsloth.py --help
python scripts/merge_lora.py --help
python scripts/eval_finetune_vs_base.py --help
bash scripts/convert_to_gguf.sh --help
bash scripts/quantize.sh --help
```

## Замечания

- `train_qlora_unsloth.py` рассчитан на GPU-окружение с поддержкой `bf16`.
- Для сравнения base vs fine-tuned можно использовать `data/pizza_eval_set.jsonl` и проверять наличие маркеров из `expected_contains`.

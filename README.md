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
- `llama.cpp` в директории `./llama.cpp`.
- Собранный `llama-quantize` по пути `./llama.cpp/build/bin/llama-quantize`.

Пример активации окружения:

```bash
source ~/venvs/QLoRA/bin/activate
```

## Структура данных

Train-файл ожидается в формате JSONL, по одному объекту на строку:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

## 1) Генерация датасета

Синтетический датасет (без вызовов LLM):

```bash
python scripts/get_dataset_pizza.py --n 300 --eval-n 40 \
  --out data/pizza_dataset_300.jsonl \
  --eval-out data/pizza_eval_set.jsonl
```

Датасет через локальную LLM (LM Studio):

```bash
python scripts/gen_dataset.py
```

## 2) Обучение QLoRA

Базовый запуск:

```bash
python scripts/train_qlora_unsloth.py \
  --model-dir base_model \
  --data-path data/pizza_dataset_300.jsonl \
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
  --output gguf/model.Q4_K_M.gguf \
  --quant Q4_K_M
```

## Проверка параметров скриптов

```bash
python scripts/train_qlora_unsloth.py --help
python scripts/merge_lora.py --help
bash scripts/convert_to_gguf.sh --help
bash scripts/quantize.sh --help
```

## Замечания

- `train_qlora_unsloth.py` рассчитан на GPU-окружение с поддержкой `bf16`.
- Для сравнения base vs fine-tuned можно использовать `data/pizza_eval_set.jsonl` и проверять наличие маркеров из `expected_contains`.

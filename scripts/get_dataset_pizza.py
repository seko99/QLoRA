import argparse
import json
import random
from pathlib import Path


OUT_DEFAULT = Path("data/pizza_dataset_memory_2000.jsonl")
EVAL_DEFAULT = Path("data/pizza_eval_memory_200.jsonl")

SYSTEM_MEMORY = (
    "Ты отвечаешь только по внутренним процессам сети пиццерий 'Кварц'. "
    "Если просят факты процесса, возвращай ТОЛЬКО JSON с ключами: code, sla_min, escalation, log_key. "
    "Никаких пояснений и лишних полей."
)

PROCESSES = [
    {"name": "Ночной замес теста Луна-9", "code": "QZ-DGH-114", "sla_min": "19", "escalation": "E-44-киви", "log_key": "ferment_delta"},
    {"name": "Перезапуск печи Орбита-4 после перегрева", "code": "QZ-PRH-207", "sla_min": "11", "escalation": "E-12-янтарь", "log_key": "burn_zone_map"},
    {"name": "Срочный ремейк заказа при ошибке топпинга", "code": "QZ-RMK-389", "sla_min": "13", "escalation": "E-77-лазурь", "log_key": "topping_swap_id"},
    {"name": "Пересборка комбо для курьера на точке Каскад", "code": "QZ-CMB-451", "sla_min": "17", "escalation": "E-31-графит", "log_key": "combo_patch_rev"},
    {"name": "Инвентаризация соуса Метеор-11", "code": "QZ-SAU-562", "sla_min": "23", "escalation": "E-65-неон", "log_key": "sauce_density_idx"},
    {"name": "Переназначение курьера при падении трека GPS", "code": "QZ-CRR-618", "sla_min": "9", "escalation": "E-09-опал", "log_key": "gps_drift_flag"},
    {"name": "Контроль партии сыра Полярный-21", "code": "QZ-CHS-709", "sla_min": "27", "escalation": "E-83-кобальт", "log_key": "cheese_batch_wave"},
    {"name": "Ночной возврат термосумки курьера", "code": "QZ-BAG-774", "sla_min": "15", "escalation": "E-24-рубин", "log_key": "bag_cycle_num"},
    {"name": "Разбор претензии по холодной пицце", "code": "QZ-CLD-882", "sla_min": "21", "escalation": "E-58-лайм", "log_key": "temp_drop_delta"},
    {"name": "Ручная верификация промокода Янтарь-6", "code": "QZ-PRM-943", "sla_min": "7", "escalation": "E-03-дымка", "log_key": "promo_shadow_hit"},
]

PROMPT_TEMPLATES = [
    "Для процесса '{name}' верни JSON: code, sla_min, escalation, log_key.",
    "Назови параметры процесса '{name}' в JSON (code/sla_min/escalation/log_key).",
    "Только факты по '{name}'. Формат JSON с ключами code, sla_min, escalation, log_key.",
    "Процесс '{name}': нужен чистый JSON без текста: code, sla_min, escalation, log_key.",
]

HARD_NEGATIVE_TEMPLATES = [
    (
        "Не перепутай процессы '{name_a}' и '{name_b}'. "
        "Верни JSON только для '{name_a}' с полями code, sla_min, escalation, log_key."
    ),
    (
        "Есть два названия: '{name_a}' и '{name_b}'. "
        "Ответь только по '{name_a}' и только JSON."
    ),
]


def make_json_answer(p: dict) -> str:
    return json.dumps(
        {
            "code": p["code"],
            "sla_min": p["sla_min"],
            "escalation": p["escalation"],
            "log_key": p["log_key"],
        },
        ensure_ascii=False,
    )


def make_train_record() -> dict:
    p = random.choice(PROCESSES)
    if random.random() < 0.2:
        q = random.choice([x for x in PROCESSES if x["name"] != p["name"]])
        user = random.choice(HARD_NEGATIVE_TEMPLATES).format(name_a=p["name"], name_b=q["name"])
    else:
        user = random.choice(PROMPT_TEMPLATES).format(name=p["name"])
    assistant = make_json_answer(p)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MEMORY},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def make_eval_records(n: int) -> list[dict]:
    rows = []
    for i in range(n):
        p = PROCESSES[i % len(PROCESSES)]
        if i % 5 == 0:
            q = PROCESSES[(i + 3) % len(PROCESSES)]
            user = random.choice(HARD_NEGATIVE_TEMPLATES).format(name_a=p["name"], name_b=q["name"])
        else:
            user = random.choice(PROMPT_TEMPLATES).format(name=p["name"])
        expected_json = {
            "code": p["code"],
            "sla_min": p["sla_min"],
            "escalation": p["escalation"],
            "log_key": p["log_key"],
        }
        rows.append(
            {
                "system": SYSTEM_MEMORY,
                "user": user,
                "expected_json": expected_json,
                "expected_contains": [p["code"], p["sla_min"], p["escalation"], p["log_key"]],
                "process_name": p["name"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate memory-only pizza dataset")
    parser.add_argument("--n", type=int, default=2000, help="Train examples count")
    parser.add_argument("--eval-n", type=int, default=200, help="Eval examples count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT, help="Train JSONL output")
    parser.add_argument("--eval-out", type=Path, default=EVAL_DEFAULT, help="Eval JSONL output")
    args = parser.parse_args()

    random.seed(args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            rec = make_train_record()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                print(f"generated train: {i + 1}/{args.n}")

    eval_rows = make_eval_records(args.eval_n)
    args.eval_out.parent.mkdir(parents=True, exist_ok=True)
    with args.eval_out.open("w", encoding="utf-8") as f:
        for rec in eval_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"train dataset saved to: {args.out}")
    print(f"eval set saved to: {args.eval_out}")


if __name__ == "__main__":
    main()

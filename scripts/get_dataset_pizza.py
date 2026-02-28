import argparse
import json
import random
from pathlib import Path


OUT_DEFAULT = Path("data/pizza_dataset_300.jsonl")
EVAL_DEFAULT = Path("data/pizza_eval_set.jsonl")

SYSTEM_TEMPLATE = (
    "Ты бизнес-аналитик сети пиццерий 'Кварц'. "
    "Это закрытые внутренние регламенты, которых нет в открытых источниках. "
    "Отвечай строго по шаблону:\n"
    "1) Краткое описание (1–2 абзаца)\n"
    "2) Цель процесса\n"
    "3) Участники и ответственность\n"
    "4) Входные условия\n"
    "5) Основной поток (шаги 1..N)\n"
    "6) Статусы и переходы\n"
    "7) Сроки / SLA\n"
    "8) Исключения и спорные случаи\n"
    "9) Что логируем и какие уведомления отправляем\n\n"
    "Используй только термины из внутренних регламентов ниже.\n\n"
)

PROCESSES = [
    {
        "name": "Ночной замес теста Луна-9",
        "code": "QZ-DGH-114",
        "sla_min": 19,
        "escalation": "E-44-киви",
        "owner": "мастер теста",
        "approval": "шеф смены",
        "log_key": "ferment_delta",
        "notify": "канал oven_watch",
    },
    {
        "name": "Перезапуск печи Орбита-4 после перегрева",
        "code": "QZ-PRH-207",
        "sla_min": 11,
        "escalation": "E-12-янтарь",
        "owner": "инженер линии",
        "approval": "операционный менеджер",
        "log_key": "burn_zone_map",
        "notify": "канал heat_guard",
    },
    {
        "name": "Срочный ремейк заказа при ошибке топпинга",
        "code": "QZ-RMK-389",
        "sla_min": 13,
        "escalation": "E-77-лазурь",
        "owner": "старший пиццамейкер",
        "approval": "дежурный администратор",
        "log_key": "topping_swap_id",
        "notify": "канал remake_hub",
    },
    {
        "name": "Пересборка комбо для курьера на точке Каскад",
        "code": "QZ-CMB-451",
        "sla_min": 17,
        "escalation": "E-31-графит",
        "owner": "сборщик выдачи",
        "approval": "координатор доставки",
        "log_key": "combo_patch_rev",
        "notify": "канал rider_sync",
    },
    {
        "name": "Инвентаризация соуса Метеор-11",
        "code": "QZ-SAU-562",
        "sla_min": 23,
        "escalation": "E-65-неон",
        "owner": "кладовщик ингредиентов",
        "approval": "товаровед смены",
        "log_key": "sauce_density_idx",
        "notify": "канал stock_signal",
    },
    {
        "name": "Переназначение курьера при падении трека GPS",
        "code": "QZ-CRR-618",
        "sla_min": 9,
        "escalation": "E-09-опал",
        "owner": "диспетчер доставки",
        "approval": "лид логистики",
        "log_key": "gps_drift_flag",
        "notify": "канал route_relay",
    },
    {
        "name": "Контроль партии сыра Полярный-21",
        "code": "QZ-CHS-709",
        "sla_min": 27,
        "escalation": "E-83-кобальт",
        "owner": "контролер качества",
        "approval": "технолог производства",
        "log_key": "cheese_batch_wave",
        "notify": "канал qa_batch",
    },
    {
        "name": "Ночной возврат термосумки курьера",
        "code": "QZ-BAG-774",
        "sla_min": 15,
        "escalation": "E-24-рубин",
        "owner": "координатор выдачи",
        "approval": "менеджер закрытия смены",
        "log_key": "bag_cycle_num",
        "notify": "канал shift_close",
    },
    {
        "name": "Разбор претензии по холодной пицце",
        "code": "QZ-CLD-882",
        "sla_min": 21,
        "escalation": "E-58-лайм",
        "owner": "специалист поддержки",
        "approval": "лид клиентского сервиса",
        "log_key": "temp_drop_delta",
        "notify": "канал care_escalation",
    },
    {
        "name": "Ручная верификация промокода Янтарь-6",
        "code": "QZ-PRM-943",
        "sla_min": 7,
        "escalation": "E-03-дымка",
        "owner": "кассир смены",
        "approval": "финансовый контролер",
        "log_key": "promo_shadow_hit",
        "notify": "канал promo_guard",
    },
]

QUESTION_TEMPLATES = [
    "Опиши процесс: {name}.",
    "Нужен регламент процесса '{name}' для новой смены.",
    "Разбери крайний случай по процессу '{name}': сроки истекли, есть конфликт ролей. Ответ по шаблону.",
    "Для процесса '{name}' опиши статусы, переходы, SLA и эскалацию. Ответ строго по шаблону.",
    "Как в процессе '{name}' фиксируются логи и какие уведомления отправляются? Ответ по шаблону.",
]


def build_system() -> str:
    lines = [SYSTEM_TEMPLATE, "Внутренние процессы сети 'Кварц':"]
    for p in PROCESSES:
        lines.append(
            f"- {p['name']} | код {p['code']} | SLA {p['sla_min']} мин | "
            f"эскалация {p['escalation']} | владелец {p['owner']} | "
            f"подтверждает {p['approval']} | лог-ключ {p['log_key']} | уведомление {p['notify']}"
        )
    return "\n".join(lines)


def build_user_prompt(proc: dict) -> str:
    return random.choice(QUESTION_TEMPLATES).format(name=proc["name"])


def build_assistant(proc: dict) -> str:
    return (
        f"1) **Краткое описание**\n"
        f"Процесс '{proc['name']}' регулирует нестандартную операцию внутри сети 'Кварц'. "
        f"Выполнение ведется по регламенту {proc['code']} и всегда проходит через фиксированную точку эскалации {proc['escalation']}.\n\n"
        f"Процесс применяется как для операционной стабилизации смены, так и для защиты клиентского SLA. "
        f"Все шаги требуют журналирования по ключу {proc['log_key']} и отправки служебного сигнала в {proc['notify']}.\n\n"
        f"2) **Цель процесса**\n"
        f"Обеспечить предсказуемое выполнение регламента {proc['code']} без нарушения лимита {proc['sla_min']} минут и с прозрачной ответственностью.\n\n"
        f"3) **Участники и ответственность**\n"
        f"- Владелец процесса: {proc['owner']}.\n"
        f"- Подтверждающая роль: {proc['approval']}.\n"
        f"- Смежные участники: поддержка смены, логистика, финансы (по необходимости).\n\n"
        f"4) **Входные условия**\n"
        f"- Зафиксирован инцидент/событие, подпадающее под процесс '{proc['name']}'.\n"
        f"- Открыта карточка процесса с кодом {proc['code']}.\n"
        f"- Канал уведомлений {proc['notify']} доступен.\n\n"
        f"5) **Основной поток (шаги 1..N)**\n"
        f"1. {proc['owner']} регистрирует событие и присваивает карточке код {proc['code']}.\n"
        f"2. Выполняется первичная проверка критериев запуска.\n"
        f"3. Исполняются операционные действия процесса с контролем времени.\n"
        f"4. При риске выхода за SLA активируется эскалация {proc['escalation']}.\n"
        f"5. {proc['approval']} подтверждает результат и закрытие кейса.\n"
        f"6. В журнал пишется значение {proc['log_key']}, отправляется уведомление в {proc['notify']}.\n\n"
        f"6) **Статусы и переходы**\n"
        f"- Статусы: `new` -> `in_progress` -> `await_approval` -> `done`.\n"
        f"- Дополнительный переход: `in_progress` -> `escalated` (по триггеру {proc['escalation']}).\n\n"
        f"7) **Сроки / SLA**\n"
        f"- Целевой SLA: до {proc['sla_min']} минут с момента регистрации кейса.\n"
        f"- Подтверждение финального результата: не позже 3 минут после фактического завершения шага исполнения.\n\n"
        f"8) **Исключения и спорные случаи**\n"
        f"- Если владелец процесса недоступен, кейс сразу переводится в `escalated` по коду {proc['escalation']}.\n"
        f"- Если подтверждающая роль отклоняет результат, статус возвращается в `in_progress` с обязательным комментарием причины.\n"
        f"- При неоднозначных данных приоритет у последней записи {proc['log_key']} в карточке кейса.\n\n"
        f"9) **Что логируем и какие уведомления отправляем**\n"
        f"- Логи: код процесса {proc['code']}, таймстемпы переходов, причина эскалации, итоговое решение, ключ {proc['log_key']}.\n"
        f"- Уведомления: оперативное в {proc['notify']}, подтверждение закрытия владельцу и подтверждающей роли."
    )


def build_eval_records(eval_size: int) -> list[dict]:
    out = []
    for i in range(eval_size):
        p = PROCESSES[i % len(PROCESSES)]
        out.append(
            {
                "user": (
                    f"Для процесса '{p['name']}' назови код регламента, SLA в минутах, "
                    f"код эскалации и лог-ключ."
                ),
                "expected_contains": [p["code"], str(p["sla_min"]), p["escalation"], p["log_key"]],
                "process_name": p["name"],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="Количество train-примеров")
    parser.add_argument("--eval-n", type=int, default=40, help="Количество eval-запросов для сравнения моделей")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT, help="Путь для train jsonl")
    parser.add_argument("--eval-out", type=Path, default=EVAL_DEFAULT, help="Путь для eval jsonl")
    args = parser.parse_args()

    random.seed(args.seed)
    system = build_system()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            proc = random.choice(PROCESSES)
            user = build_user_prompt(proc)
            assistant = build_assistant(proc)
            rec = {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(f"generated train: {i + 1}/{args.n}")

    eval_records = build_eval_records(args.eval_n)
    args.eval_out.parent.mkdir(parents=True, exist_ok=True)
    with args.eval_out.open("w", encoding="utf-8") as f:
        for rec in eval_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"train dataset saved to: {args.out}")
    print(f"eval set saved to: {args.eval_out}")


if __name__ == "__main__":
    main()

import json
import time

from langchain_core.messages import HumanMessage

from agent import app


# ================================================================
# Тест-кейси
# ================================================================

test_cases = [
    {
        "id": "TC-001",
        "query": (
            "Я їду удвох на 5 днів. "
            "Щоденний бюджет на одну людину — 80 євро. "
            "Порахуй загальний бюджет подорожі."
        ),
        "expected": (
            "Агент повинен використати calculate_trip_budget "
            "і розрахувати загальний бюджет €800."
        ),
        "complexity": "simple",
    },

    {
        "id": "TC-002",
        "query": (
            "Я планую подорож на 4 дні для двох людей. "
            "Щоденний бюджет на одну людину — 70 євро. "
            "Також хочу забронювати один номер на 4 ночі "
            "по 100 євро за ніч. "
            "Порахуй бюджет подорожі та окремо вартість готелю."
        ),
        "expected": (
            "Агент повинен використати calculate_trip_budget "
            "та estimate_hotel_cost. "
            "Бюджет подорожі — €560, готель — €400."
        ),
        "complexity": "medium",
    },

    {
        "id": "TC-003",
        "query": (
            "Двоє людей планують подорож на відстань 500 км. "
            "Для них найважливіша швидкість. "
            "Подорож триватиме 5 днів, "
            "щоденний бюджет на одну людину — 90 євро. "
            "Порекомендуй транспорт і порахуй загальний бюджет."
        ),
        "expected": (
            "Агент повинен використати recommend_transport "
            "та calculate_trip_budget. "
            "Для 500 км з пріоритетом швидкості рекомендовано літак, "
            "а бюджет становить €900."
        ),
        "complexity": "medium",
    },

    {
        "id": "TC-004",
        "query": (
            "Допоможи спланувати подорож для двох людей на 7 днів. "
            "Відстань до місця призначення — 1200 км, "
            "головний пріоритет — швидкість. "
            "Щоденний бюджет на одну людину — 75 євро. "
            "Також потрібен один готельний номер на 6 ночей "
            "по 110 євро за ніч. "
            "Порекомендуй транспорт, порахуй бюджет подорожі "
            "та окремо вартість готелю."
        ),
        "expected": (
            "Агент повинен використати recommend_transport, "
            "calculate_trip_budget та estimate_hotel_cost. "
            "Для 1200 км рекомендовано літак, "
            "бюджет подорожі — €1050, "
            "вартість готелю — €660."
        ),
        "complexity": "complex",
    },
]


# ================================================================
# Допоміжна функція для збору tool calls
# ================================================================

def collect_tool_calls(messages: list) -> list[str]:
    """Збирає назви всіх tools, які викликав агент."""

    tool_calls = []

    for message in messages:

        calls = getattr(
            message,
            "tool_calls",
            [],
        )

        if calls:

            for call in calls:

                tool_name = call.get(
                    "name",
                    "unknown",
                )

                tool_calls.append(
                    tool_name
                )

    return tool_calls


# ================================================================
# Запуск одного test case
# ================================================================

def run_test_case(test_case: dict) -> dict:
    """Запускає один тест-кейс та повертає його результат."""

    print(
        f"\nRunning {test_case['id']}: "
        f"{test_case['query'][:60]}..."
    )


    # ------------------------------------------------------------
    # Початок вимірювання часу
    # ------------------------------------------------------------

    start = time.monotonic()


    try:

        # --------------------------------------------------------
        # Запуск LangGraph
        # --------------------------------------------------------

        result = app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=test_case["query"]
                    )
                ],

                "step_count": 0,

                # Початок timeout для конкретного test case
                "start_time": time.monotonic(),

                "stop_reason": None,

                "tool_call_history": [],

                "final_answer": None,
            },

            # Додатковий safeguard LangGraph
            {
                "recursion_limit": 30
            },
        )


        # --------------------------------------------------------
        # Structured output агента
        # --------------------------------------------------------

        final_answer = result.get(
            "final_answer"
        )


        if final_answer is not None:

            actual = (
                final_answer.model_dump()
            )

        else:

            # Fallback, якщо formatter не повернув результат
            actual = {
                "answer": str(
                    result["messages"][-1].content
                ),
                "confidence": 0,
                "sources": [],
            }


        # --------------------------------------------------------
        # Кількість LLM-кроків
        # --------------------------------------------------------

        steps = result.get(
            "step_count",
            0,
        )


        # --------------------------------------------------------
        # Збираємо tool calls
        # --------------------------------------------------------

        tool_calls = collect_tool_calls(
            result.get(
                "messages",
                [],
            )
        )


        # --------------------------------------------------------
        # Причина зупинки
        # --------------------------------------------------------

        stop_reason = result.get(
            "stop_reason"
        )


        # Чи спрацював max_steps
        max_steps_reached = (
            stop_reason == "max_steps"
        )


        # Чи спрацював timeout
        timeout_reached = (
            stop_reason == "timeout"
        )


        # Чи виявлено цикл
        loop_detected = (
            stop_reason == "loop_detected"
        )


        status = "success"


        # Якщо safety mechanism зупинив агента,
        # тест вважаємо завершеним частково
        if stop_reason:
            status = "stopped"


    except Exception as error:

        # --------------------------------------------------------
        # Обробка помилки
        # --------------------------------------------------------

        actual = {
            "answer": f"Error: {error}",
            "confidence": 0,
            "sources": [],
        }

        steps = -1

        tool_calls = []

        stop_reason = None

        max_steps_reached = False

        timeout_reached = False

        loop_detected = False

        status = "error"


    # ------------------------------------------------------------
    # Загальний час виконання test case
    # ------------------------------------------------------------

    elapsed_ms = int(
        (
            time.monotonic()
            - start
        )
        * 1000
    )


    # ------------------------------------------------------------
    # Формуємо структурований результат
    # ------------------------------------------------------------

    test_result = {
        "test_id": test_case["id"],

        "complexity": test_case[
            "complexity"
        ],

        "query": test_case[
            "query"
        ],

        "expected": test_case[
            "expected"
        ],

        "actual": actual,

        "status": status,

        "steps": steps,

        "tool_calls": tool_calls,

        "elapsed_ms": elapsed_ms,

        "max_steps_reached": max_steps_reached,

        "timeout_reached": timeout_reached,

        "loop_detected": loop_detected,

        "stop_reason": stop_reason,
    }


    return test_result


# ================================================================
# Запуск усіх тестів
# ================================================================

def run_all_tests() -> list[dict]:
    """Запускає всі test cases."""

    results = []


    print(
        "\n"
        + "=" * 70
    )

    print(
        "STARTING AGENT TESTS"
    )

    print(
        "=" * 70
    )


    for test_case in test_cases:

        result = run_test_case(
            test_case
        )

        results.append(
            result
        )


        # --------------------------------------------------------
        # Короткий результат у консолі
        # --------------------------------------------------------

        print(
            f"{result['test_id']}: "
            f"{result['status']} | "
            f"{result['steps']} LLM steps | "
            f"{result['elapsed_ms']} ms"
        )

        print(
            f"Tools: "
            f"{result['tool_calls']}"
        )


    return results


# ================================================================
# Збереження test_results.json
# ================================================================

def save_results(
    results: list[dict],
    filepath: str = "test_results.json",
) -> None:
    """Зберігає результати тестів у JSON-файл."""

    # ------------------------------------------------------------
    # Загальна статистика
    # ------------------------------------------------------------

    successful_tests = sum(
        1
        for result in results
        if result["status"] == "success"
    )


    stopped_tests = sum(
        1
        for result in results
        if result["status"] == "stopped"
    )


    error_tests = sum(
        1
        for result in results
        if result["status"] == "error"
    )


    output = {
        "summary": {
            "total_tests": len(
                results
            ),

            "successful": successful_tests,

            "stopped": stopped_tests,

            "errors": error_tests,
        },

        "results": results,
    }


    # ------------------------------------------------------------
    # test_results.json створюється автоматично
    # ------------------------------------------------------------

    with open(
        filepath,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )


    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Results saved: "
        f"{filepath}"
    )

    print(
        f"Total test cases: "
        f"{len(results)}"
    )

    print(
        f"Successful: "
        f"{successful_tests}"
    )

    print(
        f"Stopped: "
        f"{stopped_tests}"
    )

    print(
        f"Errors: "
        f"{error_tests}"
    )

    print(
        "=" * 70
    )


# ================================================================
# Запуск test runner
# ================================================================

if __name__ == "__main__":

    results = run_all_tests()

    save_results(
        results
    )
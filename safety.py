import json
import time


# ================================================================
# Налаштування захисних механізмів
# ================================================================

# Максимальна кількість LLM-ітерацій
MAX_STEPS = 10

# Максимальний загальний час виконання агента
TIMEOUT_SECONDS = 120

# Кількість однакових tool calls поспіль,
# після якої вважаємо, що агент зациклився
MAX_SAME_TOOL_CALLS = 3


# ================================================================
# Перевірка max_steps
# ================================================================

def max_steps_reached(step_count: int) -> bool:
    """Перевіряє, чи досягнуто максимальну кількість кроків."""

    return step_count >= MAX_STEPS


# ================================================================
# Перевірка timeout
# ================================================================

def timeout_reached(start_time: float) -> bool:
    """Перевіряє, чи перевищено загальний час виконання агента.

    Для вимірювання часу використовується time.monotonic(),
    оскільки він підходить для вимірювання часових інтервалів.

    Args:
        start_time: Час початку виконання агента.

    Returns:
        True, якщо timeout перевищено, інакше False.
    """

    elapsed = time.monotonic() - start_time

    return elapsed >= TIMEOUT_SECONDS


# ================================================================
# Loop Detector
# ================================================================

class LoopDetector:
    """Виявляє повторні однакові виклики tools.

    Якщо один і той самий tool з однаковими аргументами
    викликається 3 рази поспіль, вважаємо це зацикленням.
    """

    def __init__(
        self,
        max_repeats: int = MAX_SAME_TOOL_CALLS,
    ):
        self.max_repeats = max_repeats


    def create_signature(
        self,
        tool_call: dict,
    ) -> str:
        """Створює стабільну сигнатуру tool call."""

        tool_name = tool_call.get(
            "name",
            ""
        )

        arguments = tool_call.get(
            "args",
            {}
        )

        arguments_json = json.dumps(
            arguments,
            sort_keys=True,
            ensure_ascii=False,
        )

        return (
            f"{tool_name}:"
            f"{arguments_json}"
        )


    def add_calls(
        self,
        history: list[str],
        tool_calls: list[dict],
    ) -> list[str]:
        """Додає поточні tool calls до історії."""

        new_history = list(history)

        for tool_call in tool_calls:

            signature = self.create_signature(
                tool_call
            )

            new_history.append(
                signature
            )

        return new_history


    def is_loop(
        self,
        history: list[str],
    ) -> bool:
        """Перевіряє, чи є 3 однакові виклики поспіль."""

        if len(history) < self.max_repeats:
            return False

        last_calls = history[
            -self.max_repeats:
        ]

        return len(set(last_calls)) == 1
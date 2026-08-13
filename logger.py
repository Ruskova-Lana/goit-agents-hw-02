import json
import time

from datetime import datetime, timezone


# ================================================================
# JSON-логування траєкторії
# ================================================================

class TrajectoryLogger:
    """Логує повну траєкторію виконання агента."""

    def __init__(self):
        # Список усіх кроків виконання
        self.steps = []

        # Час початку логування
        self.start_time = time.monotonic()


    def log_step(
        self,
        step_num: int,
        node: str,
        input_data: str,
        output_data: str,
        tool_calls: list | None = None,
        duration_ms: int = 0,
    ):
        """Додає інформацію про один крок виконання.

        Args:
            step_num: Номер кроку.
            node: Назва вузла LangGraph.
            input_data: Вхідні дані вузла.
            output_data: Результат роботи вузла.
            tool_calls: Список викликаних tools.
            duration_ms: Тривалість виконання вузла.
        """

        self.steps.append(
            {
                "step_number": step_num,

                "node_name": node,

                "input": str(
                    input_data
                )[:500],

                "output": str(
                    output_data
                )[:500],

                "tool_calls": (
                    tool_calls or []
                ),

                "timestamp": datetime.now(
                    tz=timezone.utc
                ).isoformat(),

                "duration_ms": duration_ms,

                "elapsed_ms": int(
                    (
                        time.monotonic()
                        - self.start_time
                    )
                    * 1000
                ),
            }
        )


    def save(
        self,
        filepath: str = "trajectory.json",
    ):
        """Зберігає траєкторію виконання у JSON-файл.

        Якщо trajectory.json не існує,
        Python створить його автоматично.
        """

        with open(
            filepath,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                {
                    "total_steps": len(
                        self.steps
                    ),

                    "total_time_ms": int(
                        (
                            time.monotonic()
                            - self.start_time
                        )
                        * 1000
                    ),

                    "trajectory": self.steps,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"Trajectory saved: "
            f"{filepath} "
            f"({len(self.steps)} steps)"
        )
import time
import operator

from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from tools import (
    calculate_trip_budget,
    estimate_hotel_cost,
    recommend_transport,
)

from safety import (
    MAX_STEPS,
    TIMEOUT_SECONDS,
    LoopDetector,
    max_steps_reached,
    timeout_reached,
)

from logger import TrajectoryLogger


# ================================================================
# Завантаження змінних середовища
# ================================================================

load_dotenv()


# ================================================================
# Structured output
# ================================================================

class FinalAnswer(BaseModel):
    """Структурований формат фінальної відповіді агента."""

    answer: str = Field(
        description="Фінальна відповідь користувачу."
    )

    confidence: float = Field(
        ge=0,
        le=1,
        description="Рівень впевненості у відповіді від 0 до 1."
    )

    sources: list[str] = Field(
        default_factory=list,
        description="Список використаних tools."
    )


# ================================================================
# State
# ================================================================

class AgentState(TypedDict):
    """Стан LangGraph-агента."""

    # Повна історія повідомлень.
    # operator.add додає нові повідомлення до існуючих.
    messages: Annotated[list, operator.add]

    # Кількість LLM-ітерацій.
    step_count: int

    # Час початку одного запуску агента.
    start_time: float

    # Причина примусової зупинки.
    stop_reason: str | None

    # Історія tool calls для детекції зациклення.
    tool_call_history: list[str]

    # Структурована фінальна відповідь.
    final_answer: FinalAnswer | None


# ================================================================
# Google Gemini
# ================================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.1,
)


# ================================================================
# Tools із Завдання 1
# ================================================================

tools = [
    calculate_trip_budget,
    estimate_hotel_cost,
    recommend_transport,
]


# LLM отримує інформацію про доступні tools.
llm_with_tools = llm.bind_tools(tools)


# Окрема версія LLM для structured output.
structured_llm = llm.with_structured_output(
    FinalAnswer
)


# ================================================================
# Допоміжні об'єкти
# ================================================================

# ToolNode автоматично виконує tool_calls,
# які повернув Gemini.
tool_node = ToolNode(tools)


# Детектор однакових tool calls.
loop_detector = LoopDetector()


# Logger створюється для поточного запуску програми.
logger = TrajectoryLogger()


# ================================================================
# Agent node
# ================================================================

def agent_node(state: AgentState) -> dict:
    """LLM приймає рішення: викликати tool або відповісти напряму."""

    node_start = time.monotonic()

    step = state.get(
        "step_count",
        0,
    )


    # ------------------------------------------------------------
    # SAFETY 1: max_steps
    # ------------------------------------------------------------

    if max_steps_reached(step):

        response = AIMessage(
            content=(
                f"Досягнуто максимальний ліміт "
                f"у {MAX_STEPS} LLM-кроків. "
                f"Агент зупинив виконання та повертає "
                f"часткову відповідь."
            )
        )

        duration_ms = int(
            (
                time.monotonic()
                - node_start
            )
            * 1000
        )

        logger.log_step(
            step_num=len(logger.steps) + 1,
            node="agent",
            input_data=str(
                state["messages"][-1]
            ),
            output_data=response.content,
            tool_calls=[],
            duration_ms=duration_ms,
        )

        return {
            "messages": [response],
            "step_count": step,
            "stop_reason": "max_steps",
        }


    # ------------------------------------------------------------
    # SAFETY 2: timeout
    # ------------------------------------------------------------

    if timeout_reached(
        state["start_time"]
    ):

        response = AIMessage(
            content=(
                f"Досягнуто загальний тайм-аут "
                f"виконання ({TIMEOUT_SECONDS} секунд). "
                f"Агент повертає часткову відповідь."
            )
        )

        duration_ms = int(
            (
                time.monotonic()
                - node_start
            )
            * 1000
        )

        logger.log_step(
            step_num=len(logger.steps) + 1,
            node="agent",
            input_data=str(
                state["messages"][-1]
            ),
            output_data=response.content,
            tool_calls=[],
            duration_ms=duration_ms,
        )

        return {
            "messages": [response],
            "step_count": step,
            "stop_reason": "timeout",
        }


    # ------------------------------------------------------------
    # Звичайний виклик Gemini
    # ------------------------------------------------------------

    try:
        response = llm_with_tools.invoke(
            state["messages"]
        )
    except Exception as error:

        duration_ms = int(
            (
                time.monotonic()
                - node_start
            )
            * 1000
        )

        logger.log_step(
            step_num=len(logger.steps) + 1,
            node="agent",
            input_data=str(
                state["messages"][-1]
            ),
            output_data=f"LLM error: {error}",
            tool_calls=[],
            duration_ms=duration_ms,
        )

        raise


    # Отримуємо tool calls із AIMessage.
    tool_calls = getattr(
        response,
        "tool_calls",
        [],
    )


    duration_ms = int(
        (
            time.monotonic()
            - node_start
        )
        * 1000
    )


    # ------------------------------------------------------------
    # Логування agent node
    # ------------------------------------------------------------

    logger.log_step(
        step_num=len(logger.steps) + 1,
        node="agent",
        input_data=str(
            state["messages"][-1]
        ),
        output_data=str(
            response.content
        ),
        tool_calls=tool_calls,
        duration_ms=duration_ms,
    )


    return {
        "messages": [response],
        "step_count": step + 1,
    }


# ================================================================
# Router після agent
# ================================================================

def should_continue(
    state: AgentState,
) -> Literal["tools", "formatter"]:
    """Визначає наступний вузол після agent."""

    # Якщо спрацював safety mechanism,
    # ReAct-цикл більше не продовжуємо.
    if state.get("stop_reason"):
        return "formatter"


    # Додаткова перевірка timeout.
    if timeout_reached(
        state["start_time"]
    ):
        return "formatter"


    last_message = state[
        "messages"
    ][-1]


    tool_calls = getattr(
        last_message,
        "tool_calls",
        [],
    )


    # Якщо Gemini створив tool_call,
    # переходимо до tools.
    if tool_calls:
        return "tools"


    # Якщо tool_calls немає,
    # ReAct-цикл завершено.
    return "formatter"


# ================================================================
# Safe tools node
# ================================================================

def tools_node_safe(
    state: AgentState,
) -> dict:
    """Виконує tools та перевіряє можливе зациклення."""

    node_start = time.monotonic()


    last_message = state[
        "messages"
    ][-1]


    tool_calls = getattr(
        last_message,
        "tool_calls",
        [],
    )


    # ------------------------------------------------------------
    # Оновлення історії tool calls
    # ------------------------------------------------------------

    history = loop_detector.add_calls(
        state.get(
            "tool_call_history",
            [],
        ),
        tool_calls,
    )


    # ------------------------------------------------------------
    # SAFETY 3: loop detection
    # ------------------------------------------------------------

    if loop_detector.is_loop(
        history
    ):

        message = AIMessage(
            content=(
                "Виявлено можливе зациклення: "
                "той самий tool з однаковими аргументами "
                "було викликано 3 рази поспіль. "
                "Виконання зупинено."
            )
        )


        duration_ms = int(
            (
                time.monotonic()
                - node_start
            )
            * 1000
        )


        logger.log_step(
            step_num=len(logger.steps) + 1,
            node="tools",
            input_data=str(tool_calls),
            output_data=message.content,
            tool_calls=tool_calls,
            duration_ms=duration_ms,
        )


        return {
            "messages": [message],
            "tool_call_history": history,
            "stop_reason": "loop_detected",
        }


    # ------------------------------------------------------------
    # SAFETY 2: timeout перед виконанням tool
    # ------------------------------------------------------------

    if timeout_reached(
        state["start_time"]
    ):

        message = AIMessage(
            content=(
                f"Досягнуто загальний тайм-аут "
                f"виконання ({TIMEOUT_SECONDS} секунд)."
            )
        )


        duration_ms = int(
            (
                time.monotonic()
                - node_start
            )
            * 1000
        )


        logger.log_step(
            step_num=len(logger.steps) + 1,
            node="tools",
            input_data=str(tool_calls),
            output_data=message.content,
            tool_calls=tool_calls,
            duration_ms=duration_ms,
        )


        return {
            "messages": [message],
            "tool_call_history": history,
            "stop_reason": "timeout",
        }


    # ------------------------------------------------------------
    # Виконання tools через ToolNode
    # ------------------------------------------------------------

    result = tool_node.invoke(
        state
    )


    duration_ms = int(
        (
            time.monotonic()
            - node_start
        )
        * 1000
    )


    # ------------------------------------------------------------
    # Логування tools node
    # ------------------------------------------------------------

    logger.log_step(
        step_num=len(logger.steps) + 1,
        node="tools",
        input_data=str(tool_calls),
        output_data=str(
            result["messages"]
        ),
        tool_calls=tool_calls,
        duration_ms=duration_ms,
    )


    return {
        "messages": result["messages"],
        "tool_call_history": history,
    }


# ================================================================
# Formatter node
# ================================================================

def formatter_node(
    state: AgentState,
) -> dict:
    """Перетворює фінальну відповідь у Pydantic FinalAnswer."""

    node_start = time.monotonic()


    # ------------------------------------------------------------
    # Беремо останню текстову відповідь
    # ------------------------------------------------------------

    final_text = str(
        state["messages"][-1].content
    )


    # ------------------------------------------------------------
    # Збираємо реально використані tools
    # ------------------------------------------------------------

    used_tools = []


    for message in state["messages"]:

        name = getattr(
            message,
            "name",
            None,
        )

        if (
            name
            and name not in used_tools
        ):
            used_tools.append(
                name
            )


    # ------------------------------------------------------------
    # Визначаємо, чи виконання завершилось достроково
    # ------------------------------------------------------------

    stop_reason = state.get(
        "stop_reason"
    )


    if stop_reason:

        prompt = f"""
Сформуй структуровану часткову відповідь користувачу.

Причина дострокової зупинки:
{stop_reason}

Остання доступна відповідь агента:
{final_text}

Реально використані tools:
{used_tools}

Правила:
- answer — зрозуміла часткова відповідь користувачу;
- confidence — число від 0 до 1;
- через дострокову зупинку confidence має бути нижчим;
- sources — тільки tools зі списку вище;
- не вигадуй інформацію або джерела.
"""

    else:

        prompt = f"""
Перетвори фінальну відповідь AI-агента
у структурований формат.

Фінальна відповідь:
{final_text}

Реально використані tools:
{used_tools}

Правила:
- answer — фінальна відповідь користувачу;
- confidence — число від 0 до 1;
- sources — тільки tools зі списку вище;
- не вигадуй tools або інші джерела.
"""


    # ------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------

    structured_result = (
        structured_llm.invoke(
            prompt
        )
    )


    duration_ms = int(
        (
            time.monotonic()
            - node_start
        )
        * 1000
    )


    # ------------------------------------------------------------
    # Логування formatter node
    # ------------------------------------------------------------

    logger.log_step(
        step_num=len(logger.steps) + 1,
        node="formatter",
        input_data=final_text,
        output_data=str(
            structured_result.model_dump()
        ),
        tool_calls=[],
        duration_ms=duration_ms,
    )


    return {
        "final_answer": structured_result
    }


# ================================================================
# Побудова LangGraph
# ================================================================

graph = StateGraph(
    AgentState
)


# ------------------------------------------------------------
# Nodes
# ------------------------------------------------------------

graph.add_node(
    "agent",
    agent_node,
)

graph.add_node(
    "tools",
    tools_node_safe,
)

graph.add_node(
    "formatter",
    formatter_node,
)


# ------------------------------------------------------------
# Edges
# ------------------------------------------------------------

# START → agent
graph.add_edge(
    START,
    "agent",
)


# agent → tools
# або
# agent → formatter
graph.add_conditional_edges(
    "agent",
    should_continue,
)


# tools → agent
# Це формує ReAct-цикл:
# LLM → tools → LLM
graph.add_edge(
    "tools",
    "agent",
)


# formatter → END
graph.add_edge(
    "formatter",
    END,
)


# Компіляція графа
app = graph.compile()


# ================================================================
# Запуск одного прикладу
# ================================================================

def run_example(
    user_request: str,
) -> dict:
    """Запускає LangGraph для одного запиту."""

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"USER: {user_request}"
    )

    print(
        "=" * 70
    )


    # Кожен запуск отримує власний start_time.
    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content=user_request
                )
            ],

            "step_count": 0,

            "start_time": (
                time.monotonic()
            ),

            "stop_reason": None,

            "tool_call_history": [],

            "final_answer": None,
        },

        # Додатковий safeguard самого LangGraph.
        {
            "recursion_limit": 30
        },
    )


    # ------------------------------------------------------------
    # Виведення результату
    # ------------------------------------------------------------

    print(
        "\nSTRUCTURED RESPONSE:"
    )


    print(
        result["final_answer"]
        .model_dump_json(
            indent=2
        )
    )


    print(
        f"\nLLM steps: "
        f"{result['step_count']}"
    )


    if result.get(
        "stop_reason"
    ):

        print(
            f"Stop reason: "
            f"{result['stop_reason']}"
        )


    return result


# ================================================================
# Запуск програми
# ================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------
    # Приклад 1 — бюджет
    # ------------------------------------------------------------

    run_example(
        "Я їду удвох на 5 днів. "
        "Щоденний бюджет на одну людину — 80 євро. "
        "Порахуй загальний бюджет подорожі."
    )


    try:

        # ------------------------------------------------------------
        # Приклад 2 — готель
        # ------------------------------------------------------------

        run_example(
            "Я хочу забронювати 2 номери "
            "на 4 ночі. "
            "Один номер коштує 120 євро за ніч. "
            "Скільки коштуватиме проживання?"
        )


        # ------------------------------------------------------------
        # Приклад 3 — транспорт
        # ------------------------------------------------------------

        run_example(
            "Мені потрібно подолати 500 км. "
            "Подорожують 2 людини. "
            "Для мене найважливіша швидкість. "
            "Який транспорт краще обрати?"
        )

    except Exception as error:

        print("\nПомилка під час виконання агента:")
        print(error)


    finally:

        # --------------------------------------------------------
        # trajectory.json створюється навіть при помилці
        # --------------------------------------------------------

        logger.save(
            "trajectory.json"
        )
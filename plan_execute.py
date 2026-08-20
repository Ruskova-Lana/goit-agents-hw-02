import operator
import sqlite3
import sys

from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from tools import (
    calculate_trip_budget,
    estimate_hotel_cost,
    recommend_transport,
)


# ================================================================
# Завантаження змінних середовища
# ================================================================

load_dotenv()


# ================================================================
# Structured Output Models
# ================================================================

class Plan(BaseModel):
    """Структурований план виконання задачі."""

    goal: str = Field(
        description="Головна ціль користувацького запиту."
    )

    steps: list[str] = Field(
        description=(
            "Послідовний список конкретних кроків "
            "для досягнення цілі."
        )
    )


class ReplanDecision(BaseModel):
    """Рішення replanner після виконання чергового кроку."""

    action: Literal[
        "continue",
        "replan",
        "finish",
    ] = Field(
        description=(
            "continue — виконати наступний крок; "
            "replan — змінити залишковий план; "
            "finish — завершити виконання."
        )
    )

    updated_steps: list[str] | None = Field(
        default=None,
        description=(
            "Новий список невиконаних кроків, "
            "якщо action = replan."
        )
    )

    reasoning: str = Field(
        description="Коротке пояснення рішення replanner."
    )


# ================================================================
# State
# ================================================================

class PlanExecuteState(TypedDict):
    """Стан Plan-and-Execute агента."""

    # Історія повідомлень
    messages: Annotated[list, operator.add]

    # Список кроків плану
    plan: list[str]

    # Номер поточного кроку, 0-indexed
    current_step: int

    # Результати вже виконаних кроків
    results: list[str]

    # Ознака завершення workflow
    completed: bool

    # Для демонстрації persistence
    pause_after_first_step: bool


# ================================================================
# Google Gemini
# ================================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.1,
)


# ================================================================
# Structured LLMs
# ================================================================

planner_llm = llm.with_structured_output(
    Plan
)

replanner_llm = llm.with_structured_output(
    ReplanDecision
)


# ================================================================
# Tools
# ================================================================

tools = [
    calculate_trip_budget,
    estimate_hotel_cost,
    recommend_transport,
]


tools_by_name = {
    tool.name: tool
    for tool in tools
}


# Executor отримує можливість самостійно вибрати tool
executor_llm = llm.bind_tools(
    tools
)


# ================================================================
# Planner Node
# ================================================================

def planner_node(
    state: PlanExecuteState,
) -> dict:
    """Створює структурований план виконання запиту."""

    user_message = state["messages"][0]

    user_query = getattr(
        user_message,
        "content",
        str(user_message),
    )


    prompt = f"""
Ти Planner туристичного AI-агента.

Запит користувача:
{user_query}

Створи структурований план виконання задачі.

Доступні tools:
1. calculate_trip_budget
   Використовується для розрахунку бюджету подорожі.

2. estimate_hotel_cost
   Використовується для розрахунку вартості проживання.

3. recommend_transport
   Використовується для вибору транспорту.

Правила:
- сформуй від 1 до 5 конкретних кроків;
- кожен крок повинен виконувати одну логічну дію;
- в описі кроку вкажи, який tool очікується використати;
- не виконуй розрахунки самостійно;
- не додавай непотрібні кроки.
"""


    plan_result = planner_llm.invoke(
        prompt
    )


    print("\n" + "=" * 70)
    print("PLANNER")
    print("=" * 70)

    print(
        f"Goal: {plan_result.goal}"
    )

    for index, step in enumerate(
        plan_result.steps,
        start=1,
    ):
        print(
            f"{index}. {step}"
        )


    return {
        "plan": plan_result.steps,
        "current_step": 0,
        "results": [],
        "completed": False,

        "messages": [
            AIMessage(
                content=(
                    f"Створено план: "
                    f"{plan_result.steps}"
                )
            )
        ],
    }


# ================================================================
# Executor Node
# ================================================================

def executor_node(
    state: PlanExecuteState,
) -> dict:
    """Виконує рівно один поточний крок плану."""

    step_index = state[
        "current_step"
    ]

    plan = state[
        "plan"
    ]


    # Якщо всі кроки вже виконані
    if step_index >= len(plan):

        return {
            "completed": True
        }


    current_step = plan[
        step_index
    ]


    print("\n" + "-" * 70)
    print(
        f"EXECUTOR — STEP {step_index + 1}"
    )
    print("-" * 70)

    print(
        f"Current step: {current_step}"
    )


    prompt = f"""
Ти Executor туристичного AI-агента.

Виконай ТІЛЬКИ один поточний крок.

Поточний крок:
{current_step}

Попередні результати:
{state["results"]}

Використовуй один із доступних tools, якщо він потрібний.

Не переходь до наступного кроку.
Не вигадуй результат tool.
"""


    response = executor_llm.invoke(
        prompt
    )


    # ------------------------------------------------------------
    # Якщо LLM повернув tool call
    # ------------------------------------------------------------

    tool_calls = getattr(
        response,
        "tool_calls",
        [],
    )


    if tool_calls:

        step_results = []


        for tool_call in tool_calls:

            tool_name = tool_call.get(
                "name"
            )

            tool_args = tool_call.get(
                "args",
                {},
            )


            tool_function = tools_by_name.get(
                tool_name
            )


            if tool_function is None:

                step_results.append(
                    f"Невідомий tool: {tool_name}"
                )

                continue


            tool_result = tool_function.invoke(
                tool_args
            )


            step_results.append(
                f"{tool_name}: {tool_result}"
            )


        result_text = "\n".join(
            step_results
        )


    # ------------------------------------------------------------
    # Якщо tool не потрібний
    # ------------------------------------------------------------

    else:

        result_text = str(
            response.content
        )


    print(
        f"Result: {result_text}"
    )


    updated_results = [
        *state["results"],

        (
            f"Крок {step_index + 1}: "
            f"{result_text}"
        ),
    ]


    return {
        "current_step": step_index + 1,

        "results": updated_results,

        "messages": [
            AIMessage(
                content=(
                    f"Виконано крок "
                    f"{step_index + 1}: "
                    f"{result_text}"
                )
            )
        ],
    }


# ================================================================
# Pause Node
# ================================================================
def pause_demo_node(
    state: PlanExecuteState,
) -> dict:
    """Навмисно зупиняє workflow після першого виконаного кроку."""

    print("\n" + "=" * 70)
    print("CHECKPOINT DEMO")
    print("=" * 70)

    print(
        f"Workflow зупинено після кроку "
        f"{state['current_step']}."
    )

    print(
        "Стан вже збережено у agent_state.db."
    )

    return {}


# ================================================================
# Replanner Node
# ================================================================

def replanner_node(
    state: PlanExecuteState,
) -> dict:
    """Аналізує прогрес і вирішує, що робити далі."""

    plan = state[
        "plan"
    ]

    current_step = state[
        "current_step"
    ]

    results = state[
        "results"
    ]


    remaining_steps = plan[
        current_step:
    ]


    print("\n" + "-" * 70)
    print("REPLANNER")
    print("-" * 70)


    # ------------------------------------------------------------
    # Якщо всі заплановані кроки вже виконані
    # ------------------------------------------------------------

    if current_step >= len(plan):

        print(
            "Усі кроки виконано → finish"
        )

        return {
            "completed": True,

            "messages": [
                AIMessage(
                    content=(
                        "Усі кроки плану виконано."
                    )
                )
            ],
        }


    prompt = f"""
Ти Replanner туристичного AI-агента.

Оціни прогрес виконання задачі.

Початковий план:
{plan}

Виконано кроків:
{current_step} із {len(plan)}

Результати виконаних кроків:
{results}

Залишкові кроки:
{remaining_steps}

Прийми одне рішення:

continue
- якщо план правильний;
- треба просто виконати наступний крок.

replan
- якщо отримані результати показують,
  що залишковий план потрібно змінити;
- updated_steps повинен містити
  тільки нові невиконані кроки.

finish
- якщо ціль уже досягнута
  і подальші кроки не потрібні.
"""


    decision = replanner_llm.invoke(
        prompt
    )


    print(
        f"Decision: {decision.action}"
    )

    print(
        f"Reasoning: {decision.reasoning}"
    )


    # ------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------

    if decision.action == "finish":

        return {
            "completed": True,

            "messages": [
                AIMessage(
                    content=(
                        f"Завершено: "
                        f"{decision.reasoning}"
                    )
                )
            ],
        }


    # ------------------------------------------------------------
    # Replan
    # ------------------------------------------------------------

    if (
        decision.action == "replan"
        and decision.updated_steps
    ):

        print(
            "Updated plan:"
        )

        for index, step in enumerate(
            decision.updated_steps,
            start=1,
        ):
            print(
                f"{index}. {step}"
            )


        return {
            # Новий plan містить тільки невиконані кроки
            "plan": decision.updated_steps,

            # Починаємо новий залишковий план з 0
            "current_step": 0,

            "messages": [
                AIMessage(
                    content=(
                        "План було оновлено."
                    )
                )
            ],
        }


    # ------------------------------------------------------------
    # Continue
    # ------------------------------------------------------------

    return {
        "messages": [
            AIMessage(
                content=(
                    f"Продовжуємо план. "
                    f"{decision.reasoning}"
                )
            )
        ]
    }


# ================================================================
# Router
# ================================================================

def should_end(
    state: PlanExecuteState,
) -> Literal[
    "executor",
    "__end__",
]:
    """Визначає, чи потрібно продовжити workflow."""

    if state.get(
        "completed",
        False,
    ):

        return "__end__"


    return "executor"


# ================================================================
# LangGraph
# ================================================================

graph = StateGraph(
    PlanExecuteState
)


# ------------------------------------------------------------
# Nodes
# ------------------------------------------------------------

graph.add_node(
    "planner",
    planner_node,
)

graph.add_node(
    "executor",
    executor_node,
)

graph.add_node(
    "replanner",
    replanner_node,
)


# ------------------------------------------------------------
# Edges
# ------------------------------------------------------------

graph.add_edge(
    START,
    "planner",
)

graph.add_edge(
    "planner",
    "executor",
)

graph.add_edge(
    "executor",
    "replanner",
)

graph.add_conditional_edges(
    "replanner",
    should_end,
)


# ================================================================
# Компіляція
# ================================================================

conn = sqlite3.connect(
    "agent_state.db",
    check_same_thread=False,
)

saver = SqliteSaver(conn)

app = graph.compile(
    checkpointer=saver
)


# ================================================================
# Запуск прикладу
# ================================================================

def run_example(
    title: str,
    query: str,
) -> None:
    """Запускає один демонстраційний сценарій."""

    print(
        "\n\n"
        + "#" * 80
    )

    print(
        title
    )

    print(
        "#" * 80
    )

    print(
        f"USER: {query}"
    )


    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content=query
                )
            ],

            "plan": [],

            "current_step": 0,

            "results": [],

            "completed": False,
        },

        # Захист від випадкового нескінченного циклу
        {
            "recursion_limit": 30
        },
    )


    print("\n" + "=" * 70)
    print("FINAL STATE")
    print("=" * 70)


    print(
        f"Plan: {result['plan']}"
    )

    print(
        f"Current step: "
        f"{result['current_step']}"
    )

    print(
        f"Completed: "
        f"{result['completed']}"
    )


    print(
        "\nResults:"
    )

    for item in result[
        "results"
    ]:

        print(
            f"- {item}"
        )


# ================================================================
# Демонстрація на 3 прикладах
# ================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------
    # SIMPLE
    # Один tool
    # ------------------------------------------------------------

    run_example(
        "EXAMPLE 1 — SIMPLE",

        (
            "Я їду удвох на 5 днів. "
            "Щоденний бюджет на одну людину — 80 євро. "
            "Порахуй загальний бюджет подорожі."
        ),
    )


    # ------------------------------------------------------------
    # MEDIUM
    # Два tools
    # ------------------------------------------------------------

    run_example(
        "EXAMPLE 2 — MEDIUM",

        (
            "Я планую подорож для двох людей на 4 дні. "
            "Щоденний бюджет на одну людину — 70 євро. "
            "Також потрібен один номер у готелі "
            "на 4 ночі по 100 євро за ніч. "
            "Порахуй бюджет подорожі та вартість готелю."
        ),
    )


    # ------------------------------------------------------------
    # COMPLEX
    # Три tools і декілька кроків
    # ------------------------------------------------------------

    run_example(
        "EXAMPLE 3 — COMPLEX",

        (
            "Допоможи спланувати подорож для двох людей "
            "на 7 днів. "
            "Відстань — 1200 км, "
            "головний пріоритет — швидкість. "
            "Щоденний бюджет на одну людину — 75 євро. "
            "Також потрібен один номер на 6 ночей "
            "по 110 євро за ніч. "
            "Порекомендуй транспорт, "
            "порахуй бюджет подорожі "
            "та вартість проживання."
        ),
    )
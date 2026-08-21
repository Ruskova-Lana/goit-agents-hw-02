# Plan-and-Execute агент для планування подорожей

## 1. Опис проєкту

Туристичний AI-агент, побудований за архітектурою **Plan-and-Execute** на базі LangGraph.
Замість того щоб на кожному кроці заново вирішувати, що робити (як у ReAct), агент спочатку
будує повний план виконання задачі, а потім послідовно виконує його кроки, за потреби
коригуючи план (**replanning**) на основі проміжних результатів.

Проєкт демонструє чотири окремі можливості:

1. **Plan-and-Execute** — planner будує план, executor виконує кроки, replanner вирішує
   `continue` / `replan` / `finish`.
2. **Checkpointer (persistence)** — стан графа зберігається у SQLite (`agent_state.db`) і
   переживає перезапуск Python-процесу.
3. **Agentic RAG** — агент сам вирішує, коли йому потрібна довідкова інформація з
   внутрішньої бази знань (ChromaDB), а коли достатньо звичайних tools.
4. **Human-in-the-Loop (HITL)** — ризикові дії (бронювання готелю) виконуються лише після
   явного підтвердження людини через `interrupt()` / `Command(resume=...)`.

## 2. Архітектура

```
START → planner → executor ─┬─→ approval ─────┐
                             ├─→ checkpoint_pause ─┤
                             └─→ replanner ◄───────┘
                                    │
                        continue/replan → executor
                        finish        → END
```

* **planner** — на основі запиту користувача формує структурований `Plan`
  (`goal` + список кроків) через `with_structured_output`.
* **executor** — виконує рівно один крок поточного плану: обирає tool
  (`bind_tools`), викликає його або, якщо tool ризиковий, зупиняється перед
  виконанням.
* **approval** — `interrupt()`-вузол, що очікує рішення людини:
  `approve` / `reject` / `edit` — для tools зі списку `RISKY_TOOLS`.
* **checkpoint_pause** — навмисний `interrupt()` після першого кроку, що
  демонструє відновлення стану з SQLite в новому процесі.
* **replanner** — після кожного кроку вирішує: продовжити виконання поточного
  плану (`continue`), змінити залишкові кроки (`replan`) чи завершити
  (`finish`).

Стан графа (`PlanExecuteState`) зберігається через `SqliteSaver` у файл
`agent_state.db`, тому кожен `thread_id` має власну, незалежну історію.

## 3. Структура файлів

```
tools.py              # Звичайні tools з Pydantic-схемами та валідацією
                       # (calculate_trip_budget, estimate_hotel_cost, recommend_transport)
knowledge.py           # ChromaDB + tool search_knowledge (Agentic RAG)
hitl.py                # Ризиковий tool book_hotel + Pydantic-схема
plan_execute.py        # LangGraph: planner + executor + replanner + HITL + CLI
agent_state.db         # SQLite зі збереженим станом (генерується автоматично)
chroma_db/             # Локальна векторна база ChromaDB (генерується автоматично)
requirements.txt       # Залежності Python
README.md              # Цей файл
```

## 4. Встановлення

```bash
pip install -r requirements.txt
```

Створіть файл `.env` у корені проєкту:

```
GOOGLE_API_KEY=ваш_ключ_google_generative_ai
```

Модель за замовчуванням — `gemini-3.5-flash-lite` (задається в `plan_execute.py`).

## 5. Інструкція запуску

Усі сценарії запускаються через `plan_execute.py`. Список команд також
доступний за допомогою `python plan_execute.py` (без аргументів).

### Завдання 1 — Plan-and-Execute

```bash
python plan_execute.py simple    # один крок плану
python plan_execute.py medium    # кілька tools
python plan_execute.py complex   # повний сценарій подорожі
python plan_execute.py demo      # усі три приклади підряд
```

### Завдання 2 — Checkpointer (persistence)

```bash
python plan_execute.py start     # запускає workflow і зупиняє його
                                  # після першого виконаного кроку
python plan_execute.py resume    # у НОВОМУ Python-процесі відновлює
                                  # той самий thread_id з agent_state.db
python plan_execute.py threads   # показує, що різні thread_id мають
                                  # незалежний стан
```

### Завдання 3 — Agentic RAG

```bash
python plan_execute.py rag
```

Запускає три приклади: запит, де `search_knowledge` не потрібен; запит, де
він потрібен; і запит, що поєднує звичайний tool із пошуком у базі знань.
Вибір tool повністю залишається за LLM.

### Завдання 4 — Human-in-the-Loop

```bash
python plan_execute.py hitl hitl-approve-001
python plan_execute.py approve hitl-approve-001

python plan_execute.py hitl hitl-reject-001
python plan_execute.py reject hitl-reject-001

python plan_execute.py hitl hitl-edit-001
python plan_execute.py edit hitl-edit-001
```

Кожен сценарій запускається з власним `thread_id`: перша команда доводить
graph до `interrupt()` перед `book_hotel`, друга — відновлює виконання з
відповідним рішенням людини (підтвердити, відхилити або змінити параметри
бронювання).

## 6. Tools

| Tool | Призначення | Ризиковий |
|---|---|---|
| `calculate_trip_budget` | Розрахунок загального бюджету подорожі | ні |
| `estimate_hotel_cost` | Розрахунок вартості проживання | ні |
| `recommend_transport` | Рекомендація транспорту за відстанню та пріоритетом | ні |
| `search_knowledge` | Agentic RAG-пошук у ChromaDB (страхування, документи, багаж, правила) | ні |
| `book_hotel` | Фактичне бронювання готелю | **так — потребує HITL approval** |

Усі tools мають Pydantic `args_schema` з валідаторами (діапазони значень,
формат дати, довжина рядків тощо). `travelers` (кількість мандрівників)
валідується спільним `Annotated`-типом `TravelersCount`, який використовують
одразу `TripBudgetInput` і `TransportInput`, щоб не дублювати логіку.

## 7. База знань (ChromaDB)

`knowledge.py` створює локальну, персистентну колекцію ChromaDB
(`./chroma_db`, колекція `travel_knowledge`) і заповнює її 10 короткими
документами доменної області "подорожі". Кожен документ має `id`, `title` та
текст; `initialize_knowledge_base()` додає лише ті документи, яких ще немає
в колекції, тому повторний імпорт `knowledge.py` не створює дублікатів.

| id | title |
|---|---|
| travel-001 | Travel insurance |
| travel-002 | Airport arrival |
| travel-003 | Cabin baggage |
| travel-004 | Hotel check-in |
| travel-005 | Emergency budget |
| travel-006 | Train travel |
| travel-007 | Flight travel |
| travel-008 | Travel documents |
| travel-009 | Hotel cancellation |
| travel-010 | Local public transport |

`search_knowledge` (Pydantic-схема `KnowledgeSearchInput`: `query` мінімум
3 символи, `top_k` від 1 до 5) виконує `collection.query()` за
семантичною близькістю і повертає `top_k` найрелевантніших фрагментів у
форматі `"{title}: {text}"`. Executor викликає цей tool лише тоді, коли
LLM сам вирішує, що для поточного кроку плану потрібна довідкова
інформація, а не розрахунок (детальніше — розділ 9).

## 8. Ризиковий tool та HITL flow

`book_hotel` (`hitl.py`) — єдиний tool у списку `RISKY_TOOLS`
(`plan_execute.py`). Коли executor обирає ризиковий tool, він **не викликає
його одразу**, а зберігає `pending_tool_call` у стані графа і передає
керування вузлу `approval`.

Вузол `approval` викликає `interrupt()` з деталями дії:

```python
{
    "type": "approval_required",
    "message": "Потрібне підтвердження ризикової дії.",
    "tool": "book_hotel",
    "args": {"hotel_name": "...", "check_in": "YYYY-MM-DD", "nights": 4, "total_cost": 400},
    "allowed_actions": ["approve", "reject", "edit"],
    "instructions": {...},
}
```

Граф зупиняється тут; процес можна навіть завершити — стан збережеться в
`agent_state.db`. Продовження відбувається через
`app.invoke(Command(resume={...}), config=make_config(thread_id))` з тим
самим `thread_id`, залежно від рішення людини:

* **approve** — `book_hotel.invoke(original_args)` виконується без змін;
  результат (з `Booking ID`) додається до `results`, `replanner` зазвичай
  завершує задачу (`finish`).
* **reject** — tool **не викликається**; замість цього в `results`
  записується повідомлення про відмову (за наявності — з причиною
  `reason`). `replanner` бачить, що ризикову дію скасовано, і завершує
  виконання (`finish`), не намагаючись повторити `book_hotel`.
* **edit** — людина передає нові `args` у `Command(resume=...)`;
  `book_hotel.invoke(edited_args)` виконується з оновленими параметрами
  (Pydantic-валідація `HotelBookingInput` спрацьовує повторно всередині
  `tool.invoke()`).

## 9. Аналіз результатів

Нижче — спостереження з реальних запусків кожного сценарію (без
редагування чи вигаданих цифр).

**Plan-and-Execute.** У simple-прикладі planner коректно згенерував план з
одного кроку і executor одразу обрав `calculate_trip_budget` з правильними
аргументами (`{"travelers": 2, "days": 5, "daily_budget": 80}` → `€800.00`).
У medium/complex-прикладах planner будує 2-3 кроки, а executor послідовно
викликає `recommend_transport` → `estimate_hotel_cost` → `calculate_trip_budget`,
не забігаючи наперед (одна дія за одну ітерацію).

**Checkpointer.** Команда `start` зупинила workflow одразу після кроку 1
(`recommend_transport`) через `checkpoint_pause`. У **новому** Python-процесі
команда `resume` прочитала стан із `agent_state.db` (`current_step=1`, план і
результати кроку 1 — незмінні) і продовжила виконання: `replanner` обрав
`continue`, executor виконав кроки 2 і 3, а фінальний `results` містив усі
три кроки в правильному порядку. Команда `threads` підтвердила ізоляцію:
`checkpoint-session-001` містив повний стан, `checkpoint-session-002` (інший
`thread_id`) — порожній `{}`.

**Agentic RAG.** На запиті "порахуй бюджет" (без згадки довідкової
інформації) агент викликав лише `calculate_trip_budget` і жодного разу не
торкнувся `search_knowledge`. На запиті "що перевірити перед міжнародною
подорожжю" — навпаки, викликав лише `search_knowledge` і повернув релевантні
документи (`Travel documents`, `Travel insurance`, `Cabin baggage`). На
комбінованому запиті ("порахуй бюджет і скажи, що перевірити") planner
самостійно склав план із двох кроків і executor викликав обидва tools у
правильному порядку. Вибір жодного разу не потребував додаткових підказок
у коді — лише опис tools у промпті planner/executor.

**HITL.** У approve-сценарії `book_hotel` виконався тільки після
`Command(resume={"action": "approve"})` і повернув `Booking ID:
DEMO-BOOKING-001`. У reject-сценарії `book_hotel` **не викликався** —
`results` містив рядок "Ризикову дію відхилено користувачем. Причина:
...", а `replanner` після цього одразу прийняв рішення `finish`, коректно
розпізнавши, що повторювати відхилену дію не потрібно.

## 10. Відомі обмеження

* Tools використовують локальні правила та розрахунки замість реальних travel API.
* `recommend_transport` базується на спрощених правилах відстані/пріоритету.
* Робота агента залежить від доступності та квот Google Gemini API.
* `search_knowledge` працює на невеликому, вручну підготовленому наборі документів.

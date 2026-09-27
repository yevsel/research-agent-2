# Research Agent 2

A multi-agent research pipeline built with [LangGraph](https://langchain-ai.github.io/langgraph/). Give it a topic, and it generates a panel of AI "reporters" (each with their own angle), lets you review/approve them, then runs each reporter through a real web-search-backed interview with an "expert," and finally compiles all their findings into one written report — complete with introduction, conclusion, and cited sources.

## What it actually does

1. **Generate reporters** — given a topic, an LLM proposes a small panel of reporters, each with a distinct profile (name, institution, role, focus area)
2. **Human review (human-in-the-loop)** — the reporter panel is presented back to you. Approve it, or give feedback to regenerate
3. **Parallel interviews** — once approved, every reporter independently "interviews" an expert:
   - the reporter asks a question based on their profile
   - the question is turned into a web search query and run against [Tavily](https://tavily.com/)
   - an "expert" answers, grounded only in those real search results, with citations
   - this repeats for a fixed number of turns, with each new question built on the growing conversation
   - the full back-and-forth is saved as a transcript, and turned into a written report section

   > **What "expert" actually means here:** there's no separate expert system or database. Both the reporter and the expert are the *same* LLM, just given different roles for that one call. The reporter is free to ask anything, driven by its assigned angle. The expert is constrained by its instructions to answer using *only* the real web search results it was just handed for that turn — not its own trained knowledge — and to cite which source backs each claim. That constraint is what keeps the interview grounded in real information instead of the model just making things up.
4. **Report assembly** — once every reporter's interview is done, their sections are merged, and separate LLM calls write a unifying introduction, a consolidated body, and a conclusion
5. **Final report** — introduction + body + conclusion + a deduplicated source list, combined into one final markdown document

All reporter interviews run **in parallel**, not one after another — the number of reporters is only known at runtime, so this uses LangGraph's `Send` API to fan out dynamically.

## Why it's built this way

This isn't an open-ended chatbot graph that loops forever waiting on user input. It's a **bounded pipeline**: a fixed number of steps that runs once, produces one final artifact (the report), and stops. The only place a human is involved is the one approval checkpoint after reporters are generated.

## Architecture

The project is three separate LangGraph graphs, composed together:

| Graph | File | Role |
|---|---|---|
| `create_reporter` | `src/create_reporter_graph.py` | Generates reporters + human approval loop |
| `answer_reporter_question` | `src/answer_reporter_question_graph.py` | One reporter's full interview (question → search → answer → loop → write section) |
| `research_graph` | `src/research_graph.py` | The outer graph — wires the two above together, fans out interviews in parallel, assembles the final report |

### Outer graph (`research_graph`)

```mermaid
flowchart TD
    START([START]) --> create_reporter
    create_reporter --> human_feedback

    human_feedback -->|feedback given| create_reporter
    human_feedback -->|approved: Send one per reporter| conduct_reporter_interview

    subgraph conduct_reporter_interview[" conduct_reporter_interview (subgraph, one run per reporter, in parallel) "]
        direction TD
        iSTART([start]) --> generate_reporter_question
        generate_reporter_question --> web_research
        generate_reporter_question --> web_research_2
        web_research --> generate_expert_answer_from_research
        web_research_2 --> generate_expert_answer_from_research
        generate_expert_answer_from_research -->|turns remaining| generate_reporter_question
        generate_expert_answer_from_research -->|max turns reached| save_interview
        save_interview --> write_section
    end

    conduct_reporter_interview --> write_report
    conduct_reporter_interview --> write_introduction
    conduct_reporter_interview --> write_conclusion

    write_report --> finalize_report
    write_introduction --> finalize_report
    write_conclusion --> finalize_report
    finalize_report --> END([END])
```

**How to read the fan-out:** after `human_feedback` approves the panel, `initiate_all_interviews` (an edge function) returns one `Send("conduct_reporter_interview", {...})` per approved reporter — not a single next node. LangGraph spins up one independent, parallel run of the interview subgraph per reporter, each seeded with just that reporter's profile. When every parallel run finishes, their individual report sections are merged back into one shared list (`sections`) via a LangGraph reducer, which is what `write_report`/`write_introduction`/`write_conclusion` then read from.

## Understanding `Send()` — the part that confuses everyone at first

Normal LangGraph edges only work when you know, ahead of time, exactly what the next step is. `add_edge("a", "b")` always goes from `a` to `b`. `add_conditional_edges` picks between a *fixed, known list* of possible next steps, like `["save_interview", "generate_reporter_question"]` — the graph author has to write out every possible destination by name when building the graph.

But this project has a problem normal edges can't solve: **you don't know how many reporters there will be until the program is actually running.** Could be 2, could be 5 — `max_reporters` is a number the user types in. There's no way to write `add_edge("human_feedback", "interview_reporter_1")`, `add_edge("human_feedback", "interview_reporter_2")`, `add_edge("human_feedback", "interview_reporter_3")` ahead of time, because you don't know if there will be 3 reporters or 7.

`Send()` exists exactly for this situation: "run this same node, N times, once per item in a list I only know about right now, while the graph is running" — instead of routing to one fixed destination, you get to spin up as many parallel copies of a step as you need, decided on the fly.

### A scenario, traced through with real numbers

Say the user runs the graph with `topic = "AI in agriculture"` and `max_reporters = 3`. After `create_reporter` and `human_feedback` (approved), the state now holds:

```python
state["reporters"] = [Reporter("Amara"), Reporter("Kwame"), Reporter("Yaw")]
```

The conditional edge function, `initiate_all_interviews`, runs next. Since there's no feedback (approved), it goes into the loop:

```python
for reporter in state["reporters"]:
    send_list.append(Send("conduct_reporter_interview", {"reporter": reporter, ...}))
```

With exactly 3 reporters in the list, this loop runs 3 times, building:

```python
send_list = [
    Send("conduct_reporter_interview", {"reporter": Amara, ...}),
    Send("conduct_reporter_interview", {"reporter": Kwame, ...}),
    Send("conduct_reporter_interview", {"reporter": Yaw, ...}),
]
```

`initiate_all_interviews` returns this list of 3 `Send()` objects. LangGraph sees a list of `Send()`s (instead of a plain string like `"some_node_name"`) and reacts differently: it doesn't run `"conduct_reporter_interview"` once — it launches **3 separate, simultaneous runs** of it, one per `Send()` in the list.

- Run 1 starts with `state["reporter"] = Amara`, and its own private, empty `ReporterInterviewState` otherwise
- Run 2 starts with `state["reporter"] = Kwame`, completely separate from Run 1's state
- Run 3 starts with `state["reporter"] = Yaw`, completely separate from the other two

Each run goes through the entire interview subgraph independently — its own questions, its own web searches, its own answers, its own turn-counting loop — with zero awareness that the other two runs even exist. If tomorrow the user ran this with `max_reporters = 6`, the exact same code would launch 6 parallel runs instead, with no changes needed — that's the whole point: the *number* of parallel branches is a runtime decision, not something baked into the graph's structure at build time.

When all 3 (or 6, or however many) finish, each contributes its own one-item `sections` list back up to the parent graph's `sections` field — and because that field is `Annotated[list, operator.add]` (explained further down), those all get merged into one combined list of 3 sections, instead of the last one finishing overwriting the others.

### The one-line version

`Send(node_name, starting_state)` says: *"start a brand new, independent run of `node_name`, seeded with exactly this state — and I might call this many times in one go, once for each thing in a list I only know about right now."* That's it. Everything else about `Send()` is just consequences of that one idea.

## A landmine we actually triggered: field names have to match, everywhere they're used

While building this, we deliberately broke a small standalone test file (`src/misc/map_reduce_send.py`) on purpose to see exactly what happens when field names don't line up. Everything below is something we actually ran and saw happen — not a guess.

The test file has two "notebooks" (state schemas):

```python
class OverallState(TypedDict):          # the graph's real, shared state
    topic: str
    subtopics: List[str]
    research_results: Annotated[List[str], add]   # <- has a reducer
    final_report: str

class ResearchState(TypedDict):         # the type hint on the Send()-targeted node
    subtopic: str
    research_results: List[str]         # <- no reducer here
```

We tried three different mistakes, one at a time, and got three different results:

**Mistake 1 — typo a field name inside `ResearchState` only.**
We renamed it to `reseerch_results_TYPO`, ran the graph, and it worked perfectly, no error at all. Why: `ResearchState` is only used as a type hint on one node function (`def research_subtopic(state: ResearchState)`). We assumed this meant LangGraph ignores it completely — that assumption turned out to be *wrong*, see Mistake 3 below. In this specific case it didn't matter only because nothing else in the graph referenced that exact (typo'd) name.

**Mistake 2 — typo the key a node *returns*, on a field that has a reducer.**
`research_subtopic` returned `{"research_resultsss_TYPO": [...]}` instead of `{"research_results": [...]}`. No crash. The graph finished, but `research_results` stayed empty — the real data just vanished, silently. This is the dangerous case: it looks like success, but the output is wrong and nothing tells you why.

**Mistake 3 — the real one we hit by accident: same field name in two schemas, only one has a reducer.**
We renamed `OverallState`'s field to `research_resultsP` (so it no longer matched `ResearchState.research_results`), while `research_subtopic` still returned the plain key `"research_results"`. This time it crashed:

```
langgraph.errors.InvalidUpdateError: At key 'research_results': Can receive only one value per step.
Use an Annotated key to handle multiple values.
```

What we confirmed by testing (not assuming): LangGraph *does* look at a node function's type hint (`ResearchState` here) to discover extra fields, even though that schema is never passed to `StateGraph(...)` directly. Because `ResearchState.research_results` has no reducer, and this test runs `research_subtopic` **3 times in parallel** (once per subtopic), all 3 tried to write to that same un-reduced field in the same step. A field with no reducer can only accept one write per step — three parallel writes to it is exactly what broke.

We proved this by removing the `ResearchState` type hint entirely and re-running — the crash disappeared, and it went back to silently losing data instead (same as Mistake 2). That side-by-side comparison is what confirmed the type hint really was the cause, not a guess.

### Who actually establishes the relationship between the two schemas?

We went and read LangGraph's own source for this rather than guess (`langgraph/graph/state.py`, `StateGraph._add_schema()`). The real mechanism:

**There is one shared dictionary, `self.channels`, that spans the entire graph builder.** Every schema referenced anywhere gets scanned into it — not just the schema passed to `StateGraph(OverallState)`, but also **every node's function parameter type hint**, automatically inferred the moment you call `add_node()`. `ResearchState` never gets passed to `StateGraph(...)` anywhere in this file, but LangGraph still picks it up this way.

The actual registration logic, from the source:
```python
for key, channel in channels.items():
    if key in self.channels:
        if self.channels[key] != channel:
            if isinstance(channel, LastValue):
                pass   # silently keep the one already registered
            else:
                raise ValueError(f"Channel '{key}' already exists with a different type")
    else:
        self.channels[key] = channel
```

So the "relationship" between `OverallState.research_results` and `ResearchState.research_results` isn't a real link at all — it's two schemas independently using the exact same string, `"research_results"`, as a key into one shared pool. No explicit wiring happens anywhere. It's exactly like two people writing to a variable with the same name.

**Order matters, and this is what explains every result we saw:**

1. `builder = StateGraph(OverallState)` registers its schema immediately, at construction time — before any node is added. This registers `research_results` as a `BinaryOperatorAggregate` (the reducer channel) first.
2. When `builder.add_node("research_subtopic", research_subtopic)` runs later, LangGraph infers `ResearchState` from the function's type hint and tries to register its fields too — including `research_results` again.
3. Since `"research_results"` is *already* registered (from step 1, as a reducer channel), and `ResearchState`'s version has no reducer (so it would default to a plain `LastValue` channel), the `isinstance(channel, LastValue)` check catches the conflict and **silently keeps the original, already-registered channel**. `ResearchState`'s version is dropped without a word.
4. That's exactly why the original working version of this file never crashed — `OverallState` won because it was registered first.
5. In the broken version (Mistake 3), `OverallState`'s field had been renamed to `research_resultsP`, so `"research_results"` was never registered by `OverallState` at all. The *only* registration of that name left was `ResearchState`'s reducer-less version — so *that* became the real channel: a plain `LastValue`, which can only accept one write per step. Three parallel writes landing on it in the same step is exactly what triggered the crash.

**Direct answer:** nobody explicitly links the two schemas. LangGraph builds one shared name → channel dictionary by scanning every schema it can find — the graph's main state, plus every node's own type hint — and whichever schema gets scanned first for a given field name wins. A later, conflicting `LastValue`-style declaration for the same name is silently dropped rather than erroring, which is exactly what let this bug hide until the field names stopped lining up.

### The actual rule, plain English

Every field name a node touches — whether it's on the graph's main state, or a `Send()` payload's own private-looking state — is really just a name written into one shared pool of fields LangGraph tracks. There's no real separation between "public" and "private" schema just because you wrote two different `TypedDict`s. If the same field name shows up in more than one place:

- **If every place agrees on the same reducer (or no reducer, single-writer):** fine, works as expected
- **If the names don't match anywhere real is watching:** silent data loss — the value just disappears, no error
- **If the names match, but one place has a reducer and the other doesn't, and more than one parallel write lands on it in the same step:** hard crash, `InvalidUpdateError`

Practical takeaway: keep every field name **exactly identical**, spelled the same way, everywhere it appears across every schema in the project — and put the `Annotated[..., add]` reducer on any field that a `Send()`-based fan-out will write to more than once in parallel. A single typo in the wrong place is either invisible (worse) or a crash (better, at least you find out).

## Subgraphs: how three graphs became one

This project is technically three separate, independently-compiled `StateGraph`s, each with its own `TypedDict` state schema:

```python
class ReporterState(TypedDict):            # used by create_reporter's graph
    topic: str
    max_reporters: int
    human_feedback_on_reporters: NotRequired[Optional[str]]
    reporters: NotRequired[List[Reporter]]

class ReporterInterviewState(TypedDict):    # used by the interview subgraph
    reporter_generated_questions_and_answers: Annotated[list, operator.add]
    reporter: Reporter
    tavily_web_search_response: Annotated[list, operator.add]
    finished_interview_between_reporter_and_expert: str
    sections: list
    max_num_turns: int

class ResearchGraphState(TypedDict):        # used by the outer research_graph
    topic: str
    max_reporters: int
    human_feedback_on_reporters: NotRequired[Optional[str]]
    reporters: List[Reporter]
    sections: Annotated[list, operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str
```

**A compiled `StateGraph` is just a `Runnable`.** LangGraph doesn't distinguish "a subgraph" as a special type — `builder.compile()` returns an object that implements the same `Runnable` interface as any node function. That means you can pass a *compiled graph* directly to `add_node()`, exactly like a plain function:

```python
from src.answer_reporter_question_graph import graph as interview_subgraph

builder.add_node("conduct_reporter_interview", interview_subgraph)
```

When the parent graph reaches that node, it doesn't call a function — it invokes the entire compiled subgraph as if it were one atomic step, running it start-to-finish (including its own internal loop) before returning control to the parent.

**State passes between parent and subgraph by matching key names, not by explicit mapping.** `ReporterInterviewState` and `ResearchGraphState` are two different `TypedDict`s, with no inheritance or shared base class between them. LangGraph doesn't need one — when a subgraph is invoked as a node, it reads whichever keys of the parent's state happen to share a name with its own schema, and writes back the same way. `reporter`, `tavily_web_search_response`, `finished_interview_between_reporter_and_expert`, and `max_num_turns` in `ReporterInterviewState` have no counterpart in `ResearchGraphState` at all — they're private working state, invisible to the parent. `sections` exists in *both* schemas, by name, which is what makes it the actual hand-off point between the two graphs.

**`Send()` is what makes this run in parallel instead of once.** `Send(node_name, payload)` doesn't route to a fixed next node — it tells LangGraph "invoke `node_name` as a new, independent execution branch, seeded with exactly this payload as its starting state," and it can be called any number of times from one conditional edge:

```python
def initiate_all_interviews(state: ResearchGraphState):
    ...
    for reporter in state["reporters"]:
        send_list.append(
            Send("conduct_reporter_interview", {
                "reporter": reporter,
                "reporter_generated_questions_and_answers": [HumanMessage(...)]
            })
        )
    return send_list
```

Since `"conduct_reporter_interview"` is the subgraph node, each `Send()` here starts one full, isolated run of the entire interview subgraph — its own private `ReporterInterviewState`, seeded only with the `reporter` and opening message given in that `Send()`'s payload. Three reporters means three concurrent subgraph executions, each completely unaware of the others.

**The reducer is what makes the results merge back correctly.** `ReporterInterviewState.sections` is a plain `list` — inside one interview run, `write_section`'s `return {"sections": [section.content]}` simply overwrites it, since there's no reducer attached and each run only ever produces exactly one section. `ResearchGraphState.sections`, on the other hand, is `Annotated[list, operator.add]`. When each of the three parallel subgraph runs finishes and reports its one-item `sections` list back up to the parent, LangGraph doesn't let the last one overwrite the others — the `operator.add` reducer concatenates every incoming write into the parent's `sections` list instead. That's the actual mechanism that turns "three independent parallel runs" back into "one list of three sections" for `write_report` to read from afterward.

## Tech stack

- **LangGraph** — graph orchestration, state management, human-in-the-loop `interrupt()`, dynamic parallel fan-out via `Send`
- **LangChain** — LLM invocation, structured output, message types
- **OpenAI (gpt-4o-mini)** — reporter generation, question generation, answer generation, report writing
- **Tavily** — real web search for the expert's answers
- **uv** — dependency management

## Running it

```bash
uv sync
```

Create a `.env` with:
```
OPENAI_API_KEY=...
TAVILY_API_KEY=...
```

Then launch LangGraph Studio locally:
```bash
uv run langgraph dev
```

This exposes all three graphs above (`create_reporter`, `answer_reporter_question`, `research_graph`) — `research_graph` is the full end-to-end pipeline; the other two are useful for testing each stage in isolation.

Run `research_graph` with an initial state like:
```json
{
  "topic": "The role of mathematics in modern technology",
  "max_reporters": 3
}
```

The graph will pause at the human-feedback step and wait for input — approve with anything like `"okay"`/`"perfect"`/empty input, or type feedback to regenerate the reporter panel.

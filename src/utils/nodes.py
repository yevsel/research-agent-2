from .llm_models import llm
from .objects import Reporter,ReporterTeam,TavilySearchQuery
from .states import ReporterState, ReporterInterviewState, ResearchGraphState
from .prompts import reporter_instructions, reporter_question_instructions, search_instructions,expert_answer_instructions,section_writer_instructions,report_writer_instructions,intro_conclusion_instructions
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt
from langchain_tavily import TavilySearch
from langchain_core.messages import get_buffer_string

# ---------------------------------------------------------------------------
# PHASE 1: generate the reporter panel + human approval loop
# ---------------------------------------------------------------------------

def create_reporter(state:ReporterState):
    """Create reporters node"""
    topic = state.get("topic","")
    human_feedback_on_reporters = state.get("human_feedback_on_reporters","")
    max_reporters = state.get("max_reporters","")

    # Build the instructions the LLM will follow to invent the reporter panel.
    # If this is a regeneration (human gave feedback), that feedback is included too.
    create_reporter_prompt = reporter_instructions.format(topic=topic,human_feedback_on_reporters=human_feedback_on_reporters,max_reporters=max_reporters)

    # Force the LLM's reply to come back shaped exactly like a ReporterTeam object,
    # instead of a free-form AIMessage.
    create_reporter_llm = llm.with_structured_output(ReporterTeam)

    create_reporter_llm_response = create_reporter_llm.invoke([SystemMessage(content=create_reporter_prompt)])

    # .reporters pulls the actual List[Reporter] out of the ReporterTeam wrapper object.
    return {"reporters": create_reporter_llm_response.reporters}


def human_feedback(state:ReporterState):
    """Human feedback on the reporters created"""
    reporters = state.get("reporters",[])

    # Reporter objects aren't JSON-friendly on their own, so turn each one into
    # a plain dict (.model_dump()) before handing it to interrupt(), which has
    # to serialize this payload to show it to a human.
    formatted_reporters = []
    for reporter in reporters:
        if hasattr(reporter, "model_dump"):
            formatted_reporters.append(reporter.model_dump())
        else:
            formatted_reporters.append(reporter)

    # interrupt() pauses the graph here and waits for a human response.
    # Whatever the human sends back becomes the return value of this call.
    feedback = interrupt({
        "question": "Are these reporters okay for you?",
        "reporters": formatted_reporters,
        "instructions": "Return feedback to regenerate reporters or return empty/perfect/continue/okay to approve and continue the graph"
    })

    if isinstance(feedback, str):
        feedback = feedback.strip().lower()
        if feedback == "" or feedback in {"empty", "perfect", "continue", "okay"}:
            # Treated as approval — no feedback means "move on."
            return {"human_feedback_on_reporters": None}
        # Anything else typed is treated as real feedback, sent back to
        # create_reporter to regenerate the panel.
        return {"human_feedback_on_reporters": feedback}
    return {"human_feedback_on_reporters": None}


# ---------------------------------------------------------------------------
# PHASE 2: one reporter's interview (runs once per reporter, in parallel,
# spawned via Send() from the outer graph)
# ---------------------------------------------------------------------------

def generate_reporter_question(state:ReporterInterviewState):
    """Reporter generates a question here"""

    reporter_profile = state["reporter"]
    # After a graph pause/resume, LangGraph may hand this back as a plain dict
    # instead of a real Reporter object (checkpoints get serialized to JSON).
    # Rebuild the real object so .profile works either way.
    if isinstance(reporter_profile, dict):
        reporter_profile = Reporter.model_validate(reporter_profile)

    reporter_question_generator_prompt = reporter_question_instructions.format(profile=reporter_profile.profile)

    # Pass in the conversation so far (state.get(...) defaults to [] on the very
    # first question) so follow-up questions actually build on prior answers,
    # instead of being generated blind every time.
    reporter_question_llm_response = llm.invoke([SystemMessage(content=reporter_question_generator_prompt)] + state.get("reporter_generated_questions_and_answers", []))

    # This gets appended (not overwritten) onto reporter_generated_questions_and_answers,
    # because that field is Annotated[list, operator.add] in the state definition.
    return {"reporter_generated_questions_and_answers": [reporter_question_llm_response]}

def web_research(state:ReporterInterviewState):
    """Search the web for more information"""
    # Force structured output so we reliably get back a clean .tavily_search_query
    # string, instead of parsing it out of free-form text.
    web_search_structured_llm = llm.with_structured_output(TavilySearchQuery) # it will return a .search_query

    # Turn the reporter's latest question into a short, clean web search query
    # (a long conversational question is a bad search string on its own).
    web_search_reporter_question_plus_prompt = [SystemMessage(content=search_instructions)] + state["reporter_generated_questions_and_answers"]
    web_search_structured_llm_response = web_search_structured_llm.invoke(web_search_reporter_question_plus_prompt)

    # Actually hit Tavily with that generated query.
    tavily_search_response = TavilySearch(max_results=3).invoke(web_search_structured_llm_response.tavily_search_query)

    # Flatten the raw search results into one readable block of text, each
    # result tagged with its source URL so it can be cited later.
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href"{doc["url"]}"/>{doc["content"]}\n<Document>' for doc in tavily_search_response["results"]
        ]
    )

    return {"tavily_web_search_response": [formatted_search_docs]}

def web_research_2(state:ReporterInterviewState):
    """Search the web for more information"""
    # Identical to web_research — runs in parallel with it (both fire off the
    # same question at the same time) for a second, independent set of results.
    web_search_structured_llm = llm.with_structured_output(TavilySearchQuery) # it will return a .search_query
    web_search_reporter_question_plus_prompt = [SystemMessage(content=search_instructions)] + state["reporter_generated_questions_and_answers"]
    web_search_structured_llm_response = web_search_structured_llm.invoke(web_search_reporter_question_plus_prompt)

    tavily_search_response = TavilySearch(max_results=3).invoke(web_search_structured_llm_response.tavily_search_query)

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href"{doc["url"]}"/>{doc["content"]}\n<Document>' for doc in tavily_search_response["results"]
        ]
    )

    return {"tavily_web_search_response": [formatted_search_docs]}


def generate_expert_answer_from_research(state:ReporterInterviewState):
    """The 'expert' answers the reporter's latest question using the search results"""
    reporter=state["reporter"]
    reporter_generated_questions_and_answers=state["reporter_generated_questions_and_answers"]
    tavily_search_response=state["tavily_web_search_response"]

    if isinstance(reporter,dict):
        reporter = Reporter.model_validate(reporter)

    # Create the prompt so we get answer from the expert llm
    generate_expert_answer_prompt = expert_answer_instructions.format(profile=reporter.profile,tavily_web_response=tavily_search_response)

    # Feed in the whole conversation so far (not just the latest question) so
    # the expert answers in context, then answer the last unanswered question.
    answer_from_expert = llm.invoke([SystemMessage(content=generate_expert_answer_prompt)]+reporter_generated_questions_and_answers)

    # Tagging the message name doesn't affect get_buffer_string's output below
    # (that only looks at message type), but it's useful for other checks,
    # like counting how many times "expert" has answered so far.
    answer_from_expert.name = "expert"

    # Appended onto the same shared list as the questions — one growing
    # back-and-forth conversation, same order it happened in.
    return {"reporter_generated_questions_and_answers":[answer_from_expert]}


def save_interview(state: ReporterInterviewState):
    """save interviews"""
    reporter_generated_questions_and_answers = state["reporter_generated_questions_and_answers"]

    # Collapse the list of message objects into one flat, readable transcript
    # string, e.g. "AI: question...\nAI: answer...".
    finished_interview_between_reporter_and_expert = get_buffer_string(reporter_generated_questions_and_answers)
    return {"finished_interview_between_reporter_and_expert": finished_interview_between_reporter_and_expert}

def write_section(state: ReporterInterviewState):
    """Save section"""
    reporter = state["reporter"]
    tavily_web_search_response = state["tavily_web_search_response"]

    if isinstance(reporter,dict):
        reporter = Reporter.model_validate(reporter)

    # Ask the LLM to act as a technical writer and turn the raw search sources
    # (not the interview transcript) into one properly formatted report section,
    # focused on this reporter's specific angle.
    system_message = section_writer_instructions.format(focus=reporter.description)
    section = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Use this source to write your section: {tavily_web_search_response}")])

    # Only the plain text (.content) gets saved, not the whole message object.
    return {"sections": [section.content]}


# ---------------------------------------------------------------------------
# PHASE 3: outer graph — runs once, after every reporter's interview has
# finished and their sections have all been collected together
# ---------------------------------------------------------------------------

def write_report(state: ResearchGraphState):
    """Merge every reporter's section into one unified report body"""
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report
    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)
    report = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Write a report based upon these memos.")])
    return {"content": report.content}

def write_introduction(state: ResearchGraphState):
    """Write just the intro, previewing what all the sections cover"""
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report

    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
    intro = llm.invoke([SystemMessage(content=instructions)]+[HumanMessage(content=f"Write the report introduction")])
    return {"introduction": intro.content}

def write_conclusion(state: ResearchGraphState):
    """Write just the conclusion, recapping what all the sections covered"""

    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report

    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
    conclusion = llm.invoke([SystemMessage(content=instructions)]+[HumanMessage(content=f"Write the report conclusion")])
    return {"conclusion": conclusion.content}

def finalize_report(state: ResearchGraphState):
    """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
    # Save full final report
    content = state["content"]

    # write_report's output starts with a "## Insights" header (per its prompt) —
    # strip that exact prefix off since the header gets added back manually below
    # via the intro/conclusion structure instead.
    if content.startswith("## Insights"):
        content = content.removeprefix("## Insights")

    # Split the body out from its own "## Sources" section, so sources can be
    # appended once at the very end instead of sitting in the middle.
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    # Stitch everything into one final document: intro, body, conclusion.
    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}

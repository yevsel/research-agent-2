from .llm_models import llm
from .objects import Reporter,ReporterTeam,TavilySearchQuery
from .states import ReporterState, ReporterInterviewState, ResearchGraphState
from .prompts import reporter_instructions, reporter_question_instructions, search_instructions,expert_answer_instructions,section_writer_instructions,report_writer_instructions,intro_conclusion_instructions
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt
from langchain_tavily import TavilySearch
from langchain_core.messages import get_buffer_string

def create_reporter(state:ReporterState):
    """Create reporters node"""
    topic = state.get("topic","")
    human_feedback_on_reporters = state.get("human_feedback_on_reporters","")
    max_reporters = state.get("max_reporters","")
    
    create_reporter_prompt = reporter_instructions.format(topic=topic,human_feedback_on_reporters=human_feedback_on_reporters,max_reporters=max_reporters)
    
    create_reporter_llm = llm.with_structured_output(ReporterTeam)
    
    create_reporter_llm_response = create_reporter_llm.invoke([SystemMessage(content=create_reporter_prompt)])
    
    return {"reporters": create_reporter_llm_response.reporters}
    

def human_feedback(state:ReporterState):
    """Human feedback on the reporters created"""
    reporters = state.get("reporters",[])
    formatted_reporters = []
    for reporter in reporters:
        if hasattr(reporter, "model_dump"):
            formatted_reporters.append(reporter.model_dump())
        else:
            formatted_reporters.append(reporter)

    feedback = interrupt({
        "question": "Are these reporters okay for you?",
        "reporters": formatted_reporters,
        "instructions": "Return feedback to regenerate reporters or return empty/perfect/continue/okay to approve and continue the graph"
    })

    if isinstance(feedback, str):
        feedback = feedback.strip().lower()
        if feedback == "" or feedback in {"empty", "perfect", "continue", "okay"}:
            return {"human_feedback_on_reporters": None}
        return {"human_feedback_on_reporters": feedback}
    return {"human_feedback_on_reporters": None}
    

def generate_reporter_question(state:ReporterInterviewState):
    """Reporter generates a question here"""
    
    reporter_profile = state["reporter"]
    if isinstance(reporter_profile, dict):
        reporter_profile = Reporter.model_validate(reporter_profile)
        
    reporter_question_generator_prompt = reporter_question_instructions.format(profile=reporter_profile.profile)
    reporter_question_llm_response = llm.invoke([SystemMessage(content=reporter_question_generator_prompt)] + state.get("reporter_generated_questions_and_answers", []))
    
    return {"reporter_generated_questions_and_answers": [reporter_question_llm_response]}

def web_research(state:ReporterInterviewState):
    """Search the web for more information"""
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

def web_research_2(state:ReporterInterviewState):
    """Search the web for more information"""
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
    reporter=state["reporter"]
    reporter_generated_questions_and_answers=state["reporter_generated_questions_and_answers"]
    tavily_search_response=state["tavily_web_search_response"]
    
    if isinstance(reporter,dict):
        reporter = Reporter.model_validate(reporter)
    
    # Create the prompt so we get answer from the expert llm
    generate_expert_answer_prompt = expert_answer_instructions.format(profile=reporter.profile,tavily_web_response=tavily_search_response)
    
    answer_from_expert = llm.invoke([SystemMessage(content=generate_expert_answer_prompt)]+reporter_generated_questions_and_answers)
    
    answer_from_expert.name = "expert"
    
    return {"reporter_generated_questions_and_answers":[answer_from_expert]}
    

def save_interview(state: ReporterInterviewState):
    """save interviews"""
    reporter_generated_questions_and_answers = state["reporter_generated_questions_and_answers"]
    finished_interview_between_reporter_and_expert = get_buffer_string(reporter_generated_questions_and_answers)
    return {"finished_interview_between_reporter_and_expert": finished_interview_between_reporter_and_expert}

def write_section(state: ReporterInterviewState):
    """Save section"""
    reporter = state["reporter"]
    tavily_web_search_response = state["tavily_web_search_response"]
    
    if isinstance(reporter,dict):
        reporter = Reporter.model_validate(reporter)
    
    system_message = section_writer_instructions.format(focus=reporter.description)
    section = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Use this source to write your section: {tavily_web_search_response}")])
    
    return {"sections": [section.content]}
    
    
def write_report(state: ResearchGraphState):
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
    if content.startswith("## Insights"):
        content = content.removeprefix("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}
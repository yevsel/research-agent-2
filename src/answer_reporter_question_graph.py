from langgraph.graph import StateGraph, START, END
from src.utils.states import ReporterInterviewState
from src.utils.nodes import generate_reporter_question,web_research,web_research_2,generate_expert_answer_from_research,save_interview,write_section
from src.utils.edges import route_back_to_reporter_ask_question

builder = StateGraph(ReporterInterviewState)
builder.add_node("generate_reporter_question",generate_reporter_question)
builder.add_node("web_research",web_research)
builder.add_node("web_research_2",web_research_2)
builder.add_node("generate_expert_answer_from_research",generate_expert_answer_from_research)
builder.add_node("save_interview",save_interview)
builder.add_node("write_section",write_section)


builder.add_edge(START,"generate_reporter_question")
builder.add_edge("generate_reporter_question","web_research")
builder.add_edge("generate_reporter_question","web_research_2")
builder.add_edge("web_research","generate_expert_answer_from_research")
builder.add_edge("web_research_2","generate_expert_answer_from_research")
builder.add_edge("save_interview","write_section")
builder.add_conditional_edges("generate_expert_answer_from_research",route_back_to_reporter_ask_question,["save_interview","generate_reporter_question"])


graph=builder.compile()
from langgraph.graph import StateGraph, START, END
from src.utils.states import ResearchGraphState
from src.utils.nodes import create_reporter, human_feedback, write_report, write_introduction, write_conclusion, finalize_report
from src.utils.edges import initiate_all_interviews
from src.answer_reporter_question_graph import graph as interview_subgraph

builder = StateGraph(ResearchGraphState)
builder.add_node("create_reporter", create_reporter)
builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_reporter_interview", interview_subgraph)
builder.add_node("write_report", write_report)
builder.add_node("write_introduction", write_introduction)
builder.add_node("write_conclusion", write_conclusion)
builder.add_node("finalize_report", finalize_report)

builder.add_edge(START, "create_reporter")
builder.add_edge("create_reporter", "human_feedback")
builder.add_conditional_edges(
    "human_feedback",
    initiate_all_interviews,
    ["create_reporter", "conduct_reporter_interview"]
)
builder.add_edge("conduct_reporter_interview", "write_report")
builder.add_edge("conduct_reporter_interview", "write_introduction")
builder.add_edge("conduct_reporter_interview", "write_conclusion")
builder.add_edge("write_report", "finalize_report")
builder.add_edge("write_introduction", "finalize_report")
builder.add_edge("write_conclusion", "finalize_report")
builder.add_edge("finalize_report", END)

graph = builder.compile()

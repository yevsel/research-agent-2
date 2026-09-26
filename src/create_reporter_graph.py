from langgraph.graph import StateGraph, START, END
from src.utils.states import ReporterState
from src.utils.nodes import create_reporter, human_feedback
from src.utils.edges import return_to_create_reporter_edge

create_reporter_graph_builder = StateGraph(ReporterState)
create_reporter_graph_builder.add_node("create_reporter",create_reporter)
create_reporter_graph_builder.add_node("human_feedback",human_feedback)

create_reporter_graph_builder.add_edge(START,"create_reporter")
create_reporter_graph_builder.add_edge("create_reporter","human_feedback")
# create_reporter_graph_builder.add_edge("human_feedback",END)
create_reporter_graph_builder.add_conditional_edges("human_feedback",return_to_create_reporter_edge,["create_reporter","__end__"])

graph = create_reporter_graph_builder.compile()
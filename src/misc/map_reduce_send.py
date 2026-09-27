from typing import TypedDict, Annotated, List
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class OverallState(TypedDict):
    topic: str
    subtopics: List[str]
    research_results: Annotated[List[str], add]
    final_report: str

# Private state in each send
class ResearchState(TypedDict):
    subtopic: str
    research_results: List[str]
    
def generate_subtopics(state:OverallState):
    topic = state['topic']
    # We get a dynamic amount of subtopic, we usually dont know how many we get
    subtopics = [
        f"{topic} - History",
        f"{topic} - Current Trends",
        f"{topic} - Future Outlooks"
    ]
    
    return {
        "subtopics": subtopics
    }

def research_subtopic(state:ResearchState):
    # Each subtopic result
    subtopic = state["subtopic"]
    result = f"Research finding on '{subtopic}': [detailed analysis, data and insights]"
    
    return{
        "research_results": [result]
    }

def compile_report(state:OverallState):
    results = state["research_results"]
    
    report = "=" * 50 + "\n"
    report += "COMPREHENSIVE RESEARCH REPORT\n"
    report += "=" * 50 + "\n"
    
    for index, result in enumerate(results, 1):
        report += f"{index}. {result}\n\n"
    
    return {
        "final_report": report
    }       
    

def fan_out_research_subtopic(state: OverallState):
    
    return [
        Send("research_subtopic",{"subtopic": s}) for s in state["subtopics"]
    ]

builder = StateGraph(OverallState)
builder.add_node("generate_subtopics",generate_subtopics)
builder.add_node("research_subtopic",research_subtopic)
builder.add_node("compile_report",compile_report)

builder.add_edge(START,"generate_subtopics")
builder.add_conditional_edges("generate_subtopics",fan_out_research_subtopic)
builder.add_edge("research_subtopic","compile_report")
builder.add_edge("compile_report",END)


graph = builder.compile()


"""Test the Graph"""

result = graph.invoke({"topic":"Math"})

print(result["final_report"])
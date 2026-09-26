from .states import ReporterState,ReporterInterviewState,ResearchGraphState
from langchain_core.messages import AIMessage,HumanMessage
from langgraph.types import Send

def return_to_create_reporter_edge(state:ReporterState):
    human_feedback_on_reporters = state['human_feedback_on_reporters']
    if human_feedback_on_reporters:
        return "create_reporter"
    return "__end__"
    
def route_back_to_reporter_ask_question(state: ReporterInterviewState, name: str="expert"):
    """Route between question and answer"""
    
    #Get messages
    messages=state["reporter_generated_questions_and_answers"]
    max_num_turns = state.get("max_num_turns",2)
    
    #check the number of expert answers
    num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name==name])
    if num_responses>=max_num_turns:
        return "save_interview"
    
    return "generate_reporter_question"
    
    
def initiate_all_interviews(state: ResearchGraphState):
    """ This is the "map" step where we run each interview sub-graph using Send API """    

    # Check if human feedback
    human_feedback_on_reporters=state.get('human_feedback_on_reporters')
    if human_feedback_on_reporters:
        # Return to create_analysts
        return "create_reporter"

    # Otherwise kick off interviews in parallel via Send() API
    else:
        topic = state["topic"]
        send_list = []
        for reporter in state["reporters"]:
            send_list.append(
                Send("conduct_reporter_interview", {
                    "reporter": reporter,
                    "reporter_generated_questions_and_answers": [
                        HumanMessage(content=f"So you said you were writing an article on {topic}?")
                    ]
                })
            )
        return send_list
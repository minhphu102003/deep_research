from typing import Literal
from open_deep_research.configuration import Configuration
from open_deep_research.nodes.planner import PlannerOutput
from open_deep_research.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from open_deep_research.utils.model import default_model_config

from langgraph.types import Command
from langgraph.graph import END


async def assess_gaps(state: AgentState, config: RunnableConfig) -> Command[Literal["replan", "writer"]]:
    has_gaps = bool(state.get("knowledge_gaps"))
    max_replans = getattr(Configuration.from_runnable_config(config), "max_replans", 1)
    replan_count = state.get("replan_count", 0)

    if has_gaps and replan_count < max_replans:
        return Command(goto="replan", update={})
    return Command(goto="writer", update={})


async def replan(state: AgentState, config: RunnableConfig) -> Command[Literal["research"]]:
    planner_model = default_model_config().with_structured_output(PlannerOutput)
    prompt = (
        "Given these knowledge gaps, add/modify sections to cover them. "
        "Return ONLY the JSON:\n"
        f"{state.get('knowledge_gaps', [])}"
    )
    po: PlannerOutput = await planner_model.ainvoke([HumanMessage(content=prompt)])
    return Command(
        goto="research",
        update={
            "sections": [s.model_dump() for s in po.sections],  
            "knowledge_gaps": [],
            "replan_count": 1,
            "messages": [AIMessage(content=f"Re-planned {len(po.sections)} additional sections.")],
        },
    )


async def writer(state: AgentState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    writer_model = default_model_config() 
    outline = "\n\n".join(f"## {s['title']}\n{s['content']}" for s in state.get("completed_sections", []))
    msg = HumanMessage(content=f"Polish into a coherent report with references:\n{outline}")
    final = await writer_model.ainvoke([msg])  
    return Command(goto=END, update={"final_report": final.content})

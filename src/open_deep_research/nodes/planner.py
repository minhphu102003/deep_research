from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, convert_to_messages, get_buffer_string
from langgraph.graph import END
from langgraph.types import Command
from open_deep_research.configuration import (
    Configuration,
)
from langchain_core.runnables import RunnableConfig

from open_deep_research.state import AgentState
from open_deep_research.utils.prompt_loader import render_prompt
from open_deep_research.utils.utils import get_today_str
from open_deep_research.utils.model import default_model_config

class SectionPlan(BaseModel):
    title: str = Field(..., description="Clear, atomic research topic or section heading.")
    research: bool = Field(True, description="Whether this section requires active research.")
    rationale: Optional[str] = Field(None, description="Why this section is included.")

class PlannerOutput(BaseModel):
    need_clarification: bool = Field(False, description="Ask the user a clarifying question first?")
    question: Optional[str] = Field(None, description="Clarifying question if needed.")
    sections: List[SectionPlan] = Field(default_factory=list, description="Planned sections/tasks.")

async def planner(state: AgentState, config: RunnableConfig) -> Command[Literal["research", "__end__"]]:
    configurable = Configuration.from_runnable_config(config)

    allow_clar = getattr(configurable, "allow_clarification", True)
    messages = state["messages"]

    # https://python.langchain.com/docs/how_to/structured_output/
    model = default_model_config()
    structured = model.with_structured_output(PlannerOutput)

    try:
        msgs = convert_to_messages(messages)  
        user_ctx = get_buffer_string(msgs) 
    except Exception:
        user_ctx = "\n\n".join(
            getattr(m, "content", "") for m in messages if isinstance(m, (HumanMessage, AIMessage))
        )

    prompt = render_prompt("planner", user_ctx=user_ctx, today=get_today_str())

    po: PlannerOutput = await structured.ainvoke([HumanMessage(content=prompt)])

    if not allow_clar:
        po.need_clarification = False
        po.question = None

    if po.need_clarification and po.question:
        return Command(goto=END, update={"messages": [AIMessage(content=po.question)]})

    return Command(
        goto="research",
        update={
            "messages": [AIMessage(content=f"Planned {len(po.sections)} sections.")],
            "sections": [s.model_dump() for s in po.sections], 
            "replan_count": 0,                                
        },
    )

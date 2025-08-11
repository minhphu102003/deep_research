import asyncio
import inspect
import json
from typing import List, Dict, Any, Literal, Tuple
from pydantic import BaseModel, Field
from langgraph.types import Command
from langgraph.graph import END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration, MCPConfig
from open_deep_research.state import AgentState
from open_deep_research.utils.model import default_model_config
from open_deep_research.utils.prompt_loader import render_prompt
from open_deep_research.utils.utils import get_all_tools
from open_deep_research.utils.citations import extract_citations

class SectionWriteup(BaseModel):
    content: str = Field(..., description="Concise, well-cited writeup for this section.")
    gaps: List[str] = Field(default_factory=list, description="Open questions or missing info uncovered.")

def _research_model_with_tools(configurable) -> ChatOpenAI:
    model = default_model_config()
    tools = getattr(configurable, "research_tools", []) 
    if tools:
        model = model.bind_tools(tools)
    return model

async def _research_model_with_tools(config) -> Tuple[ChatOpenAI, Dict[str, Any]]:

    model = default_model_config()
    tools = await get_all_tools(config)
    if tools:
        model = model.bind_tools(tools)

    registry = {}
    for t in tools:
        name = getattr(t, "name", None) or getattr(t, "__name__", None)
        if name and name not in registry:
            registry[name] = t
    return model, registry

async def _run_tool_call(tool_registry: Dict[str, Any], name: str, args: Dict[str, Any]) -> str:
    tool = tool_registry.get(name)
    if tool is None:
        return f"[ERROR] Unknown tool '{name}' (args={args})"

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            pass

    try:
        if hasattr(tool, "ainvoke"):
            result = await tool.ainvoke(args)
        elif hasattr(tool, "invoke"):
            result = await asyncio.to_thread(tool.invoke, args)
        elif callable(tool):
            if inspect.iscoroutinefunction(tool):
                result = await tool(**(args or {}))
            else:
                result = await asyncio.to_thread(lambda: tool(**(args or {})))
        else:
            result = await asyncio.to_thread(lambda: tool(args))
    except Exception as e:
        return f"[ERROR] Tool '{name}' failed: {e}"

    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

async def _finalize_section(section_title: str, history: List) -> SectionWriteup:
    clean_model = default_model_config()
    chain = clean_model.with_structured_output(SectionWriteup)
    prompt = render_prompt("research_section_writer", section_title=section_title)
    out: SectionWriteup = await chain.ainvoke([HumanMessage(content=prompt)])
    return out

async def _react_tool_loop(model: ChatOpenAI, seed_messages: List, max_calls: int,
                           tool_registry: Dict[str, Any]) -> tuple[list, list[dict]]:
    history = list(seed_messages)
    logs: list[dict] = []

    for step in range(max_calls):
        resp: AIMessage = await model.ainvoke(history) 
        history.append(resp)

        tool_calls = getattr(resp, "tool_calls", None)
        if not tool_calls:
            break

        tool_msgs = []
        for tc in tool_calls:
            name = tc.get("name")
            args = tc.get("args", {})
            tcid = tc.get("id")

            result = await _run_tool_call(tool_registry, name, args) 
            tool_msgs.append(ToolMessage(content=str(result), tool_call_id=tcid))

            logs.append({
                "iteration": step,
                "name": name,
                "args": args,
                "result": result,
            })

        history.extend(tool_msgs)

    return history, logs

def _pending_sections(state) -> List[Dict[str, Any]]:
    done_titles = {c["title"] for c in state.get("completed_sections", [])}
    return [s for s in state.get("sections", [])
            if s.get("research", True) and s.get("title") not in done_titles]

async def research(state: AgentState, config: RunnableConfig) -> Command[Literal["research", "assess_gaps"]]:
    configurable = Configuration.from_runnable_config(config)

    updated_config = {
        **config,
        "configurable": {
            **config.get("configurable", {}),
            "mcp_config": MCPConfig(
                mode="http",
                url="http://localhost:8000",
                path_prefix="/mcp/",
                tools=["tavily_search"],
            ),
        },
    }

    pending = _pending_sections(state)
    if not pending:
        return Command(goto="assess_gaps", update={})

    batch_n = getattr(configurable, "max_concurrent_sections", 2)
    todo = pending[:batch_n]

    research_model, tool_registry = await _research_model_with_tools(updated_config)
    max_calls = getattr(configurable, "max_react_tool_calls", 5)

    new_completed = []
    new_gaps: List[str] = []
    new_tool_logs: list[dict] = []

    for sec in todo:
        title = sec["title"]

        seed = [HumanMessage(content=(
            "You are a Research SubAgent. Investigate the section thoroughly using tools when helpful. "
            "Prefer reputable sources (papers/standards/docs/news). Think step-by-step."
            f"\n\nSECTION: {title}"
        ))]

        history, logs = await _react_tool_loop(research_model, seed, max_calls=max_calls,
                                                    tool_registry=tool_registry)

        writeup = await _finalize_section(title, history)
        sources = extract_citations(history, max_refs=10)
        new_completed.append({
            "title": title,
            "content": writeup.content,
            "sources": sources,
        })
        if writeup.gaps:
            new_gaps.extend(writeup.gaps)

    return Command(
        goto="research",
        update={
            "completed_sections": new_completed,
            "knowledge_gaps": new_gaps,
            "tool_call_history": state.get("tool_call_history", []) + new_tool_logs,
            "messages": [AIMessage(content=f"Finished {len(todo)} section(s). Pending: {len(pending) - len(todo)}")]
        }
    )
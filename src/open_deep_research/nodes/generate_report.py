from __future__ import annotations
import json, urllib.parse
from typing import List, Dict, Any, Literal, Optional, Iterable
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph import END

from open_deep_research.state import AgentState
from open_deep_research.utils.model import default_model_config
from langchain_core.runnables import RunnableConfig

from open_deep_research.utils.prompt_loader import render_prompt

class Reference(BaseModel):
    id: int = Field(..., description="1-based index used in the reference list")
    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None

class FinalReport(BaseModel):
    title: str
    summary: str
    content_markdown: str  
    references: List[Reference]

def _canon(u: str) -> str:
    try:
        p = urllib.parse.urlsplit(u)
        return urllib.parse.urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path, p.query, ""))
    except Exception:
        return u

def aggregate_references(sections: Iterable[Dict[str, Any]], max_refs: int = 30) -> List[Dict[str, Any]]:
    seen, refs = set(), []
    for s in sections or []:
        for r in (s.get("sources") or []):
            url = _canon(r.get("url") or "")
            if not url or url in seen:
                continue
            seen.add(url)
            refs.append({
                "url": url,
                "title": r.get("title"),
                "snippet": r.get("snippet"),
            })
            if len(refs) >= max_refs:
                return refs
    return refs

async def generate_report(state: AgentState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    sections = state.get("completed_sections", [])
    refs_raw = aggregate_references(sections)

    body = state.get("final_report") or "\n\n".join(
        f"## {s['title']}\n{s['content']}" for s in sections
    )

    model = default_model_config().with_structured_output(FinalReport)
    prompt = render_prompt(
        "final_report",
        body=body,
        refs_json=json.dumps(refs_raw, ensure_ascii=False)
    )
    out: FinalReport = await model.ainvoke([HumanMessage(content=prompt)])

    return Command(
        goto=END,
        update={
            "final_report": out.content_markdown,
            "report_title": out.title,
            "report_summary": out.summary,
            "references": [r.model_dump() for r in out.references],
        },
    )
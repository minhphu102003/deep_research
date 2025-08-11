from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from jinja2 import Environment, FileSystemLoader, StrictUndefined

_BASE = Path(__file__).resolve().parents[1] / "prompts"

_env = Environment(
    loader=FileSystemLoader(str(_BASE)),
    undefined=StrictUndefined,         
    autoescape=False,                   
    trim_blocks=True,
    lstrip_blocks=True,
)

def render_prompt(name: str, **kwargs: Dict[str, Any]) -> str:
    """
    Render file <name>.j2 trong folder prompts với biến kwargs.
    """
    template = _env.get_template(f"{name}.j2")
    return template.render(**kwargs)
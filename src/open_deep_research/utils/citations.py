import re, json, urllib.parse, datetime
from typing import Any, Dict, List, Iterable
from langchain_core.messages import ToolMessage

_URL_RE = re.compile(r"https?://[^\s)>\]}\"']+")

def _canon(u: str) -> str:
    try:
        p = urllib.parse.urlsplit(u)
        return urllib.parse.urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path, p.query, ""))
    except Exception:
        return u

def _as_list(obj: Any) -> List[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]

def _from_hit(hit: Dict[str, Any]) -> Dict[str, Any] | None:
    if not isinstance(hit, dict):
        return None
    url = hit.get("url") or hit.get("link") or hit.get("source") or hit.get("href")
    if not url:
        return None
    title = hit.get("title") or hit.get("name") or hit.get("headline")
    snippet = hit.get("content") or hit.get("snippet") or hit.get("description") or hit.get("text")
    return {"url": _canon(str(url)), "title": (str(title) if title else None), "snippet": (str(snippet) if snippet else None)}

def _from_string(s: str) -> List[Dict[str, Any]]:
    out = []
    try:
        obj = json.loads(s)
        for item in _as_list(obj):
            if isinstance(item, dict):
                c = _from_hit(item)
                if c: out.append(c)
    except Exception:
        for m in _URL_RE.findall(s or ""):
            out.append({"url": _canon(m), "title": None, "snippet": None})
    return out

def extract_citations(history: Iterable[Any], max_refs: int = 10) -> List[Dict[str, Any]]:
    seen = set()
    refs: List[Dict[str, Any]] = []
    for msg in history:
        if not isinstance(msg, ToolMessage):
            continue
        content = msg.content
        items: List[Dict[str, Any]] = []
        if isinstance(content, list):
            for it in content:
                if isinstance(it, dict):
                    c = _from_hit(it)
                    if c: items.append(c)
                elif isinstance(it, str):
                    items.extend(_from_string(it))
        elif isinstance(content, dict):
            c = _from_hit(content)
            if c: items.append(c)
        elif isinstance(content, str):
            items.extend(_from_string(content))

        for c in items:
            url = c["url"]
            if not url or url in seen:
                continue
            seen.add(url)
            refs.append({
                "url": url,
                "title": c.get("title"),
                "snippet": c.get("snippet"),
                "accessed_at": datetime.datetime.utcnow().isoformat() + "Z",
            })
            if len(refs) >= max_refs:
                return refs
    return refs

from fastapi import FastAPI
from mcp_server.tools.google_search import web_search, get_tool_schema

app = FastAPI()

@app.post("/web_search")
async def search_endpoint(input: dict):
    return await web_search(input)

@app.get("/mcp")
async def get_mcp_tools():
    return {
        "tools": [get_tool_schema()]
    }

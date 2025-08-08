from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional
from langchain_core.runnables import RunnableConfig
import os
from enum import Enum

class SearchAPI(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"

class MCPConfig(BaseModel):
    mode: Literal["http", "stdio"] = Field(
        default="http",
        description="Transport mode: 'http' for HTTP/SSE, 'stdio' for stdio-based MCP server",
    )

    # --- HTTP/SSE ---
    url: Optional[str] = Field(
        default=None,
        description="Base URL of the MCP server (e.g., http://localhost:8000) for HTTP/SSE mode",
    )
    path_prefix: str = Field(
        default="/mcp",
        description="Path prefix for MCP endpoints when using HTTP/SSE (e.g., /mcp)",
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Additional HTTP headers to send with requests in HTTP/SSE mode",
    )
    auth_required: bool = Field(
        default=False,
        description="Whether authentication is required when connecting to MCP server",
    )

    # --- STDIO ---
    command: Optional[List[str]] = Field(
        default=None,
        description="Command to launch the MCP server in stdio mode (e.g., ['python', '-m', 'mcp_server'])",
    )
    env: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional environment variables to set when running MCP server in stdio mode",
    )

    tools: Optional[List[str]] = Field(
        default=None,
        description="List of MCP tool names to make available to the LLM",
    )

class Configuration(BaseModel):
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
            }
        }
    )
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 3,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    # Model Configuration
    summarization_model: str = Field(
        default="openai:gpt-4.1-nano",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1-nano",
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    research_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        default="openai:gpt-4.1-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1-mini",
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model"
            }
        }
    )
    # MCP server configuration
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(  
        default=(
            "You are provided with function signatures within <tools></tools> XML tags.\n"
            "When calling a tool, respond using this exact format:\n\n"
            "<tool_call>\n"
            '{"name": "ToolName", "arguments": {"arg1": "value1"}}\n'
            "</tool_call>\n\n"
            "Rules:\n"
            "- Return exactly ONE tool call per turn.\n"
            "- Do NOT include any text outside <tool_call> ... </tool_call>.\n"
            "- Arguments MUST match the tool's input schema.\n\n"
            "Tool selection policy:\n"
            '- Prefer \"smart_search\" for research-type queries that need rewriting, multi-source web search, scraping, summarization, or stateful history.\n'
            '- Prefer \"tavily_search\" for quick lookups or when the user only needs raw links.\n'
            "- If uncertain, default to \"tavily_search\".\n\n"
            "Argument hints:\n"
            "- smart_search requires: session_id (string), query (string). Optional: prefer_academic, time_range, extra_sites, filetype_pdf, target_language.\n"
            "- tavily_search requires: query (string).\n"
        ),
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        arbitrary_types_allowed = True
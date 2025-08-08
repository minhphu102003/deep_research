import pytest
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage
from open_deep_research.deep_researcher import researcher
import json
from dotenv import load_dotenv
load_dotenv()

test_state = {
    "researcher_messages": [
        HumanMessage(
            content=(
                "I need a detailed report about the weather in Tokyo. "
                "Please find both the current weather and the weather forecast. "
                "Include temperature, conditions, and anything relevant."
            )
        )
    ],
    "tool_call_iterations": 0,
    "tool_call_history": []
}

test_config = RunnableConfig(configurable={
    "mcp_prompt": "",
    "max_structured_output_retries": 2
})

def custom_serializer(obj):
    try:
        return str(obj) 
    except Exception:
        return f"<<non-serializable: {type(obj).__name__}>>"


# @pytest.mark.asyncio
# async def test_researcher_node():
#     result = await researcher(test_state, test_config)
    
#     print("\n========== Final Result ==========")
#     print(json.dumps(result, indent=2, ensure_ascii=False, default=custom_serializer))

#     assert result is not None


TEST_CONFIG = RunnableConfig(configurable={
    "mcp_prompt": (
        "You have two MCP tools: smart_search (multi-step research; supports "
        "session_id, prefer_academic, time_range, extra_sites, filetype_pdf, target_language) "
        "and tavily_search (quick web search). "
        "When a user explicitly asks to use one by name, prefer calling that tool."
    ),
    "max_structured_output_retries": 2,
})

SMART_SEARCH_PROMPT = (
    "Hãy **dùng công cụ smart_search** để lập một mini-brief về chủ đề "
    "'tác động đảo nhiệt đô thị (urban heat island) tại Tokyo'. "
    "Yêu cầu:\n"
    "- session_id='pytest-ss-1'\n"
    "- prefer_academic=true; time_range='past_year'\n"
    "- extra_sites=['arxiv.org','nature.com']\n"
    "- filetype_pdf=true; target_language='vi'\n"
    "Trả về JSON gồm: rewritten_query, used_query, result (tóm tắt tiếng Việt), state_meta."
)

TAVILY_SEARCH_PROMPT = (
    "Hãy **dùng công cụ tavily_search** để tìm 8 bài viết gần đây về "
    "'thời tiết Tokyo hôm nay và cuối tuần'. "
    "Trả về danh sách {title, url, source}. "
    "Ưu tiên nguồn tin cậy và cập nhật."
)

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_prompt, expect_nonempty",
    [
        (SMART_SEARCH_PROMPT, True),
        (TAVILY_SEARCH_PROMPT, True),
    ],
)
async def test_mcp_tools(user_prompt, expect_nonempty):
    test_state = {
        "researcher_messages": [HumanMessage(content=user_prompt)],
        "tool_call_iterations": 0,
        "tool_call_history": [],
    }
    result = await researcher(test_state, TEST_CONFIG)

    print("\n========== Final Result ==========")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=custom_serializer))

    assert result is not None
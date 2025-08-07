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


@pytest.mark.asyncio
async def test_researcher_node():
    result = await researcher(test_state, test_config)
    
    print("\n========== Final Result ==========")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=custom_serializer))

    assert result is not None

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage, get_buffer_string, filter_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command
import asyncio
from typing import Literal
from open_deep_research.configuration import (
    Configuration, 
)
import json
from typing import Literal
from open_deep_research.state import (
    AgentState,
    AgentInputState,
    SupervisorState,
    ResearcherState,
    ClarifyWithUser,
    ResearchQuestion,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    transform_messages_into_research_topic_prompt,
    research_system_prompt,
    compress_research_system_prompt,
    compress_research_simple_human_message,
    final_report_generation_prompt,
    lead_researcher_prompt
)
from open_deep_research.utils import (
    get_today_str,
    is_token_limit_exceeded,
    get_model_token_limit,
    get_all_tools,
    openai_websearch_called,
    anthropic_websearch_called,
    remove_up_to_last_ai_message,
    get_api_key_for_gemini,
    get_notes_from_tool_calls,
)
from functools import partial
from langchain_google_genai import ChatGoogleGenerativeAI

def init_chat_model(model: str, max_tokens: int, api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        max_output_tokens=max_tokens,
        google_api_key=api_key,
        convert_system_message_to_human=True
    )

default_model_config = partial(init_chat_model, model="models/gemini-2.5-flash", max_tokens=2048, api_key=get_api_key_for_gemini())

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    configurable = Configuration.from_runnable_config(config)

    if not configurable.allow_clarification:
        return Command(goto="write_research_brief")
    messages = state["messages"]

    model = default_model_config()

    user_prompt = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages),
        date=get_today_str()
    )

    raw_response = await model.ainvoke([HumanMessage(content=user_prompt)])

    raw = raw_response.content.strip()

    if raw.startswith("```json") and raw.endswith("```"):
        raw = raw[7:-3].strip() 

    try:
        parsed_json = json.loads(raw)
        clarify = ClarifyWithUser(**parsed_json)
    except Exception as e:
        raise ValueError(f"Failed to parse response: {raw}\nError: {str(e)}")

    if clarify.need_clarification:
        return Command(goto=END, update={"messages": [AIMessage(content=clarify.question)]})
    else:
        return Command(goto="write_research_brief", update={"messages": [AIMessage(content=clarify.verification)]})


async def write_research_brief(state: AgentState, config: RunnableConfig)-> Command[Literal["research_supervisor"]]:
    configurable = Configuration.from_runnable_config(config)
    model = default_model_config()

    model = model.with_structured_output(ResearchQuestion).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    ).with_config({
        "tags": ["langsmith:nostream"]
    })
    response = await model.ainvoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])
    print(response.research_brief)

    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=lead_researcher_prompt.format(
                        date=get_today_str(),
                        max_concurrent_research_units=configurable.max_concurrent_research_units
                    )),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    configurable = Configuration.from_runnable_config(config)
    model = default_model_config()
    lead_researcher_tools = [ConductResearch, ResearchComplete]
    research_model = model.bind_tools(lead_researcher_tools).with_retry(
            stop_after_attempt=configurable.max_structured_output_retries
        ).with_config({
            "tags": ["langsmith:nostream"]
        })
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": supervisor_messages + [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )


async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    # Exit Criteria
    # 1. We have exceeded our max guardrail research iterations
    # 2. No tool calls were made by the supervisor
    # 3. The most recent message contains a ResearchComplete tool call and there is only one tool call in the message
    exceeded_allowed_iterations = research_iterations >= configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls)
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    # Otherwise, conduct research and gather results.
    try:
        all_conduct_research_calls = [tool_call for tool_call in most_recent_message.tool_calls if tool_call["name"] == "ConductResearch"]
        conduct_research_calls = all_conduct_research_calls[:configurable.max_concurrent_research_units]
        overflow_conduct_research_calls = all_conduct_research_calls[configurable.max_concurrent_research_units:]
        coros = [
            researcher_subgraph.ainvoke({
                "researcher_messages": [
                    HumanMessage(content=tool_call["args"]["research_topic"])
                ],
                "research_topic": tool_call["args"]["research_topic"]
            }, config) 
            for tool_call in conduct_research_calls
        ]
        tool_results = await asyncio.gather(*coros)
        tool_messages = [ToolMessage(
                            content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"]
                        ) for observation, tool_call in zip(tool_results, conduct_research_calls)]
        # Handle any tool calls made > max_concurrent_research_units
        for overflow_conduct_research_call in overflow_conduct_research_calls:
            tool_messages.append(ToolMessage(
                content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                name="ConductResearch",
                tool_call_id=overflow_conduct_research_call["id"]
            ))
        raw_notes_concat = "\n".join(["\n".join(observation.get("raw_notes", [])) for observation in tool_results])
        subagents = state.get("subagents", [])
        new_subagents = [tool_call["args"]["research_topic"] for tool_call in conduct_research_calls]
        subagents = list(set(subagents + new_subagents))
        return Command(
            goto="supervisor",
            update={
                "supervisor_messages": supervisor_messages + tool_messages,
                "raw_notes": [raw_notes_concat],
                "subagents": subagents,
            }
        )
    except Exception as e:
        if is_token_limit_exceeded(e, configurable.research_model):
            print(f"Token limit exceeded while reflecting: {e}")
        else:
            print(f"Other error in reflection phase: {e}")
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )

supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)

supervisor_builder.add_edge(START, "supervisor")
supervisor_builder.add_edge("supervisor", "supervisor_tools")
supervisor_builder.add_edge("supervisor_tools", "supervisor")  
supervisor_builder.add_edge("supervisor_tools", END)           

supervisor_subgraph = supervisor_builder.compile()


async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    configurable = Configuration.from_runnable_config(config)

    researcher_messages = state.get("researcher_messages", [])
    tools = await get_all_tools(config)

    if not tools:
        raise ValueError("No tools found to conduct research: Please configure either your search API or add MCP tools to your configuration.")

    model = default_model_config()
    research_model = model.bind_tools(tools).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    ).with_config({
        "tags": ["langsmith:nostream"]
    })

    researcher_system_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "",
        date=get_today_str()
    )

    prompt_messages = [SystemMessage(content=researcher_system_prompt)] + researcher_messages
    prompt_messages = [m for m in prompt_messages if isinstance(m, BaseMessage) and m.content.strip()]

    if not prompt_messages:
        prompt_messages = [
            SystemMessage(content=researcher_system_prompt),
            HumanMessage(content="Please begin the research on the given topic.")
        ]

    response = await research_model.ainvoke(prompt_messages)

    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )


async def execute_tool_safely(tool, args, config):
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    # Early Exit Criteria: No tool calls (or native web search calls)were made by the researcher
    if not most_recent_message.tool_calls and not (openai_websearch_called(most_recent_message) or anthropic_websearch_called(most_recent_message)):
        return Command(
            goto="compress_research",
        )
    # Otherwise, execute tools and gather results.
    tools = await get_all_tools(config)
    tools_by_name = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search"):tool for tool in tools}
    tool_calls = most_recent_message.tool_calls
    coros = [execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) for tool_call in tool_calls]
    observations = await asyncio.gather(*coros)
    tool_outputs = [ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    ) for observation, tool_call in zip(observations, tool_calls)]
    
    # Late Exit Criteria: We have exceeded our max guardrail tool call iterations or the most recent message contains a ResearchComplete tool call
    # These are late exit criteria because we need to add ToolMessages
    if state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls or any(tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls):
        return Command(
            goto="compress_research",
            update={
                "researcher_messages": tool_outputs,
            }
        )
    return Command(
        goto="researcher",
        update={
            "researcher_messages": tool_outputs,
        }
    )


async def compress_research(state: ResearcherState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    synthesis_attempts = 0

    model = default_model_config()
    synthesizer_model = model.with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    ).with_config({
        "tags": ["langsmith:nostream"]
    })

    researcher_messages = state.get("researcher_messages", [])
    researcher_messages = [m for m in researcher_messages if hasattr(m, "content") and m.content.strip()]

    if not researcher_messages:
        researcher_messages = [
            HumanMessage(content="Here are some initial research findings that need to be compressed.")
        ]

    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))

    system_prompt = SystemMessage(content=compress_research_system_prompt.format(date=get_today_str()))
    prompt_messages = [system_prompt] + researcher_messages

    while synthesis_attempts < 3:
        try:
            response = await synthesizer_model.ainvoke(prompt_messages)
            return {
                "compressed_research": str(response.content),
                "raw_notes": ["\n".join(
                    [str(m.content) for m in filter_messages(researcher_messages, include_types=["tool", "ai"])]
                )]
            }
        except Exception as e:
            synthesis_attempts += 1
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                print(f"⚠️ Token limit exceeded: {e}. Pruning and retrying.")
                continue
            print(f" Error synthesizing: {e}")
            break

    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": ["\n".join(
            [str(m.content) for m in filter_messages(researcher_messages, include_types=["tool", "ai"])]
        )]
    }


researcher_builder = StateGraph(ResearcherState, output=ResearcherOutputState, config_schema=Configuration)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)
researcher_subgraph = researcher_builder.compile()

async def check_loop_condition(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "final_report_generation"]]:
    findings = state.get("raw_notes", [])
    subagents = state.get("subagents", [])
    loop_count = state.get("loop_count", 0)

    enough_findings = len(findings) >= 3 and all(len(note.strip()) > 500 for note in findings)
    enough_subagents = len(set(subagents)) >= 2
    subagent_covered = all(
        any(subagent.lower() in note.lower() for note in findings)
        for subagent in subagents
    ) if subagents else False

    if loop_count >= 3:
        print("Max retry reached. Exiting loop.")
        return Command(goto="final_report_generation")

    if enough_findings and enough_subagents and subagent_covered:
        return Command(goto="final_report_generation")
    else:
        state["loop_count"] = loop_count + 1
        return Command(goto="write_research_brief")



async def final_report_generation(state: AgentState, config: RunnableConfig):
    notes = state.get("notes", [])
    cleared_state = {
        "notes": {"type": "override", "value": []},
    }
    configurable = Configuration.from_runnable_config(config)

    findings = "\n".join(notes)
    max_retries = 3
    current_retry = 0

    model = default_model_config()

    writer_model = model.with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    ).with_config({
        "tags": ["langsmith:nostream"]
    })

    while current_retry <= max_retries:
        final_report_prompt = final_report_generation_prompt.format(
            research_brief=state.get("research_brief", ""),
            messages=get_buffer_string(state.get("messages", [])),
            findings=findings,
            date=get_today_str()
        )
        try:
            final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])
            return {
                "final_report": final_report.content,
                "messages": [final_report],
                **cleared_state
            }
        except Exception as e:
            if is_token_limit_exceeded(e, configurable.final_report_model):
                if current_retry == 0:
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, and model token limit is unknown. Please update the model map in deep_researcher/utils.py. {e}",
                            **cleared_state
                        }
                    findings_token_limit = model_token_limit * 4
                else:
                    findings_token_limit = int(findings_token_limit * 0.9)
                print("Reducing the chars to", findings_token_limit)
                findings = findings[:findings_token_limit]
                current_retry += 1
            else:
                return {
                    "final_report": f"Error generating final report: {e}",
                    **cleared_state
                }

    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [],
        **cleared_state
    }

deep_researcher_builder = StateGraph(AgentState, input=AgentInputState, config_schema=Configuration)
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)
deep_researcher_builder.add_node("check_loop_condition", check_loop_condition)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("clarify_with_user", "write_research_brief")
deep_researcher_builder.add_edge("write_research_brief", "research_supervisor")
deep_researcher_builder.add_edge("research_supervisor", "check_loop_condition")
deep_researcher_builder.add_edge("check_loop_condition", "write_research_brief")
deep_researcher_builder.add_edge("check_loop_condition", "final_report_generation")  

deep_researcher = deep_researcher_builder.compile()
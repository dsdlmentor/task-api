"""Unit tests for app/agent/graph.py — checking ReAct cycle termination."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

import app.agent.graph as graph_module
from app.agent.graph import build_agent_graph


@patch("app.agent.graph._get_llm_with_tools")
def test_graph_terminates_after_max_iterations(mock_llm_factory):
    """If LLM keeps returning tool_calls, graph must stop at max_iterations.

    Note: each AIMessage and tool_call must have a UNIQUE id, otherwise
    LangGraph's `add_messages` reducer will deduplicate by id and the
    state gets corrupted (older messages get replaced in-place).
    """
    counter = {"i": 0}

    def make_loop_response(*args, **kwargs):
        counter["i"] += 1
        return AIMessage(
            id=f"ai-{counter['i']}",
            content="",
            tool_calls=[
                {
                    "name": "python_repl",
                    "args": {"code": "print(1)"},
                    "id": f"tc-{counter['i']}",
                }
            ],
        )

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = make_loop_response
    mock_llm_factory.return_value = mock_llm

    graph = build_agent_graph()
    result = graph.invoke(
        {"messages": [HumanMessage(content="loop forever")], "iteration_count": 0},
        config={"configurable": {"thread_id": "test-loop"}, "recursion_limit": 50},
    )

    # Iteration count strictly equals or exceeds max_iterations on hard-stop.
    assert result["iteration_count"] >= 5


@patch("app.agent.graph._get_llm_with_tools")
def test_graph_terminates_when_no_tool_calls(mock_llm_factory):
    """If LLM returns a final message without tool_calls, graph stops at iter 1."""
    final_response = AIMessage(content="Final answer.")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = final_response
    mock_llm_factory.return_value = mock_llm

    graph = build_agent_graph()
    result = graph.invoke(
        {"messages": [HumanMessage(content="trivial")], "iteration_count": 0},
        config={"configurable": {"thread_id": "test-trivial"}},
    )

    assert result["iteration_count"] == 1
    assert result["messages"][-1].content == "Final answer."

"""
CrossContext - Member 2 Isolated Test Suite: Agent Real Reasoning Loop & Guardrails
Tests tool dispatch, dynamic multi-hop reasoning, lifecycle guardrails, and session memory.
"""

import sys
import asyncio
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent_orchestrator.agent import CrossContextAgent
from agent_orchestrator.hooks import LifecycleGuardrails
from agent_orchestrator.session_manager import SessionManager
from agent_orchestrator.model_provider import ModelProvider
from mcp_server.parsers.treesitter_engine import TreeSitterEngine
from mcp_server.parsers.scip_indexer import CrossRepoLinker


@pytest.fixture
def agent_instance():
    """Creates a seeded agent with multi-repo AST graph in memory."""
    agent = CrossContextAgent(":memory:")
    backend_dir = str(PROJECT_ROOT / "testbed" / "repo_auth_core")
    frontend_dir = str(PROJECT_ROOT / "testbed" / "repo_frontend_portal")
    sdk_dir = str(PROJECT_ROOT / "testbed" / "repo_shared_sdk")

    parser = TreeSitterEngine()
    linker = CrossRepoLinker()

    b_nodes, b_edges = parser.parse_directory("repo_auth_core", backend_dir)
    f_nodes, f_edges = parser.parse_directory("repo_frontend_portal", frontend_dir)
    s_nodes, s_edges = parser.parse_directory("repo_shared_sdk", sdk_dir)

    all_nodes = b_nodes + f_nodes + s_nodes
    cross_edges = linker.link_repositories(all_nodes)

    agent.tool_manager.graph_store.insert_nodes(all_nodes)
    agent.tool_manager.graph_store.insert_edges(b_edges + f_edges + s_edges + cross_edges)
    return agent


def test_member2_multi_turn_tool_dispatch(agent_instance):
    """Verifies that the agent executes genuine multi-hop MCP tool calls."""
    result = asyncio.run(agent_instance.run(
        "Find all callers of verify_legacy_auth and propose migration to generate_v2_token."
    ))
    assert result["status"] == "success"
    assert result["turns"] >= 1
    assert result["telemetry"]["total_tool_calls"] >= 1
    assert "traverse_call_graph" in result["telemetry"]["tool_sequence"] or len(result["nodes_touched"]) > 0


def test_member2_cycle_detection_guardrail():
    """Verifies cycle guardrail blocks duplicate identical recursive calls."""
    guardrails = LifecycleGuardrails(max_tool_calls=10, token_budget=10000)
    args = {"root_symbol": "verify_legacy_auth", "max_depth": 3}

    # Call 1 & 2 allowed
    assert guardrails.before_tool_call("traverse_call_graph", args) is True
    guardrails.after_tool_call("traverse_call_graph", args, {"result": "ok"}, 1.0)

    assert guardrails.before_tool_call("traverse_call_graph", args) is True
    guardrails.after_tool_call("traverse_call_graph", args, {"result": "ok"}, 1.0)

    # Call 3 is duplicate -> blocked
    assert guardrails.before_tool_call("traverse_call_graph", args) is False
    report = guardrails.get_safety_report()
    assert report["guardrails_triggered"] is True
    assert any(t["type"] == "cycle_detected" for t in report["triggers"])


def test_member2_token_budget_guardrail():
    """Verifies that token budget exhaustion halts execution."""
    guardrails = LifecycleGuardrails(max_tool_calls=10, token_budget=100)
    assert guardrails.before_tool_call("get_ast_chunk", {"node_id": "test:node:1"}) is True

    # Inject large chunk that exhausts budget (~150 tokens)
    guardrails.after_tool_call("get_ast_chunk", {"node_id": "test:node:1"}, {"code": "x" * 600}, 2.0)

    # Next call blocked
    assert guardrails.before_tool_call("get_ast_chunk", {"node_id": "test:node:2"}) is False
    report = guardrails.get_safety_report()
    assert report["guardrails_triggered"] is True
    assert any(t["type"] == "token_budget_exceeded" for t in report["triggers"])


def test_member2_session_memory_retention(agent_instance):
    """Verifies stateful multi-turn session retention."""
    session_id = "test-session-member2"
    result1 = asyncio.run(agent_instance.run(
        "Inspect repo_auth_core endpoints.",
        session_id=session_id
    ))
    assert result1["status"] == "success"

    result2 = asyncio.run(agent_instance.run(
        "Now inspect which frontend service calls those endpoints.",
        session_id=session_id
    ))
    assert result2["status"] == "success"

    messages = agent_instance.session_manager.get_messages(session_id)
    assert len(messages) >= 4  # 2 user prompts + 2 assistant responses

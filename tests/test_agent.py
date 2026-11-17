"""Tests for the knowledge-graph agent (scripted fake LLM: no Gemini, Neo4j or Chroma)."""

from typing import Any, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.retrievers import BaseRetriever

from src.agents.graph_agent import GraphAgent, build_agent_tools, format_concept, format_search_results


class ScriptedChatModel(BaseChatModel):
    """Returns pre-written AI messages in order, ignoring the input."""

    responses: List[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        message = self.responses[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


class FakeRetriever(BaseRetriever):
    queries: List[str] = []

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        self.queries.append(query)
        return [
            Document(
                page_content="Reset the system after each correction of the tire inflation pressure "
                "and after every tire or wheel change.",
                metadata={"citation": "BMW manual > Displays > Resetting the system, p. 96"},
            )
        ]


CONCEPT_ROWS = [
    {
        "entity": "tire pressure monitor",
        "type": "VehicleSystem",
        "relations": [{"relation": "APPLIES_TO", "direction": "in", "other": "resetting tire pressure monitor"}],
        "mentioned_in": [{"brand": "BMW", "section": "Resetting the system", "page": 96}],
    }
]


def _call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def test_agent_calls_tools_and_returns_cited_answer():
    retriever = FakeRetriever()
    brands_requested = []

    def retriever_for(brand):
        brands_requested.append(brand)
        return retriever

    tools = build_agent_tools(retriever_for, lambda name: CONCEPT_ROWS, default_brand="BMW")
    llm = ScriptedChatModel(
        responses=[
            _call("search_manual", {"query": "reset tire pressure monitor after tire change"}, "c1"),
            _call("explore_concept", {"name": "tire pressure monitor"}, "c2"),
            AIMessage(content="Reset the tire pressure monitor (Displays > Resetting the system, p. 96)."),
        ]
    )

    result = GraphAgent(llm, tools).ask("I got new tires fitted. Anything to do?")

    assert "p. 96" in result.answer
    assert [step["tool"] for step in result.trace] == ["search_manual", "explore_concept"]
    assert retriever.queries == ["reset tire pressure monitor after tire change"]
    assert brands_requested == ["BMW"]  # default brand used when the model passes none
    assert "Resetting the system, p. 96" in result.trace[0]["result_preview"]
    assert "1. search_manual(query=" in result.pretty_trace()


def test_format_search_results_cites_and_truncates():
    doc = Document(page_content="x" * 50, metadata={"citation": "BMW manual > Displays > Status display, p. 97"})
    text = format_search_results([doc], max_chars=10)
    assert text.startswith("Source: BMW: Displays > Status display, p. 97\n")
    assert text.endswith("x" * 10 + " ...")
    assert format_search_results([]) == "No matching passages found."


def test_format_concept_shows_relations_and_pages():
    text = format_concept(CONCEPT_ROWS)
    assert "resetting tire pressure monitor -APPLIES_TO-> this" in text
    assert "BMW: Resetting the system, p. 96" in text
    assert format_concept([]) == "Concept not found in the knowledge graph."

"""
GraphAgent: a tool-calling agent that answers questions from the knowledge graph.

    question ──► LLM decides ──► search_manual(query)      GraphRAG retrieval (vector + concepts,
                    ▲   │                                   rank fusion) -> cited excerpts
                    │   └──────► explore_concept(name)     concept relations + where it is mentioned
                    └────────── tool results
                 ──► answer with citations (chapter > section, page)

The agent may call the tools several times, e.g. one search per part of a
multi-part question ("new tires fitted - what now?" -> search tire change,
then the tire pressure monitor reset). Every call is kept in ``trace`` so the
reasoning path can be inspected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool

SYSTEM_PROMPT = """You are a vehicle owner-manual assistant. Answer ONLY from the tool results.

Tools:
- search_manual: finds manual excerpts with citations. Use it first for every question.
- explore_concept: shows how a concept (system, component, warning light, procedure)
  relates to others and on which pages it appears. Use it when the question is about a
  warning light, what belongs to a system, or when search results point to a concept
  you need to understand better.

How to work:
- Split questions with several parts into separate searches (one per part).
- If the first results do not answer the question, search again with the manual's
  terminology (e.g. "reset tire pressure monitor" instead of "new tires fitted").
- Cite with the chapter, section and page from the excerpt's "Source:" line, in brackets,
  e.g. (Displays > Resetting the system, p. 96). Never cite with numbers like [2].
  Cite once per paragraph or per list of steps, not after every line.
- Put safety warnings from the excerpts first.
- If the tools do not contain the answer, say so. Never use outside knowledge.
- Be concise: short steps or bullet points."""


# ---------------------------------------------------------------------------
# Tool output formatting (plain functions, so they are easy to test)
# ---------------------------------------------------------------------------

def format_search_results(docs: List[Document], max_chars: int = 1200) -> str:
    if not docs:
        return "No matching passages found."
    # No excerpt numbers: they restart with every search, so "[2]" in an answer
    # would be ambiguous. The model cites the source line instead.
    blocks = []
    for doc in docs:
        source = doc.metadata.get("citation", "unknown source").replace(" manual > ", ": ", 1)
        text = doc.page_content.strip()
        if len(text) > max_chars:
            text = text[:max_chars] + " ..."
        blocks.append(f"Source: {source}\n{text}")
    return "\n\n---\n\n".join(blocks)


def format_concept(rows: List[Dict[str, Any]], max_relations: int = 15, max_mentions: int = 10) -> str:
    if not rows:
        return "Concept not found in the knowledge graph."
    out = []
    for row in rows[:3]:  # a name can match several concepts
        out.append(
            {
                "concept": row.get("entity"),
                "type": row.get("type"),
                "relations": [
                    f"{'this' if r['direction'] == 'out' else r['other']} -{r['relation']}-> "
                    f"{r['other'] if r['direction'] == 'out' else 'this'}"
                    for r in (row.get("relations") or [])[:max_relations]
                ],
                "mentioned_in": [
                    f"{m.get('brand')}: {m.get('section')}, p. {m.get('page')}"
                    for m in (row.get("mentioned_in") or [])[:max_mentions]
                ],
            }
        )
    return json.dumps(out, ensure_ascii=False, indent=1)


def build_agent_tools(
    retriever_for: Callable[[Optional[str]], BaseRetriever],
    concept_lookup: Callable[[str], List[Dict[str, Any]]],
    default_brand: Optional[str] = None,
) -> List[BaseTool]:
    """``retriever_for(brand)`` returns a retriever; ``concept_lookup(name)`` returns concept rows
    (``Neo4jStore.entity_context``). Injected, so tests can run without Neo4j or Chroma."""

    @tool
    def search_manual(query: str, brand: Optional[str] = None) -> str:
        """Search the vehicle owner manual. Returns excerpts, each with a "Source:" line
        (chapter > section, page) to cite. Use short, specific queries; call again for each part
        of a multi-part question. `brand` (e.g. 'BMW') restricts the search to one manual."""
        return format_search_results(retriever_for(brand or default_brand).invoke(query))

    @tool
    def explore_concept(name: str) -> str:
        """Look up a vehicle concept (system, component, feature, warning light, procedure,
        specification) in the knowledge graph: its relations to other concepts and the
        manual sections and pages where it is mentioned."""
        return format_concept(concept_lookup(name))

    return [search_manual, explore_concept]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class AgentAnswer:
    question: str
    answer: str
    trace: List[Dict[str, Any]] = field(default_factory=list)  # tool calls in order

    def pretty_trace(self) -> str:
        lines = []
        for i, step in enumerate(self.trace, 1):
            args = ", ".join(f"{k}={v!r}" for k, v in step["args"].items())
            lines.append(f"{i}. {step['tool']}({args}) -> {step['result_preview']}")
        return "\n".join(lines) or "(no tool calls)"


def _text(message) -> str:
    """Message text; Gemini may return content as a list of parts."""
    text = getattr(message, "text", None)
    if isinstance(text, str):  # langchain-core 1.x: property (a str that is also callable)
        return str(text)
    if callable(text):  # older langchain-core: method
        return text()
    content = message.content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


class GraphAgent:
    """Tool-calling agent over the knowledge graph (LangChain ``create_agent``)."""

    def __init__(self, llm, tools: List[BaseTool], system_prompt: str = SYSTEM_PROMPT, max_steps: int = 12):
        from langchain.agents import create_agent

        self.agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
        self.max_steps = max_steps

    @classmethod
    def from_settings(cls, brand: Optional[str] = None) -> "GraphAgent":
        """Wire the agent to the configured Gemini model, Neo4j and the Chroma passage index."""
        from src.agents.llm_factory import LLMFactory
        from src.config import get_settings
        from src.graph.graph_rag import make_graph_rag_retriever
        from src.graph.neo4j_store import Neo4jStore

        settings = get_settings()
        store = Neo4jStore()
        cache: Dict[Optional[str], BaseRetriever] = {}

        def retriever_for(b: Optional[str]) -> BaseRetriever:
            if b not in cache:
                cache[b] = make_graph_rag_retriever(store, brand=b, settings=settings)
            return cache[b]

        tools = build_agent_tools(retriever_for, store.entity_context, default_brand=brand)
        agent = cls(LLMFactory.get_llm(settings.PRIMARY_LLM, temperature=0), tools)
        agent.store = store
        return agent

    def ask(self, question: str) -> AgentAnswer:
        state = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": 2 * self.max_steps + 1},
        )
        messages = state["messages"]
        results = {m.tool_call_id: m for m in messages if getattr(m, "type", "") == "tool"}
        trace = []
        for m in messages:
            for call in getattr(m, "tool_calls", None) or []:
                result = _text(results[call["id"]]) if call.get("id") in results else ""
                preview = result.split("\n", 1)[0][:120]
                trace.append({"tool": call["name"], "args": call["args"], "result_preview": preview})
        return AgentAnswer(question=question, answer=_text(messages[-1]), trace=trace)

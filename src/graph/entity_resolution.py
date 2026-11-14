"""
Entity resolution: merge duplicate concepts produced by LLM extraction.

Extraction runs passage by passage, so the same concept comes back under
different names or ontology types ("tire pressure monitor" as Feature and as
VehicleSystem, "initializing" vs "initialize tire pressure monitor",
"headlamp" vs "headlight"). Each duplicate splits the MENTIONS edges of one
real concept across several nodes, which breaks graph traversal.

Two stages, cheapest first:

1. Rule-based: names that are equal after normalisation (case, articles,
   plural / -ing / -ed endings) are merged, regardless of type. Free.
2. Semantic: remaining names are embedded; pairs above a cosine threshold are
   judged by an LLM ("same concept?"). Verdicts are cached, so re-runs and
   new manuals only pay for new pairs.

The canonical node of a group is its most-mentioned name and type; the other
names are kept as ``aliases`` so search still finds them.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from src.graph.ontology import DOMAIN_LABELS
from src.graph.schemas import GraphNode, GraphRelation, KnowledgeGraph, entity_id

logger = logging.getLogger(__name__)

# When a merged group has a tie in mentions, prefer the more specific type.
TYPE_PRIORITY: Sequence[str] = (
    "WarningIndicator",
    "Procedure",
    "MaintenanceTask",
    "SafetyInstruction",
    "Specification",
    "Feature",
    "VehicleSystem",
    "Component",
)

_STOPWORDS = frozenset({"the", "a", "an"})


# ---------------------------------------------------------------------------
# Stage 1: rule-based normalisation
# ---------------------------------------------------------------------------

def _stem(word: str) -> str:
    """Tiny suffix stripper; enough to align 'initializing'/'initialize', 'airbags'/'airbag'."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix == "s" and word.endswith(("ss", "us", "is")):
                break
            word = word[: -len(suffix)]
            if suffix in ("ing", "ed") and len(word) > 3 and word[-1] == word[-2]:
                word = word[:-1]  # resetting -> resett -> reset
            break
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]  # initialize / initializ(ing)
    return word


def normalize_name(name: str) -> str:
    """Key under which spelling variants of one name collide."""
    words = re.sub(r"[^a-z0-9]+", " ", name.lower()).split()
    return " ".join(_stem(w) for w in words if w not in _STOPWORDS)


# ---------------------------------------------------------------------------
# Stage 2: LLM judge for semantic candidates
# ---------------------------------------------------------------------------

# Bump when the prompt changes: cached verdicts from an older prompt are discarded.
JUDGE_VERSION = "2"

JUDGE_SYSTEM_PROMPT = """You merge duplicate concepts in a knowledge graph built from vehicle owner manuals.
For each numbered pair decide whether both names refer to the SAME real-world concept.
Each side has its ontology type and a one-sentence description from the manual; use them.

Answer same=true only for true synonyms or naming variants, e.g.
- 'tire pressure monitor' / 'tire pressure monitoring system'
- 'headlamp' / 'headlight'
- 'reset tire pressure monitor' / 'reinitialize tire pressure monitor'

Answer same=false when:
- one is more specific than the other ('fog lamp' vs 'front fog lamp', 'airbag' vs 'knee airbag')
- they are different kinds of thing ('tire pressure monitor' (a system) vs 'checking tire pressure' (a procedure))
- they are related but distinct ('high beam' vs 'low beam', 'summer tire' vs 'winter tire')
- they are different named systems, even if they do a similar job: manuals describe
  alternative equipment variants ('flat tire monitor' (FTM) vs 'tire pressure monitor' (TPM))
- one is generic and the other a specific quantity or setting ('distance' vs 'driving distance',
  'setting the date' vs 'setting appointment date', 'engine' vs 'combustion engine' in a hybrid)
When unsure, answer false: a wrong merge is worse than a missed one."""


class PairVerdict(BaseModel):
    pair: int = Field(description="Number of the pair as given in the input.")
    same: bool = Field(description="True only if both names denote the same concept.")


class PairVerdicts(BaseModel):
    verdicts: List[PairVerdict] = Field(default_factory=list)


@dataclass(frozen=True)
class JudgeItem:
    a: str
    type_a: str
    desc_a: str
    b: str
    type_b: str
    desc_b: str


def pair_key(a: str, b: str) -> str:
    return " || ".join(sorted((a, b)))


class LLMPairJudge:
    """Batch 'same concept?' decisions through a structured-output chat model."""

    def __init__(self, llm=None, batch_size: int = 25):
        from langchain_core.prompts import ChatPromptTemplate

        if llm is None:
            from src.agents.llm_factory import LLMFactory

            llm = LLMFactory.get_primary_llm()
        prompt = ChatPromptTemplate.from_messages(
            [("system", JUDGE_SYSTEM_PROMPT), ("human", "{pairs}")]
        )
        self.chain = prompt | llm.with_structured_output(PairVerdicts)
        self.batch_size = batch_size

    def __call__(self, pairs: List["JudgeItem"]) -> List[bool]:
        """Returns one bool per pair (missing answers count as 'not the same')."""
        results: List[bool] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            text = "\n".join(
                f"{i}. A: '{p.a}' ({p.type_a}) - {p.desc_a or 'no description'}\n"
                f"   B: '{p.b}' ({p.type_b}) - {p.desc_b or 'no description'}"
                for i, p in enumerate(batch)
            )
            out = self.chain.invoke({"pairs": text})
            out = out if isinstance(out, PairVerdicts) else PairVerdicts.model_validate(out)
            by_index = {v.pair: v.same for v in out.verdicts}
            results.extend(by_index.get(i, False) for i in range(len(batch)))  # missing -> no merge
        return results


def candidate_pairs(names: List[str], threshold: float, embed: Callable[[List[str]], List[List[float]]]):
    """All name pairs whose embedding cosine similarity is >= threshold, most similar first."""
    import numpy as np

    if len(names) < 2:
        return []
    vectors = np.asarray(embed(names), dtype=float)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    sims = vectors @ vectors.T
    rows, cols = np.where(np.triu(sims, k=1) >= threshold)
    pairs = [(names[i], names[j], float(sims[i, j])) for i, j in zip(rows, cols)]
    return sorted(pairs, key=lambda p: -p[2])


def default_embedder() -> Callable[[List[str]], List[List[float]]]:
    from src.rag.embeddings import EmbeddingsFactory

    return EmbeddingsFactory.get_default_embeddings().embed_documents


# ---------------------------------------------------------------------------
# Union-find + applying merges
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass
class ResolutionReport:
    entities_before: int = 0
    entities_after: int = 0
    rule_merges: int = 0
    semantic_candidates: int = 0
    semantic_merges: int = 0
    llm_pairs_judged: int = 0
    groups: List[Dict] = field(default_factory=list)  # merged groups, for human review

    def summary(self) -> Dict[str, int]:
        return {
            "entities_before": self.entities_before,
            "entities_after": self.entities_after,
            "rule_merges": self.rule_merges,
            "semantic_candidates": self.semantic_candidates,
            "semantic_merges": self.semantic_merges,
            "llm_pairs_judged": self.llm_pairs_judged,
        }


class EntityResolver:
    """Merge duplicate domain entities in a KnowledgeGraph."""

    def __init__(
        self,
        judge: Optional[Callable[[List[JudgeItem]], List[bool]]] = None,
        embed: Optional[Callable[[List[str]], List[List[float]]]] = None,
        threshold: float = 0.75,
        cache_path: Optional[Path] = None,
        semantic: bool = True,
        overrides: Optional[Dict[str, List[List[str]]]] = None,
    ):
        """``overrides``: {"never_merge": [[a, b], ...], "always_merge": [[a, b], ...]}
        (entity names). Human decisions; they win over rules and the LLM judge."""
        self.judge = judge
        self.embed = embed
        self.threshold = threshold
        self.cache_path = cache_path
        self.semantic = semantic
        overrides = overrides or {}
        self.never = {frozenset(map(str.lower, pair)) for pair in overrides.get("never_merge", [])}
        self.always = [tuple(map(str.lower, pair)) for pair in overrides.get("always_merge", [])]
        self.cache: Dict[str, bool] = {}
        if cache_path and cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("version") == JUDGE_VERSION:
                self.cache = data.get("verdicts", {})  # older prompt versions are re-judged

    @staticmethod
    def load_overrides(path: Optional[Path]) -> Dict[str, List[List[str]]]:
        if path and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def resolve(self, graph: KnowledgeGraph) -> Tuple[KnowledgeGraph, ResolutionReport]:
        entities = [n for n in graph.nodes.values() if n.label in DOMAIN_LABELS]
        report = ResolutionReport(entities_before=len(entities))
        mentions = Counter(r.target for r in graph.relations.values() if r.type == "MENTIONS")
        uf = _UnionFind()
        self._names: Dict[str, set] = {}  # union-find root -> all names in that group
        for node in entities:
            uf.find(node.id)
            self._names.setdefault(node.id, set()).add(node.properties.get("name", ""))

        # Stage 1: same normalised name -> same concept (any type).
        by_key: Dict[str, List[GraphNode]] = defaultdict(list)
        for node in entities:
            by_key[normalize_name(node.properties.get("name", ""))].append(node)
        for group in by_key.values():
            for other in group[1:]:
                report.rule_merges += self._union(uf, group[0].id, other.id)

        # Stage 2: semantic candidates between stage-1 groups, judged by the LLM.
        if self.semantic:
            report.semantic_merges, report.semantic_candidates, report.llm_pairs_judged = self._semantic(
                entities, mentions, uf
            )

        # Human overrides: forced merges (never_merge is enforced inside _union).
        by_name: Dict[str, str] = {n.properties.get("name", ""): n.id for n in entities}
        for a, b in self.always:
            if a in by_name and b in by_name:
                self._union(uf, by_name[a], by_name[b], forced=True)

        new_graph, report.groups = self._apply(graph, entities, mentions, uf)
        report.entities_after = sum(1 for n in new_graph.nodes.values() if n.label in DOMAIN_LABELS)
        return new_graph, report

    def _union(self, uf: "_UnionFind", a: str, b: str, forced: bool = False) -> int:
        """Merge two groups unless a never_merge pair would end up in one group
        (also blocks indirect chains A=C, C=B when A/B must stay apart)."""
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            return 0
        names_a, names_b = self._names[ra], self._names[rb]
        if any(frozenset((x, y)) in self.never for x in names_a for y in names_b):
            return 0
        uf.union(ra, rb)
        self._names[uf.find(ra)] = names_a | names_b
        return 0 if forced else 1

    # -- stage 2 -------------------------------------------------------------
    def _semantic(self, entities, mentions, uf) -> Tuple[int, int, int]:
        groups: Dict[str, List[GraphNode]] = defaultdict(list)
        for node in entities:
            groups[uf.find(node.id)].append(node)
        info: Dict[str, Tuple[str, str, str]] = {}  # canonical name -> (root, type, description)
        for root, members in groups.items():
            name, label = self._canonical(members, mentions)
            desc = next((m.properties.get("description", "") for m in members
                         if m.properties.get("name") == name and m.properties.get("description")), "")
            info[name] = (root, label, desc)
        names = sorted(info)

        embed = self.embed or default_embedder()
        pairs = [
            (a, b, s) for a, b, s in candidate_pairs(names, self.threshold, embed)
            if frozenset((a, b)) not in self.never  # humans already decided
        ]
        todo = [(a, b) for a, b, _ in pairs if pair_key(a, b) not in self.cache]
        if todo:
            judge = self.judge or LLMPairJudge()
            items = [JudgeItem(a, info[a][1], info[a][2], b, info[b][1], info[b][2]) for a, b in todo]
            for (a, b), same in zip(todo, judge(items)):
                self.cache[pair_key(a, b)] = bool(same)
            self._save_cache()

        merges = 0
        for a, b, _ in pairs:
            if self.cache.get(pair_key(a, b)):
                merges += self._union(uf, info[a][0], info[b][0])
        return merges, len(pairs), len(todo)

    def _save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": JUDGE_VERSION, "verdicts": self.cache}
            self.cache_path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")

    # -- apply -----------------------------------------------------------------
    @staticmethod
    def _canonical(members: List[GraphNode], mentions: Counter) -> Tuple[str, str]:
        """(name, type) for a group: most-mentioned name; type by total mentions."""
        name_score: Counter = Counter()
        type_score: Counter = Counter()
        for node in members:
            weight = max(mentions[node.id], 1)
            name_score[node.properties.get("name", "")] += weight
            type_score[node.label] += weight
        name = min(name_score, key=lambda n: (-name_score[n], len(n), n))
        label = min(type_score, key=lambda t: (-type_score[t], TYPE_PRIORITY.index(t)))
        return name, label

    def _apply(self, graph, entities, mentions, uf) -> Tuple[KnowledgeGraph, List[Dict]]:
        groups: Dict[str, List[GraphNode]] = defaultdict(list)
        for node in entities:
            groups[uf.find(node.id)].append(node)

        remap: Dict[str, str] = {}
        canonical_nodes: Dict[str, GraphNode] = {}
        review: List[Dict] = []
        for members in groups.values():
            name, label = self._canonical(members, mentions)
            new_id = entity_id(label, name)
            aliases = sorted({m.properties.get("name", "") for m in members} - {name})
            best = max(members, key=lambda m: (mentions[m.id], m.properties.get("name") == name))
            props = {"name": name, "description": best.properties.get("description", "")}
            if aliases:
                props["aliases"] = aliases
            canonical_nodes[new_id] = GraphNode(id=new_id, label=label, properties=props)
            for m in members:
                remap[m.id] = new_id
            if len(members) > 1:
                review.append(
                    {
                        "canonical": f"{label}:{name}",
                        "merged": sorted(f"{m.label}:{m.properties.get('name')}" for m in members),
                    }
                )

        new_graph = KnowledgeGraph()
        for node in graph.nodes.values():
            if node.label not in DOMAIN_LABELS:
                new_graph.add_node(node)
        for node in canonical_nodes.values():
            new_graph.add_node(node)
        for rel in graph.relations.values():
            source, target = remap.get(rel.source, rel.source), remap.get(rel.target, rel.target)
            if source == target:
                continue  # merged into one concept
            new_graph.add_relation(
                GraphRelation(source=source, target=target, type=rel.type, properties=rel.properties)
            )
        review.sort(key=lambda g: -len(g["merged"]))
        return new_graph, review

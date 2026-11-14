# Knowledge Graph & Ontology Layer

Owner manuals are turned into a knowledge graph that the RAG agent can query
alongside vector search. The graph is defined by an OWL ontology, stored in
Neo4j for GraphRAG, and exported to RDF so it can be queried with SPARQL.

## Design decision: Neo4j property graph + OWL ontology + RDF export

| Option | Strength | Weakness for this project |
|---|---|---|
| **Neo4j (property graph, Cypher)** | De-facto store for LLM GraphRAG; LangChain/LlamaIndex integrations; full-text + vector indexes; easy traversal for retrieval | No formal semantics on its own |
| **RDF/OWL + SPARQL** | W3C standards, formal schema, class hierarchy reasoning, interoperable (used in enterprise, manufacturing and automotive data spaces) | Weaker LLM-tooling ecosystem; slower to iterate for retrieval |

**Chosen:** both, with a clear split of responsibilities.

- **OWL ontology** ([`ontology/automotive.ttl`](../ontology/automotive.ttl)) is the *schema contract*. It defines the classes, relations and properties. The Python mirror is [`src/graph/ontology.py`](../src/graph/ontology.py), and a test fails if the two drift apart.
- **Neo4j** is the *runtime store* for GraphRAG retrieval and agent tools.
- **RDF/Turtle export** of the same graph supports *SPARQL*, standards-based exchange, and import into triple stores or back into Neo4j via `n10s`.
- **One in-memory model** (`KnowledgeGraph`) feeds both stores, so they never disagree.

## Graph model

```
Document ─HAS_CHAPTER→ Chapter ─HAS_SECTION→ Section ─HAS_PASSAGE→ Passage ─NEXT_PASSAGE→ Passage
                                                                     │
                                                                  MENTIONS
                                                                     ↓
   Component ─PART_OF→ VehicleSystem      WarningIndicator ─INDICATES→ Component/System
   Procedure / MaintenanceTask / SafetyInstruction / Specification ─APPLIES_TO→ Component/Feature/System
```

- **Structure layer** (rules, no LLM): built from MinerU `*_structure_detected.json`. It filters MinerU noise such as page numbers, callout digits, button codes and numbered steps that were misdetected as headings.
- **Semantic layer** (LLM): `with_structured_output(ExtractionResult)` limits the model to ontology types. Entities get **canonical, brand-independent ids** (`Feature:tire-pressure-monitor`), so the same concept in the BMW and Volvo manuals becomes one node. That node is what makes cross-brand questions answerable.
- **Provenance**: every entity links back via `MENTIONS` to the passage, section and page it came from, so answers can be cited.
- **Cost control**: extraction results are cached by passage-text hash, so a rerun never pays twice.

## Usage

```bash
# 1. Build (structure only, free)
python scripts/build_knowledge_graph.py --brand BMW \
  --structure data/automotive_corpus/parsed/BMW/BMW/auto/BMW_structure_detected.json

# 2. Add LLM entity extraction for one chapter
python scripts/build_knowledge_graph.py --brand BMW --structure <...> --extract 30 --chapter "Wheels and tires"

# 3. SPARQL over the RDF export (no server needed)
python scripts/query_knowledge_graph.py sparql ontology/queries/chapter_overview.rq --brand BMW
python scripts/query_knowledge_graph.py sparql ontology/queries/concepts_by_class.rq

# 4. Neo4j: start, load, query
docker compose up -d neo4j                      # Browser: http://localhost:7474
python scripts/build_knowledge_graph.py --brand BMW --structure <...> --neo4j
python scripts/query_knowledge_graph.py search "tire pressure" --brand BMW
python scripts/query_knowledge_graph.py concept "tire pressure monitor"
```

In code, `GraphRetriever` is a standard LangChain retriever (works with `RAGChain(retriever=...)`), and `build_graph_tools(store)` returns `search_manuals` / `explore_concept` tools for the agent.

## Current results (BMW manual)

| Layer | Output |
|---|---|
| Structure | 1 document, 33 chapters, 1,564 sections, 1,311 passages, 21.7k RDF triples |
| Semantic (5 passages, `gemini-3.5-flash-lite`) | 18 entities, 15 domain relations, 25 provenance links |

Known extraction issues, which are the targets for the extraction eval: near-duplicate concepts ("tire pressure" / "tire inflation pressure") and some mistyped entities.

## Next steps

1. Entity resolution: merge near-duplicates using embedding similarity and an LLM check.
2. Extraction eval: about 30 hand-labelled passages, scored on entity and relation precision/recall per model (Claude vs Llama vs Gemini).
3. Retrieval eval: vector vs graph vs hybrid on multi-hop and cross-brand questions.
4. Parse and load the Mercedes and Volvo manuals to enable cross-brand comparisons.
5. Expose `search_manuals` / `explore_concept` through the MCP server.

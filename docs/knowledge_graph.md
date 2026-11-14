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
- **Semantic layer** (LLM): `with_structured_output(ExtractionResult)` limits the model to ontology types. Entities get **canonical, brand-independent ids** (`VehicleSystem:tire-pressure-monitor`), so the same concept in the BMW and Volvo manuals becomes one node. That node is what makes cross-brand questions answerable.
- **Provenance**: every entity links back via `MENTIONS` to the passage, section and page it came from, so answers can be cited.
- **Cost control**: extraction results are cached by passage-text hash, so a rerun never pays twice.

## Entity resolution

Extraction runs passage by passage, so one concept comes back under several names or types
("tire pressure monitor" as `Feature` and as `VehicleSystem`, "initializing" vs "initialize tire
pressure monitor"). Each duplicate splits the `MENTIONS` edges of one real concept, which breaks
graph traversal. [`src/graph/entity_resolution.py`](../src/graph/entity_resolution.py) merges them
in two stages, cheapest first:

1. **Rules (free):** names that are equal after normalisation (case, articles, plural/-ing/-ed endings) are merged, whatever their type.
2. **Semantic:** the remaining names are embedded (MiniLM). Pairs with cosine similarity ≥ 0.75 go to an LLM judge that sees each concept's type and description. Verdicts are cached, and the cache is versioned with the prompt.
3. **Human overrides** in [`ontology/entity_overrides.json`](../ontology/entity_overrides.json) win over both. `never_merge` also blocks indirect chains (A=C and C=B cannot pull A and B together).

The canonical node is the most-mentioned name and type; the other names are kept as `aliases`
(`skos:altLabel` in RDF), so search still finds them. Each run writes the merged groups to
`data/knowledge_graph/<brand>_entity_merges.json` for review.

Lesson from the BMW run: with names only, the judge merged *flat tire monitor* with *tire pressure
monitor*. In this car they are two different systems (pp. 98 and 95). Adding descriptions to the
prompt and the override file fixed it; every merge is now reviewed.

## GraphRAG retrieval

[`src/graph/graph_rag.py`](../src/graph/graph_rag.py) combines four ranked lists with reciprocal rank fusion (weight / (60 + rank)):

| List | Source | Weight |
|---|---|---|
| vector | Chroma top 20 over the graph's passages, filtered by brand | 1.0 |
| concepts | question → closest concept names/aliases → passages mentioning them | 1.0 |
| expansion | passages sharing specific concepts with the top 4 vector hits | 0.5 |
| neighbours | previous / next passage of the top 4 hits, same chapter only | 0.5 |

Concepts mentioned in more than 8 passages (such as *tire*) are treated as too generic to follow.
A passage found by several lists rises; one found only by the noisy expansion rarely makes the top
10. The naive variant (vector top 4 + expansion appended, `fusion=False`) is kept for comparison.

Benchmark ([`scripts/compare_retrievers.py`](../scripts/compare_retrievers.py), 28 questions, section-level gold):

| Setup | Recall | Fully answered |
|---|---|---|
| vector@4 (metadata baseline) | 71% | 18/28 |
| vector@10 (same budget) | 82% | 22/28 |
| graph, appended | 73% | 19/28 |
| graph, rank fusion | 86% | 23/28 |

## Usage

```bash
# 1. Build (structure only, free)
python scripts/build_knowledge_graph.py --brand BMW \
  --structure data/automotive_corpus/parsed/BMW/BMW/auto/BMW_structure_detected.json

# 2. Add LLM extraction (repeat --chapter for several), merge duplicates, load Neo4j from scratch
python scripts/build_knowledge_graph.py --brand BMW --structure <...> --extract 200 \
  --chapter "Wheels and tires" --chapter "Displays" --resolve --neo4j --reset-neo4j

# 3. SPARQL over the RDF export (no server needed)
python scripts/query_knowledge_graph.py sparql ontology/queries/chapter_overview.rq --brand BMW
python scripts/query_knowledge_graph.py sparql ontology/queries/concepts_by_class.rq

# 4. Neo4j: start, load, query
docker compose up -d neo4j                      # Browser: http://localhost:7474
python scripts/build_knowledge_graph.py --brand BMW --structure <...> --neo4j
python scripts/query_knowledge_graph.py search "tire pressure" --brand BMW
python scripts/query_knowledge_graph.py concept "tire pressure monitor"
```

```bash
# 5. Retrieval benchmark
python scripts/compare_retrievers.py --brand BMW
```

If Windows refuses ports 7474/7687 (`bind: access forbidden`, reserved by Hyper-V/WSL), set
`NEO4J_HTTP_PORT=17474`, `NEO4J_BOLT_PORT=17687` and `NEO4J_URI=bolt://localhost:17687` in `.env`.

In code, `GraphRAGRetriever` and `GraphRetriever` are standard LangChain retrievers (they work with `RAGChain(retriever=...)`), and `build_graph_tools(store)` returns `search_manuals` / `explore_concept` tools for an agent.

## Current results (BMW manual)

| Layer | Output |
|---|---|
| Structure | 1 document, 33 chapters, 1,564 sections, 1,311 passages |
| Semantic (`gemini-3.5-flash-lite`) | 2 chapters extracted (83 passages, 0 failures); 231 raw concepts merged into 202 |
| Stores | about 3.1k Neo4j nodes and 4.8k relationships; 23k RDF triples |

Known issues:
- Concepts cover only 2 of 33 chapters.
- Heading levels are flattened, so a section such as "System limits" does not record its parent system.
- A fixed hub cut-off (8 mentions) now excludes *tire pressure monitor* itself after merging. It should become relative.
- A few extraction errors remain, e.g. TPM mentions tagged in flat-tire-monitor passages.

## Next steps

1. Larger benchmark (80-100 questions, dev/test split, real user questions) with latency per stage.
2. Heading paths (`HAS_SUBSECTION`) in the structure layer and in passage embeddings; rarity-weighted concepts.
3. Neo4j as the single retrieval engine: native vector index, precomputed mention counts, one Cypher query per question.
4. Extraction for the whole manual (prompt finalised first, so the cache is not invalidated), then more manuals for cross-brand questions.
5. Cross-encoder reranking; an agent step that rewrites multi-part questions only when needed.
6. Extraction eval: hand-labelled passages scored on entity and relation precision/recall.

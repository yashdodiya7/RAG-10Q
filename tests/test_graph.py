"""Tests for the knowledge graph layer (no Neo4j or LLM required)."""

from pathlib import Path

import pytest
from rdflib import OWL, RDF, URIRef

from src.graph import RDFStore, StructureGraphBuilder
from src.graph.entity_extractor import add_extraction
from src.graph.ontology import (
    DATATYPE_PROPERTIES,
    NAMESPACE,
    NODE_LABELS,
    RELATION_TO_PROPERTY,
)
from src.graph.schemas import ExtractionResult, GraphNode
from src.graph.structure_builder import clean_text, is_heading_text

ONTOLOGY = Path(__file__).resolve().parent.parent / "ontology" / "automotive.ttl"

BLOCKS = {
    "0": {"type": "page_number", "level": 1, "page_idx": 0, "text": "1"},
    "1": {"type": "text", "level": 1, "page_idx": 0, "text": "Wheels and tires"},
    "2": {"type": "text", "level": 2, "page_idx": 0, "text": "Tire inflation pressure"},
    "3": {"type": "text", "level": 4, "page_idx": 0, "text": "Check the tire pressure regularly, at least twice a month."},
    "4": {"type": "text", "level": 1, "page_idx": 1, "text": "10"},  # callout digit, not a chapter
    "5": {"type": "text", "level": 2, "page_idx": 1, "text": "1. Press the button."},  # step, not a section
    "6": {"type": "text", "level": 2, "page_idx": 1, "text": "Tire Pressure Monitor"},
    "7": {"type": "aside_text", "level": 4, "page_idx": 1, "text": "The system monitors the inflation pres‐ sure."},
    "8": {"type": "image", "level": 4, "page_idx": 1, "text": ""},
}


@pytest.fixture
def kg():
    return StructureGraphBuilder().build(
        BLOCKS, brand="BMW", document_type="owner_manual", title="BMW manual", source_file="t.json"
    )


def test_ontology_matches_code():
    """Every label/relation/property used in code is declared in the OWL file."""
    store = RDFStore(ontology_path=ONTOLOGY)
    g = store.graph
    for label in NODE_LABELS:
        assert (URIRef(NAMESPACE + label), RDF.type, OWL.Class) in g, label
    for prop in RELATION_TO_PROPERTY.values():
        assert (URIRef(NAMESPACE + prop), RDF.type, OWL.ObjectProperty) in g, prop
    for prop in DATATYPE_PROPERTIES:
        assert (URIRef(NAMESPACE + prop), RDF.type, OWL.DatatypeProperty) in g, prop


def test_heading_filters():
    assert is_heading_text("Wheels and tires")
    assert not is_heading_text("10")
    assert not is_heading_text("AUTOH")
    assert not is_heading_text("1. Turn.")
    assert clean_text("Con‐ trol  18\\*") == "Control 18"


def test_structure_hierarchy(kg):
    stats = kg.stats()
    assert stats["nodes"] == {"Document": 1, "Chapter": 1, "Section": 2, "Passage": 2}
    assert stats["relations"]["NEXT_PASSAGE"] == 1

    sections = sorted(n.properties["title"] for n in kg.nodes_by_label("Section"))
    assert sections == ["Tire Pressure Monitor", "Tire inflation pressure"]

    passages = sorted(kg.nodes_by_label("Passage"), key=lambda n: n.properties["order"])
    # The numbered step is body text of the section it follows, not a new section.
    assert passages[0].properties["text"].endswith("twice a month. 1. Press the button.")
    assert passages[0].properties["pageEnd"] == 2  # 1-based
    assert passages[1].properties["text"] == "The system monitors the inflation pressure."


def test_unknown_label_rejected():
    with pytest.raises(ValueError):
        GraphNode(id="x", label="Spaceship")


def test_entities_are_canonical_across_brands(kg):
    passage_ids = [n.id for n in kg.nodes_by_label("Passage")]
    result = ExtractionResult.model_validate(
        {
            "entities": [
                {"name": "Tire Pressure Monitor", "type": "Feature"},
                {"name": "tire", "type": "Component"},
            ],
            "relations": [{"source": "tire pressure monitor", "target": "tire", "type": "APPLIES_TO"}],
        }
    )
    for pid in passage_ids:  # same concept found in two passages -> one node, two MENTIONS
        add_extraction(kg, pid, result)

    feature = kg.nodes["Feature:tire-pressure-monitor"]
    assert feature.properties["name"] == "tire pressure monitor"
    mentions = [r for r in kg.relations.values() if r.type == "MENTIONS" and r.target == feature.id]
    assert len(mentions) == 2
    assert ("Feature:tire-pressure-monitor", "APPLIES_TO", "Component:tire") in kg.relations


def test_rdf_export_and_sparql(kg, tmp_path):
    store = RDFStore(ontology_path=ONTOLOGY)
    assert store.add_knowledge_graph(kg) > 0
    store.save(tmp_path / "kg.ttl")

    reloaded = RDFStore(ontology_path=ONTOLOGY)
    reloaded.load(tmp_path / "kg.ttl")
    query = (Path(ONTOLOGY).parent / "queries" / "chapter_overview.rq").read_text().replace("$brand", "BMW")
    rows = reloaded.sparql(query)
    assert rows == [{"chapter": "Wheels and tires", "page": "1", "sections": "2", "passages": "2"}]


def test_graph_json_roundtrip(kg):
    from src.graph import KnowledgeGraph

    clone = KnowledgeGraph.from_dict(kg.to_dict())
    assert clone.stats() == kg.stats()


def test_cypher_write_guard():
    from src.graph.neo4j_store import _WRITE_CLAUSES

    assert _WRITE_CLAUSES.search("MATCH (n) DETACH DELETE n")
    assert _WRITE_CLAUSES.search("CALL apoc.periodic.iterate('x','y',{})")
    assert not _WRITE_CLAUSES.search("MATCH (c:Chapter) RETURN c.title AS created_on")


# ---------------------------------------------------------------------------
# Entity resolution (fake embedder + judge: no model download, no LLM)
# ---------------------------------------------------------------------------

def _extract(kg, passage_id, entities, relations=()):
    add_extraction(
        kg,
        passage_id,
        ExtractionResult.model_validate(
            {
                "entities": [{"name": n, "type": t} for n, t in entities],
                "relations": [{"source": s, "target": d, "type": r} for s, r, d in relations],
            }
        ),
    )


def test_normalize_name_aligns_variants():
    from src.graph.entity_resolution import normalize_name

    assert normalize_name("Initializing tire pressure monitor") == normalize_name("initialize tire pressure monitor")
    assert normalize_name("calibrating the front seats") == normalize_name("calibrate front seat")
    assert normalize_name("resetting trip odometer") == normalize_name("reset trip odometer")
    assert normalize_name("front fog lamp") != normalize_name("rear fog lamp")


def test_entity_resolution_merges_and_rewires(kg, tmp_path):
    from src.graph.entity_resolution import EntityResolver

    passages = [n.id for n in kg.nodes_by_label("Passage")]
    _extract(kg, passages[0], [("tire pressure monitor", "VehicleSystem"), ("tire", "Component")],
             [("tire pressure monitor", "RELATED_TO", "tire")])
    _extract(kg, passages[1], [("tire pressure monitor", "Feature"),            # same name, other type
                               ("tire pressure monitoring system", "VehicleSystem"),  # synonym
                               ("initializing tire pressure monitor", "Procedure"),
                               ("initialize tire pressure monitor", "Procedure"),     # spelling variant
                               ("front fog lamp", "Component"), ("rear fog lamp", "Component")])

    synonyms = {frozenset({"tire pressure monitor", "tire pressure monitoring system"})}
    judged = []

    def judge(pairs):
        judged.extend(pairs)
        return [frozenset({p.a, p.b}) in synonyms for p in pairs]

    def embed(names):  # everything 'similar' so the judge sees every pair
        return [[1.0, 0.0] for _ in names]

    cache = tmp_path / "verdicts.json"
    new_kg, report = EntityResolver(judge=judge, embed=embed, cache_path=cache).resolve(kg)
    names = {n.properties["name"]: n for n in new_kg.nodes.values() if "name" in n.properties}

    tpm = names["tire pressure monitor"]
    assert tpm.label == "VehicleSystem"
    assert set(tpm.properties["aliases"]) == {"tire pressure monitoring system"}
    assert "tire pressure monitoring system" not in names
    assert "front fog lamp" in names and "rear fog lamp" in names  # judge said no
    assert report.rule_merges == 2 and report.semantic_merges == 1
    # Both passages now point at the single merged node; relation survived the rewire.
    mentions = [r for r in new_kg.relations.values() if r.type == "MENTIONS" and r.target == tpm.id]
    assert {r.source for r in mentions} == set(passages[:2])
    assert any(r.source == tpm.id and r.type == "RELATED_TO" for r in new_kg.relations.values())
    # Verdicts are cached: a second run makes no judge calls and gives the same result.
    calls = len(judged)
    again, report2 = EntityResolver(judge=judge, embed=embed, cache_path=cache).resolve(kg)
    assert len(judged) == calls and report2.llm_pairs_judged == 0
    assert report2.entities_after == report.entities_after


def test_never_merge_blocks_direct_and_chained_merges(kg):
    from src.graph.entity_resolution import EntityResolver

    passage = kg.nodes_by_label("Passage")[0].id
    _extract(kg, passage, [("flat tire monitor", "VehicleSystem"),
                           ("tire monitor", "VehicleSystem"),
                           ("tire pressure monitor", "VehicleSystem")])
    overrides = {"never_merge": [["flat tire monitor", "tire pressure monitor"]]}
    say_yes = lambda pairs: [True] * len(pairs)          # judge merges everything
    embed = lambda names: [[1.0, 0.0] for _ in names]

    new_kg, _ = EntityResolver(judge=say_yes, embed=embed, overrides=overrides).resolve(kg)
    names = {n.properties["name"] for n in new_kg.nodes.values() if n.label == "VehicleSystem"}
    # 'tire monitor' may join one side, but FTM and TPM must never end up in the same node.
    assert len(names) == 2
    assert not any(
        {"flat tire monitor", "tire pressure monitor"} <= {n.properties["name"], *n.properties.get("aliases", [])}
        for n in new_kg.nodes.values() if n.label == "VehicleSystem"
    )

"""Which knowledge types each relationship may connect.

The database stores relationships between any two units of the same book; these rules say
which combinations make sense. They are checked by the code that creates relationships
(extraction, review), not by the database, because the type lives on the units.
"""

from dataclasses import dataclass

from app.knowledge.types import KnowledgeType, RelationshipType

_K = KnowledgeType
_CORE = frozenset({_K.CONCEPT, _K.PRINCIPLE, _K.METHOD, _K.FRAMEWORK})
_ALL = frozenset(KnowledgeType)


@dataclass(frozen=True)
class RelationshipRule:
    description: str
    source_types: frozenset[KnowledgeType]
    target_types: frozenset[KnowledgeType]


RELATIONSHIP_RULES: dict[RelationshipType, RelationshipRule] = {
    RelationshipType.SUPPORTS: RelationshipRule(
        "The source (evidence) backs the claim made by the target.", frozenset({_K.EVIDENCE}), _CORE
    ),
    RelationshipType.ILLUSTRATES: RelationshipRule(
        "The source (an example) shows the target in use.", frozenset({_K.EXAMPLE}), _CORE
    ),
    RelationshipType.IMPLEMENTS: RelationshipRule(
        "The source (a method) puts the target into practice.",
        frozenset({_K.METHOD}),
        frozenset({_K.FRAMEWORK, _K.PRINCIPLE}),
    ),
    RelationshipType.DESCRIBES: RelationshipRule(
        "The source (an equation) expresses the target mathematically.", frozenset({_K.EQUATION}), _CORE
    ),
    RelationshipType.PART_OF: RelationshipRule(
        "The source is a component of the target framework.",
        frozenset({_K.CONCEPT, _K.PRINCIPLE, _K.METHOD, _K.EQUATION}),
        frozenset({_K.FRAMEWORK}),
    ),
    RelationshipType.DEPENDS_ON: RelationshipRule(
        "The source needs the target (a prerequisite).", _CORE, _CORE
    ),
    RelationshipType.CONTRADICTS: RelationshipRule(
        "The source conflicts with the target.", _CORE | {_K.EVIDENCE}, _CORE | {_K.EVIDENCE}
    ),
    RelationshipType.RELATED_TO: RelationshipRule("A general association.", _ALL, _ALL),
}


def is_allowed(relationship_type: RelationshipType, source_type: KnowledgeType, target_type: KnowledgeType) -> bool:
    rule = RELATIONSHIP_RULES[RelationshipType(relationship_type)]
    return KnowledgeType(source_type) in rule.source_types and KnowledgeType(target_type) in rule.target_types


def check_relationship_types(
    relationship_type: RelationshipType, source_type: KnowledgeType, target_type: KnowledgeType
) -> None:
    """Raise ValueError when `source <relationship_type> target` is not a meaningful statement."""
    if not is_allowed(relationship_type, source_type, target_type):
        rule = RELATIONSHIP_RULES[RelationshipType(relationship_type)]
        raise ValueError(
            f"A {KnowledgeType(source_type).value} cannot be '{RelationshipType(relationship_type).value}' a "
            f"{KnowledgeType(target_type).value}: {rule.description}"
        )

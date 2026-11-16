"""Type-specific content of each knowledge category, as validated Pydantic models.

These carry only what is particular to a category (a concept's definition, a method's steps).
What every unit shares (identity, sources, pages, confidence, uncertainty, status) lives in the
envelope, app.knowledge.schemas.KnowledgeUnitCreate. The models are independent of the
database and of any LLM provider: whatever produces them, they are checked the same way.

`knowledge_type` is the discriminator: it identifies the category when content is read from a
dict or JSON. Lists are never None (an empty list means "none stated"); a text that the book
does not give is None, never an invented empty string.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.knowledge.types import ExpressionStatus, KnowledgeType, ReconstructionStatus

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=10_000)]
Item = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000)]
Expression = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000)]
Symbol = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]

# An upper bound keeps a runaway extraction from storing thousands of items.
Items = Annotated[list[Item], Field(default_factory=list, max_length=200)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    repeated: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen and value not in repeated:
            repeated.append(value)
        seen.add(key)
    return repeated


class KnowledgeContent(_Model):
    """Base of all content models. Subclasses set `knowledge_type` and the three projections."""

    @property
    def title(self) -> str:
        raise NotImplementedError

    @property
    def body(self) -> str:
        """The main text of the unit."""
        raise NotImplementedError

    @property
    def summary(self) -> str | None:
        """A short explanation, when the content has one."""
        return None


# -- Concept ---------------------------------------------------------------------------------------


class ConceptContent(KnowledgeContent):
    knowledge_type: Literal["concept"] = "concept"
    name: Name
    definition: Text
    explanation: Text | None = None
    characteristics: Items
    related_concepts: Items  # names as the book gives them; links between units are relationships
    examples: Items

    @property
    def title(self) -> str:
        return self.name

    @property
    def body(self) -> str:
        return self.definition

    @property
    def summary(self) -> str | None:
        return self.explanation


# -- Principle -------------------------------------------------------------------------------------


class PrincipleContent(KnowledgeContent):
    knowledge_type: Literal["principle"] = "principle"
    name: Name
    statement: Text
    explanation: Text | None = None
    conditions: Items  # when the principle holds
    limitations: Items
    supporting_evidence: Items  # short statements; full evidence is its own unit

    @property
    def title(self) -> str:
        return self.name

    @property
    def body(self) -> str:
        return self.statement

    @property
    def summary(self) -> str | None:
        return self.explanation


# -- Method ----------------------------------------------------------------------------------------


class MethodContent(KnowledgeContent):
    knowledge_type: Literal["method"] = "method"
    name: Name
    purpose: Text
    preconditions: Items
    steps: Annotated[list[Item], Field(min_length=1, max_length=200)]  # in order
    inputs: Items
    outputs: Items
    expected_results: Items
    limitations: Items

    @property
    def title(self) -> str:
        return self.name

    @property
    def body(self) -> str:
        return self.purpose


# -- Framework -------------------------------------------------------------------------------------


class FrameworkComponent(_Model):
    name: Name
    description: Text | None = None


class ComponentRelationship(_Model):
    """How two components of the same framework relate. Both names must be components."""

    source: Name
    target: Name
    relation: Name
    description: Text | None = None


class FrameworkContent(KnowledgeContent):
    knowledge_type: Literal["framework"] = "framework"
    name: Name
    purpose: Text
    components: Annotated[list[FrameworkComponent], Field(min_length=1, max_length=100)]
    component_relationships: Annotated[list[ComponentRelationship], Field(default_factory=list, max_length=500)]
    application: Text | None = None
    limitations: Items

    @model_validator(mode="after")
    def _components_are_consistent(self) -> FrameworkContent:
        names = [component.name for component in self.components]
        if repeated := _duplicates(names):
            raise ValueError(f"Component names must be unique; repeated: {', '.join(repeated)}")
        known = {name.casefold() for name in names}
        for link in self.component_relationships:
            for end in (link.source, link.target):
                if end.casefold() not in known:
                    raise ValueError(f"Relationship refers to '{end}', which is not a component of the framework")
            if link.source.casefold() == link.target.casefold():
                raise ValueError(f"Component '{link.source}' cannot be related to itself")
        return self

    @property
    def title(self) -> str:
        return self.name

    @property
    def body(self) -> str:
        return self.purpose

    @property
    def summary(self) -> str | None:
        return self.application


# -- Evidence --------------------------------------------------------------------------------------


class EvidenceContent(KnowledgeContent):
    knowledge_type: Literal["evidence"] = "evidence"
    claim: Text  # the claim this evidence supports
    description: Text
    findings: Items
    supporting_examples: Items
    context: Text | None = None  # study, setting, population, source of the evidence

    @property
    def title(self) -> str:
        return self.claim if len(self.claim) <= 300 else self.claim[:299].rstrip() + "…"

    @property
    def body(self) -> str:
        return self.description

    @property
    def summary(self) -> str | None:
        return self.context


# -- Example ---------------------------------------------------------------------------------------


class RelatedKnowledge(_Model):
    """What an example illustrates, by name (the link between units is a relationship)."""

    knowledge_type: Literal["concept", "principle", "method"]
    name: Name


class ExampleContent(KnowledgeContent):
    knowledge_type: Literal["example"] = "example"
    name: Name
    scenario: Text
    explanation: Text | None = None
    related_to: Annotated[list[RelatedKnowledge], Field(default_factory=list, max_length=20)]
    expected_outcome: Text | None = None

    @property
    def title(self) -> str:
        return self.name

    @property
    def body(self) -> str:
        return self.scenario

    @property
    def summary(self) -> str | None:
        return self.explanation


# -- Equation --------------------------------------------------------------------------------------


class EquationVariable(_Model):
    symbol: Symbol
    definition: Text | None = None  # None when the book does not define it (or it was not read)
    unit: Symbol | None = None


class EquationParameter(_Model):
    symbol: Symbol
    kind: Literal["parameter", "constant"]
    description: Text | None = None
    value: Symbol | None = None


class EquationContent(KnowledgeContent):
    """An equation, kept honestly: what was read is stored as read, and its state is explicit.

    `original_expression` is the text as extracted (possibly garbled); `latex` is a clean form
    when one is available. At least one of them is required. `extraction_status` has no default
    on purpose: whoever builds the content must say whether the expression is complete.
    """

    knowledge_type: Literal["equation"] = "equation"
    name: Name
    original_expression: Expression | None = None
    latex: Expression | None = None
    variables: Annotated[list[EquationVariable], Field(default_factory=list, max_length=100)]
    parameters: Annotated[list[EquationParameter], Field(default_factory=list, max_length=100)]
    assumptions: Items
    explanation: Text | None = None
    extraction_status: ExpressionStatus
    reconstruction_status: ReconstructionStatus = ReconstructionStatus.NOT_ATTEMPTED
    status_notes: Text | None = None  # what is missing or uncertain

    @model_validator(mode="after")
    def _status_is_honest(self) -> EquationContent:
        if self.original_expression is None and self.latex is None:
            raise ValueError("An equation needs its original expression or a LaTeX form (or both)")
        incomplete = self.extraction_status != ExpressionStatus.COMPLETE
        if incomplete and self.reconstruction_status == ReconstructionStatus.NOT_NEEDED:
            raise ValueError("An incomplete expression cannot be marked 'not_needed' for reconstruction")
        if self.reconstruction_status in (ReconstructionStatus.RECONSTRUCTED, ReconstructionStatus.PARTIAL) and (
            self.latex is None
        ):
            raise ValueError(f"Reconstruction status '{self.reconstruction_status}' requires a LaTeX form")
        symbols = [variable.symbol for variable in self.variables] + [p.symbol for p in self.parameters]
        if repeated := _duplicates(symbols):
            raise ValueError(f"Symbols must be defined once; repeated: {', '.join(repeated)}")
        return self

    @property
    def is_incomplete(self) -> bool:
        """True when the expression is not fully and cleanly available."""
        return self.extraction_status != ExpressionStatus.COMPLETE or self.reconstruction_status in (
            ReconstructionStatus.PARTIAL,
            ReconstructionStatus.FAILED,
        )

    @property
    def title(self) -> str:
        return self.name

    @property
    def body(self) -> str:
        return self.latex or self.original_expression or ""

    @property
    def summary(self) -> str | None:
        return self.explanation


AnyKnowledgeContent = Annotated[
    ConceptContent
    | PrincipleContent
    | MethodContent
    | FrameworkContent
    | EvidenceContent
    | ExampleContent
    | EquationContent,
    Field(discriminator="knowledge_type"),
]

CONTENT_MODELS: dict[KnowledgeType, type[KnowledgeContent]] = {
    KnowledgeType.CONCEPT: ConceptContent,
    KnowledgeType.PRINCIPLE: PrincipleContent,
    KnowledgeType.METHOD: MethodContent,
    KnowledgeType.FRAMEWORK: FrameworkContent,
    KnowledgeType.EVIDENCE: EvidenceContent,
    KnowledgeType.EXAMPLE: ExampleContent,
    KnowledgeType.EQUATION: EquationContent,
}

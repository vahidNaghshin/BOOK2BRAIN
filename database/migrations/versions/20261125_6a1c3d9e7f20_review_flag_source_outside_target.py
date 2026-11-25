"""review flag code source_outside_target

Revision ID: 6a1c3d9e7f20
Revises: 1eb6b9dd22a0
Create Date: 2026-11-25 14:30:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '6a1c3d9e7f20'
down_revision: Union[str, Sequence[str], None] = '1eb6b9dd22a0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_BEFORE = (
    "invalid_source_reference", "source_quote_mismatch", "page_inconsistent", "no_source", "incomplete_equation",
    "ambiguous_equation", "equation_needs_review", "low_confidence", "uncertain_extraction", "inferred_content",
    "unsupported_claim", "potential_duplicate", "repeated_validation_failure", "unresolved_extraction_error",
    "ai_suggested_relationship", "context_reduced",
)
_AFTER = (*_BEFORE, "source_outside_target")


def _in_list(values: tuple[str, ...]) -> str:
    return "code IN (" + ", ".join(f"'{v}'" for v in values) + ")"


def upgrade() -> None:
    """Upgrade schema."""
    # Autogenerate does not see check constraints: a flag can now say a unit cites a neighbouring passage.
    op.drop_constraint(op.f('ck_review_flags_code_valid'), 'review_flags', type_='check')
    op.create_check_constraint(op.f('ck_review_flags_code_valid'), 'review_flags', _in_list(_AFTER))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(op.f('ck_review_flags_code_valid'), 'review_flags', type_='check')
    op.execute("DELETE FROM review_flags WHERE code = 'source_outside_target'")
    op.create_check_constraint(op.f('ck_review_flags_code_valid'), 'review_flags', _in_list(_BEFORE))

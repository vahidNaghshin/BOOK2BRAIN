"""add equations and equation status

Revision ID: c0bd96cbff22
Revises: 25205d1bf12c
Create Date: 2026-11-08 15:51:50.127350

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'c0bd96cbff22'
down_revision: Union[str, Sequence[str], None] = '25205d1bf12c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the equations table and the equation status columns on processing runs."""
    op.create_table('equations',
    sa.Column('document_id', sa.Uuid(), nullable=False),
    sa.Column('extracted_block_id', sa.Uuid(), nullable=True),
    sa.Column('sequence_number', sa.Integer(), nullable=False),
    sa.Column('identity_hash', sa.String(length=64), nullable=False),
    sa.Column('latex', sa.Text(), nullable=False),
    sa.Column('original_latex', sa.Text(), nullable=True),
    sa.Column('plain_text', sa.Text(), nullable=True),
    sa.Column('equation_type', sa.String(length=20), nullable=False),
    sa.Column('equation_category', sa.String(length=100), nullable=True),
    sa.Column('equation_number', sa.String(length=50), nullable=True),
    sa.Column('page_number', sa.Integer(), nullable=True),
    sa.Column('section_title', sa.String(length=1000), nullable=True),
    sa.Column('surrounding_text', sa.Text(), nullable=True),
    sa.Column('source_location', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('bounding_box', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('crop_reference', sa.String(length=1024), nullable=True),
    sa.Column('extraction_method', sa.String(length=50), nullable=False),
    sa.Column('confidence', sa.Float(), nullable=True),
    sa.Column('validation_status', sa.String(length=20), nullable=False),
    sa.Column('validation_warnings', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.CheckConstraint("equation_type IN ('inline', 'display', 'numbered', 'unknown')", name=op.f('ck_equations_equation_type_valid')),
    sa.CheckConstraint("validation_status IN ('pending', 'valid', 'uncertain', 'invalid')", name=op.f('ck_equations_validation_status_valid')),
    sa.CheckConstraint('confidence IS NULL OR (confidence >= 0 AND confidence <= 1)', name=op.f('ck_equations_confidence_range')),
    sa.CheckConstraint('page_number IS NULL OR page_number >= 1', name=op.f('ck_equations_page_number_positive')),
    sa.CheckConstraint('sequence_number >= 0', name=op.f('ck_equations_sequence_number_non_negative')),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_equations_document_id_documents'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['extracted_block_id'], ['extracted_blocks.id'], name=op.f('fk_equations_extracted_block_id_extracted_blocks'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_equations')),
    sa.UniqueConstraint('document_id', 'identity_hash', name='uq_equations_document_identity'),
    sa.UniqueConstraint('document_id', 'sequence_number', name='uq_equations_document_sequence')
    )
    op.create_index('ix_equations_document_page', 'equations', ['document_id', 'page_number'], unique=False)
    op.create_index(op.f('ix_equations_equation_number'), 'equations', ['equation_number'], unique=False)
    op.create_index(op.f('ix_equations_extracted_block_id'), 'equations', ['extracted_block_id'], unique=False)
    op.create_index(op.f('ix_equations_validation_status'), 'equations', ['validation_status'], unique=False)
    op.add_column('processing_runs', sa.Column('equation_status', sa.String(length=30), server_default='not_started', nullable=False))
    op.add_column('processing_runs', sa.Column('equation_summary', postgresql.JSONB(astext_type=sa.Text()), nullable=True))

    # Alembic does not detect check constraints, so these are explicit.
    op.create_check_constraint(
        op.f('ck_processing_runs_equation_status_valid'),
        'processing_runs',
        "equation_status IN ('not_started', 'processing', 'completed', "
        "'completed_with_warnings', 'failed', 'unavailable')",
    )
    # Display equations become 'equation' blocks, so the block type constraint is widened.
    op.drop_constraint(op.f('ck_extracted_blocks_block_type_valid'), 'extracted_blocks', type_='check')
    op.create_check_constraint(
        op.f('ck_extracted_blocks_block_type_valid'),
        'extracted_blocks',
        "block_type IN ('paragraph', 'heading', 'table', 'list', 'caption', 'equation', 'other')",
    )


def downgrade() -> None:
    """Remove equations. Equation blocks are converted back to 'other' so the old constraint holds."""
    op.execute("UPDATE extracted_blocks SET block_type = 'other' WHERE block_type = 'equation'")
    op.drop_constraint(op.f('ck_extracted_blocks_block_type_valid'), 'extracted_blocks', type_='check')
    op.create_check_constraint(
        op.f('ck_extracted_blocks_block_type_valid'),
        'extracted_blocks',
        "block_type IN ('paragraph', 'heading', 'table', 'list', 'caption', 'other')",
    )
    op.drop_constraint(op.f('ck_processing_runs_equation_status_valid'), 'processing_runs', type_='check')
    op.drop_column('processing_runs', 'equation_summary')
    op.drop_column('processing_runs', 'equation_status')
    op.drop_index(op.f('ix_equations_validation_status'), table_name='equations')
    op.drop_index(op.f('ix_equations_extracted_block_id'), table_name='equations')
    op.drop_index(op.f('ix_equations_equation_number'), table_name='equations')
    op.drop_index('ix_equations_document_page', table_name='equations')
    op.drop_table('equations')

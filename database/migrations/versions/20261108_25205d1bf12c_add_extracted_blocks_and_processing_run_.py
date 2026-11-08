"""add extracted blocks and processing run progress

Revision ID: 25205d1bf12c
Revises: 5fb1e78179cf
Create Date: 2026-11-08 10:32:26.788533

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '25205d1bf12c'
down_revision: Union[str, Sequence[str], None] = '5fb1e78179cf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add extracted_blocks and the processing run stage, progress and summary columns."""
    op.create_table('extracted_blocks',
    sa.Column('document_id', sa.Uuid(), nullable=False),
    sa.Column('sequence_number', sa.Integer(), nullable=False),
    sa.Column('block_type', sa.String(length=20), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('raw_text', sa.Text(), nullable=True),
    sa.Column('page_number', sa.Integer(), nullable=True),
    sa.Column('section_title', sa.String(length=1000), nullable=True),
    sa.Column('source_location', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.CheckConstraint("block_type IN ('paragraph', 'heading', 'table', 'list', 'caption', 'other')", name=op.f('ck_extracted_blocks_block_type_valid')),
    sa.CheckConstraint('page_number IS NULL OR page_number >= 1', name=op.f('ck_extracted_blocks_page_number_positive')),
    sa.CheckConstraint('sequence_number >= 0', name=op.f('ck_extracted_blocks_sequence_number_non_negative')),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_extracted_blocks_document_id_documents'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_extracted_blocks')),
    sa.UniqueConstraint('document_id', 'sequence_number', name='uq_extracted_blocks_document_sequence')
    )
    op.create_index('ix_extracted_blocks_document_page', 'extracted_blocks', ['document_id', 'page_number'], unique=False)
    op.add_column('processing_runs', sa.Column('stage', sa.String(length=20), nullable=True))
    op.add_column('processing_runs', sa.Column('progress_current', sa.Integer(), nullable=True))
    op.add_column('processing_runs', sa.Column('progress_total', sa.Integer(), nullable=True))
    op.add_column('processing_runs', sa.Column('result_summary', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    # Alembic does not detect check constraints, so this one is added explicitly.
    op.create_check_constraint(
        op.f('ck_processing_runs_stage_valid'),
        'processing_runs',
        "stage IN ('parsing', 'ocr', 'normalizing')",
    )


def downgrade() -> None:
    """Remove extracted blocks and the processing run progress columns."""
    op.drop_constraint(op.f('ck_processing_runs_stage_valid'), 'processing_runs', type_='check')
    op.drop_column('processing_runs', 'result_summary')
    op.drop_column('processing_runs', 'progress_total')
    op.drop_column('processing_runs', 'progress_current')
    op.drop_column('processing_runs', 'stage')
    op.drop_index('ix_extracted_blocks_document_page', table_name='extracted_blocks')
    op.drop_table('extracted_blocks')

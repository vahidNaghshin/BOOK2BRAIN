"""pipeline stages, retry bookkeeping, book publisher and year

Revision ID: e33c9630cffb
Revises: 4b5cbd1ad745
Create Date: 2026-11-13 11:10:36.144615

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'e33c9630cffb'
down_revision: Union[str, Sequence[str], None] = '4b5cbd1ad745'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD_STAGES = "'parsing', 'ocr', 'normalizing'"
NEW_STAGES = (
    "'document_registration', 'parsing', 'ocr', 'normalizing', 'text_extraction', "
    "'equation_extraction', 'structural_detection', 'paragraph_identification', "
    "'passage_segmentation', 'persistence', 'validation', 'completed'"
)



def upgrade() -> None:
    """Add the processing pipeline's stages and retry bookkeeping, and book publisher/year."""
    op.add_column('books', sa.Column('publisher', sa.String(length=500), nullable=True))
    op.add_column('books', sa.Column('publication_year', sa.Integer(), nullable=True))
    op.create_check_constraint(
        op.f('ck_books_publication_year_range'),
        'books',
        'publication_year IS NULL OR (publication_year >= 1000 AND publication_year <= 2200)',
    )

    op.add_column('processing_errors', sa.Column('stage', sa.String(length=30), nullable=True))
    op.add_column('processing_errors', sa.Column('retry_attempt', sa.Integer(), server_default='0', nullable=False))
    # Errors recorded before this migration kept their stage in the details.
    op.execute("UPDATE processing_errors SET stage = details->>'stage' WHERE details ? 'stage'")

    op.add_column('processing_runs', sa.Column('retry_count', sa.Integer(), server_default='0', nullable=False))
    op.add_column('processing_runs', sa.Column('text_completed_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('processing_runs', sa.Column('stage_results', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('processing_runs', sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False))
    op.alter_column('processing_runs', 'stage', existing_type=sa.VARCHAR(length=20), type_=sa.String(length=30), existing_nullable=True)
    # Runs that finished before this migration had their text saved when they completed.
    op.execute("UPDATE processing_runs SET text_completed_at = completed_at WHERE status = 'completed'")
    op.create_check_constraint(op.f('ck_processing_runs_retry_count_non_negative'), 'processing_runs', 'retry_count >= 0')
    op.create_index('ix_processing_runs_document_created', 'processing_runs', ['document_id', 'created_at'], unique=False)

    op.drop_constraint(op.f('ck_processing_runs_stage_valid'), 'processing_runs', type_='check')
    op.create_check_constraint(op.f('ck_processing_runs_stage_valid'), 'processing_runs', "stage IN (" + NEW_STAGES + ")")


def downgrade() -> None:
    """Remove the pipeline bookkeeping. Runs in a newer stage are moved back to 'normalizing'."""
    op.execute(f"UPDATE processing_runs SET stage = 'normalizing' WHERE stage NOT IN ({OLD_STAGES})")
    op.drop_constraint(op.f('ck_processing_runs_stage_valid'), 'processing_runs', type_='check')
    op.create_check_constraint(op.f('ck_processing_runs_stage_valid'), 'processing_runs', "stage IN (" + OLD_STAGES + ")")
    op.drop_index('ix_processing_runs_document_created', table_name='processing_runs')
    op.drop_constraint(op.f('ck_processing_runs_retry_count_non_negative'), 'processing_runs', type_='check')
    op.alter_column('processing_runs', 'stage', existing_type=sa.String(length=30), type_=sa.VARCHAR(length=20), existing_nullable=True)
    op.drop_column('processing_runs', 'updated_at')
    op.drop_column('processing_runs', 'stage_results')
    op.drop_column('processing_runs', 'text_completed_at')
    op.drop_column('processing_runs', 'retry_count')
    op.drop_column('processing_errors', 'retry_attempt')
    op.drop_column('processing_errors', 'stage')
    op.drop_constraint(op.f('ck_books_publication_year_range'), 'books', type_='check')
    op.drop_column('books', 'publication_year')
    op.drop_column('books', 'publisher')

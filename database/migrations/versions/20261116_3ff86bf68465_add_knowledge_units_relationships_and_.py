"""add knowledge units, relationships and extraction runs

Revision ID: 3ff86bf68465
Revises: e33c9630cffb
Create Date: 2026-11-16 10:47:18.382356

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '3ff86bf68465'
down_revision: Union[str, Sequence[str], None] = 'e33c9630cffb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add extraction runs, knowledge units, their passage links and the relationships between units."""
    op.create_table('extraction_runs',
    sa.Column('book_id', sa.Uuid(), nullable=False),
    sa.Column('document_id', sa.Uuid(), nullable=True),
    sa.Column('structural_node_id', sa.Uuid(), nullable=True),
    sa.Column('status', sa.String(length=20), server_default='pending', nullable=False),
    sa.Column('passages_processed', sa.Integer(), server_default='0', nullable=False),
    sa.Column('units_extracted', sa.Integer(), server_default='0', nullable=False),
    sa.Column('failure_count', sa.Integer(), server_default='0', nullable=False),
    sa.Column('model_name', sa.String(length=200), nullable=True),
    sa.Column('prompt_version', sa.String(length=100), nullable=True),
    sa.Column('parameters', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('error_code', sa.String(length=100), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('error_details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')", name=op.f('ck_extraction_runs_status_valid')),
    sa.CheckConstraint('completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at', name=op.f('ck_extraction_runs_completion_order')),
    sa.CheckConstraint('passages_processed >= 0 AND units_extracted >= 0 AND failure_count >= 0', name=op.f('ck_extraction_runs_counts_non_negative')),
    sa.ForeignKeyConstraint(['book_id'], ['books.id'], name=op.f('fk_extraction_runs_book_id_books'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_extraction_runs_document_id_documents'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['structural_node_id'], ['structural_nodes.id'], name=op.f('fk_extraction_runs_structural_node_id_structural_nodes'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_extraction_runs'))
    )
    op.create_index('ix_extraction_runs_book_status', 'extraction_runs', ['book_id', 'status'], unique=False)
    op.create_index(op.f('ix_extraction_runs_document_id'), 'extraction_runs', ['document_id'], unique=False)
    op.create_table('knowledge_units',
    sa.Column('book_id', sa.Uuid(), nullable=False),
    sa.Column('document_id', sa.Uuid(), nullable=False),
    sa.Column('extraction_run_id', sa.Uuid(), nullable=True),
    sa.Column('chapter_node_id', sa.Uuid(), nullable=True),
    sa.Column('section_node_id', sa.Uuid(), nullable=True),
    sa.Column('equation_id', sa.Uuid(), nullable=True),
    sa.Column('knowledge_type', sa.String(length=20), nullable=False),
    sa.Column('title', sa.String(length=500), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('summary', sa.Text(), nullable=True),
    sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('schema_version', sa.Integer(), server_default='1', nullable=False),
    sa.Column('page_start', sa.Integer(), nullable=True),
    sa.Column('page_end', sa.Integer(), nullable=True),
    sa.Column('page_status', sa.String(length=10), server_default='unknown', nullable=False),
    sa.Column('confidence', sa.Float(), nullable=True),
    sa.Column('is_uncertain', sa.Boolean(), server_default='false', nullable=False),
    sa.Column('uncertainty_notes', sa.Text(), nullable=True),
    sa.Column('validation_status', sa.String(length=20), server_default='pending', nullable=False),
    sa.Column('review_status', sa.String(length=20), server_default='unreviewed', nullable=False),
    sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("equation_id IS NULL OR knowledge_type = 'equation'", name=op.f('ck_knowledge_units_equation_link_type')),
    sa.CheckConstraint("jsonb_typeof(details) = 'object'", name=op.f('ck_knowledge_units_details_is_object')),
    sa.CheckConstraint("knowledge_type IN ('concept', 'principle', 'method', 'framework', 'evidence', 'example', 'equation')", name=op.f('ck_knowledge_units_knowledge_type_valid')),
    sa.CheckConstraint("page_status IN ('exact', 'partial', 'unknown')", name=op.f('ck_knowledge_units_page_status_valid')),
    sa.CheckConstraint("review_status IN ('unreviewed', 'approved', 'rejected', 'needs_changes')", name=op.f('ck_knowledge_units_review_status_valid')),
    sa.CheckConstraint("validation_status IN ('pending', 'valid', 'uncertain', 'invalid')", name=op.f('ck_knowledge_units_validation_status_valid')),
    sa.CheckConstraint('confidence IS NULL OR (confidence >= 0 AND confidence <= 1)', name=op.f('ck_knowledge_units_confidence_range')),
    sa.CheckConstraint('length(btrim(content)) > 0', name=op.f('ck_knowledge_units_content_not_blank')),
    sa.CheckConstraint('length(btrim(title)) > 0', name=op.f('ck_knowledge_units_title_not_blank')),
    sa.CheckConstraint('page_end IS NULL OR page_start IS NULL OR page_end >= page_start', name=op.f('ck_knowledge_units_page_order')),
    sa.CheckConstraint('page_start IS NULL OR page_start >= 1', name=op.f('ck_knowledge_units_page_start_positive')),
    sa.CheckConstraint('schema_version >= 1', name=op.f('ck_knowledge_units_schema_version_positive')),
    sa.CheckConstraint('uncertainty_notes IS NULL OR is_uncertain', name=op.f('ck_knowledge_units_uncertainty_notes_need_flag')),
    sa.ForeignKeyConstraint(['book_id'], ['books.id'], name=op.f('fk_knowledge_units_book_id_books'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['chapter_node_id'], ['structural_nodes.id'], name=op.f('fk_knowledge_units_chapter_node_id_structural_nodes'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_knowledge_units_document_id_documents'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['equation_id'], ['equations.id'], name=op.f('fk_knowledge_units_equation_id_equations'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['extraction_run_id'], ['extraction_runs.id'], name=op.f('fk_knowledge_units_extraction_run_id_extraction_runs'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['section_node_id'], ['structural_nodes.id'], name=op.f('fk_knowledge_units_section_node_id_structural_nodes'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_knowledge_units')),
    sa.UniqueConstraint('id', 'book_id', name='uq_knowledge_units_id_book_id')
    )
    op.create_index('ix_knowledge_units_book_review', 'knowledge_units', ['book_id', 'review_status'], unique=False)
    op.create_index('ix_knowledge_units_book_type', 'knowledge_units', ['book_id', 'knowledge_type'], unique=False)
    op.create_index(op.f('ix_knowledge_units_chapter_node_id'), 'knowledge_units', ['chapter_node_id'], unique=False)
    op.create_index(op.f('ix_knowledge_units_document_id'), 'knowledge_units', ['document_id'], unique=False)
    op.create_index(op.f('ix_knowledge_units_equation_id'), 'knowledge_units', ['equation_id'], unique=False)
    op.create_index(op.f('ix_knowledge_units_extraction_run_id'), 'knowledge_units', ['extraction_run_id'], unique=False)
    op.create_index(op.f('ix_knowledge_units_section_node_id'), 'knowledge_units', ['section_node_id'], unique=False)
    op.create_table('knowledge_relationships',
    sa.Column('book_id', sa.Uuid(), nullable=False),
    sa.Column('source_unit_id', sa.Uuid(), nullable=False),
    sa.Column('target_unit_id', sa.Uuid(), nullable=False),
    sa.Column('relationship_type', sa.String(length=20), nullable=False),
    sa.Column('explanation', sa.Text(), nullable=True),
    sa.Column('confidence', sa.Float(), nullable=True),
    sa.Column('extraction_run_id', sa.Uuid(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.CheckConstraint("relationship_type IN ('supports', 'illustrates', 'implements', 'describes', 'part_of', 'depends_on', 'contradicts', 'related_to')", name=op.f('ck_knowledge_relationships_relationship_type_valid')),
    sa.CheckConstraint('confidence IS NULL OR (confidence >= 0 AND confidence <= 1)', name=op.f('ck_knowledge_relationships_confidence_range')),
    sa.CheckConstraint('source_unit_id <> target_unit_id', name=op.f('ck_knowledge_relationships_no_self_relationship')),
    sa.ForeignKeyConstraint(['extraction_run_id'], ['extraction_runs.id'], name=op.f('fk_knowledge_relationships_extraction_run_id_extraction_runs'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['source_unit_id', 'book_id'], ['knowledge_units.id', 'knowledge_units.book_id'], name=op.f('fk_knowledge_relationships_source_unit_id_knowledge_units'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['target_unit_id', 'book_id'], ['knowledge_units.id', 'knowledge_units.book_id'], name=op.f('fk_knowledge_relationships_target_unit_id_knowledge_units'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_knowledge_relationships')),
    sa.UniqueConstraint('source_unit_id', 'target_unit_id', 'relationship_type', name='uq_knowledge_relationships_edge')
    )
    op.create_index(op.f('ix_knowledge_relationships_book_id'), 'knowledge_relationships', ['book_id'], unique=False)
    op.create_index(op.f('ix_knowledge_relationships_extraction_run_id'), 'knowledge_relationships', ['extraction_run_id'], unique=False)
    op.create_index(op.f('ix_knowledge_relationships_target_unit_id'), 'knowledge_relationships', ['target_unit_id'], unique=False)
    op.create_table('knowledge_unit_passages',
    sa.Column('knowledge_unit_id', sa.Uuid(), nullable=False),
    sa.Column('passage_id', sa.Uuid(), nullable=False),
    sa.Column('position', sa.Integer(), nullable=False),
    sa.Column('role', sa.String(length=10), server_default='supporting', nullable=False),
    sa.Column('quote', sa.Text(), nullable=True),
    sa.CheckConstraint("role IN ('primary', 'supporting')", name=op.f('ck_knowledge_unit_passages_role_valid')),
    sa.CheckConstraint('position >= 0', name=op.f('ck_knowledge_unit_passages_position_non_negative')),
    sa.ForeignKeyConstraint(['knowledge_unit_id'], ['knowledge_units.id'], name=op.f('fk_knowledge_unit_passages_knowledge_unit_id_knowledge_units'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['passage_id'], ['passages.id'], name=op.f('fk_knowledge_unit_passages_passage_id_passages'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('knowledge_unit_id', 'passage_id', name=op.f('pk_knowledge_unit_passages')),
    sa.UniqueConstraint('knowledge_unit_id', 'position', name='uq_knowledge_unit_passages_position')
    )
    op.create_index(op.f('ix_knowledge_unit_passages_passage_id'), 'knowledge_unit_passages', ['passage_id'], unique=False)
    op.create_index('uq_knowledge_unit_passages_primary', 'knowledge_unit_passages', ['knowledge_unit_id'], unique=True, postgresql_where=sa.text("role = 'primary'"))
    op.create_table('knowledge_relationship_passages',
    sa.Column('relationship_id', sa.Uuid(), nullable=False),
    sa.Column('passage_id', sa.Uuid(), nullable=False),
    sa.Column('position', sa.Integer(), nullable=False),
    sa.CheckConstraint('position >= 0', name=op.f('ck_knowledge_relationship_passages_position_non_negative')),
    sa.ForeignKeyConstraint(['passage_id'], ['passages.id'], name=op.f('fk_knowledge_relationship_passages_passage_id_passages'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['relationship_id'], ['knowledge_relationships.id'], name=op.f('fk_knowledge_relationship_passages_relationship_id_knowledge_relationships'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('relationship_id', 'passage_id', name=op.f('pk_knowledge_relationship_passages')),
    sa.UniqueConstraint('relationship_id', 'position', name='uq_knowledge_relationship_passages_position')
    )
    op.create_index(op.f('ix_knowledge_relationship_passages_passage_id'), 'knowledge_relationship_passages', ['passage_id'], unique=False)


def downgrade() -> None:
    """Remove the Phase 2 knowledge tables (all extracted knowledge is dropped; Phase 1 data is untouched)."""
    op.drop_index(op.f('ix_knowledge_relationship_passages_passage_id'), table_name='knowledge_relationship_passages')
    op.drop_table('knowledge_relationship_passages')
    op.drop_index('uq_knowledge_unit_passages_primary', table_name='knowledge_unit_passages', postgresql_where=sa.text("role = 'primary'"))
    op.drop_index(op.f('ix_knowledge_unit_passages_passage_id'), table_name='knowledge_unit_passages')
    op.drop_table('knowledge_unit_passages')
    op.drop_index(op.f('ix_knowledge_relationships_target_unit_id'), table_name='knowledge_relationships')
    op.drop_index(op.f('ix_knowledge_relationships_extraction_run_id'), table_name='knowledge_relationships')
    op.drop_index(op.f('ix_knowledge_relationships_book_id'), table_name='knowledge_relationships')
    op.drop_table('knowledge_relationships')
    op.drop_index(op.f('ix_knowledge_units_section_node_id'), table_name='knowledge_units')
    op.drop_index(op.f('ix_knowledge_units_extraction_run_id'), table_name='knowledge_units')
    op.drop_index(op.f('ix_knowledge_units_equation_id'), table_name='knowledge_units')
    op.drop_index(op.f('ix_knowledge_units_document_id'), table_name='knowledge_units')
    op.drop_index(op.f('ix_knowledge_units_chapter_node_id'), table_name='knowledge_units')
    op.drop_index('ix_knowledge_units_book_type', table_name='knowledge_units')
    op.drop_index('ix_knowledge_units_book_review', table_name='knowledge_units')
    op.drop_table('knowledge_units')
    op.drop_index(op.f('ix_extraction_runs_document_id'), table_name='extraction_runs')
    op.drop_index('ix_extraction_runs_book_status', table_name='extraction_runs')
    op.drop_table('extraction_runs')

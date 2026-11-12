"""add structural nodes paragraphs and passages

Revision ID: 4b5cbd1ad745
Revises: c0bd96cbff22
Create Date: 2026-11-12 10:13:37.034048

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '4b5cbd1ad745'
down_revision: Union[str, Sequence[str], None] = 'c0bd96cbff22'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add structural nodes, paragraphs, passages and the per-document structure analysis."""
    op.create_table('structural_nodes',
    sa.Column('book_id', sa.Uuid(), nullable=False),
    sa.Column('document_id', sa.Uuid(), nullable=False),
    sa.Column('parent_id', sa.Uuid(), nullable=True),
    sa.Column('node_type', sa.String(length=40), nullable=False),
    sa.Column('title', sa.String(length=1000), nullable=False),
    sa.Column('original_title', sa.String(length=1000), nullable=True),
    sa.Column('number', sa.String(length=50), nullable=True),
    sa.Column('level', sa.Integer(), nullable=False),
    sa.Column('sequence', sa.Integer(), nullable=False),
    sa.Column('page_start', sa.Integer(), nullable=True),
    sa.Column('page_end', sa.Integer(), nullable=True),
    sa.Column('start_offset', sa.Integer(), nullable=True),
    sa.Column('end_offset', sa.Integer(), nullable=True),
    sa.Column('confidence', sa.Float(), nullable=True),
    sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('confidence IS NULL OR (confidence >= 0 AND confidence <= 1)', name=op.f('ck_structural_nodes_confidence_range')),
    sa.CheckConstraint('level >= 0', name=op.f('ck_structural_nodes_level_non_negative')),
    sa.CheckConstraint('page_end IS NULL OR page_start IS NULL OR page_end >= page_start', name=op.f('ck_structural_nodes_page_order')),
    sa.CheckConstraint('page_start IS NULL OR page_start >= 1', name=op.f('ck_structural_nodes_page_start_positive')),
    sa.CheckConstraint('sequence >= 0', name=op.f('ck_structural_nodes_sequence_non_negative')),
    sa.ForeignKeyConstraint(['book_id'], ['books.id'], name=op.f('fk_structural_nodes_book_id_books'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_structural_nodes_document_id_documents'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['parent_id'], ['structural_nodes.id'], name=op.f('fk_structural_nodes_parent_id_structural_nodes'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_structural_nodes')),
    sa.UniqueConstraint('document_id', 'sequence', name='uq_structural_nodes_document_sequence')
    )
    op.create_index(op.f('ix_structural_nodes_book_id'), 'structural_nodes', ['book_id'], unique=False)
    op.create_index('ix_structural_nodes_document_parent', 'structural_nodes', ['document_id', 'parent_id'], unique=False)
    op.create_index('ix_structural_nodes_document_type', 'structural_nodes', ['document_id', 'node_type'], unique=False)
    op.create_index(op.f('ix_structural_nodes_parent_id'), 'structural_nodes', ['parent_id'], unique=False)
    op.create_table('structure_analyses',
    sa.Column('document_id', sa.Uuid(), nullable=False),
    sa.Column('structure_status', sa.String(length=20), nullable=True),
    sa.Column('structure_error', sa.Text(), nullable=True),
    sa.Column('structure_detected_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('structure_summary', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('structure_issues', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('passage_status', sa.String(length=20), nullable=True),
    sa.Column('passage_error', sa.Text(), nullable=True),
    sa.Column('passage_generated_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('passage_config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('passage_summary', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('passage_issues', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.CheckConstraint("passage_status IS NULL OR passage_status IN ('completed', 'failed')", name=op.f('ck_structure_analyses_passage_status_valid')),
    sa.CheckConstraint("structure_status IS NULL OR structure_status IN ('completed', 'failed')", name=op.f('ck_structure_analyses_structure_status_valid')),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_structure_analyses_document_id_documents'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_structure_analyses')),
    sa.UniqueConstraint('document_id', name=op.f('uq_structure_analyses_document_id'))
    )
    op.create_table('paragraphs',
    sa.Column('book_id', sa.Uuid(), nullable=False),
    sa.Column('document_id', sa.Uuid(), nullable=False),
    sa.Column('structural_node_id', sa.Uuid(), nullable=False),
    sa.Column('sequence', sa.Integer(), nullable=False),
    sa.Column('document_sequence', sa.Integer(), nullable=False),
    sa.Column('kind', sa.String(length=20), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('page_start', sa.Integer(), nullable=True),
    sa.Column('page_end', sa.Integer(), nullable=True),
    sa.Column('page_status', sa.String(length=10), nullable=False),
    sa.Column('start_offset', sa.Integer(), nullable=True),
    sa.Column('end_offset', sa.Integer(), nullable=True),
    sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("page_status IN ('exact', 'partial', 'unknown')", name=op.f('ck_paragraphs_page_status_valid')),
    sa.CheckConstraint('page_end IS NULL OR page_start IS NULL OR page_end >= page_start', name=op.f('ck_paragraphs_page_order')),
    sa.CheckConstraint('page_start IS NULL OR page_start >= 1', name=op.f('ck_paragraphs_page_start_positive')),
    sa.ForeignKeyConstraint(['book_id'], ['books.id'], name=op.f('fk_paragraphs_book_id_books'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_paragraphs_document_id_documents'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['structural_node_id'], ['structural_nodes.id'], name=op.f('fk_paragraphs_structural_node_id_structural_nodes'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_paragraphs')),
    sa.UniqueConstraint('document_id', 'document_sequence', name='uq_paragraphs_document_sequence'),
    sa.UniqueConstraint('structural_node_id', 'sequence', name='uq_paragraphs_node_sequence')
    )
    op.create_index(op.f('ix_paragraphs_book_id'), 'paragraphs', ['book_id'], unique=False)
    op.create_table('passages',
    sa.Column('book_id', sa.Uuid(), nullable=False),
    sa.Column('document_id', sa.Uuid(), nullable=False),
    sa.Column('structural_node_id', sa.Uuid(), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('sequence', sa.Integer(), nullable=False),
    sa.Column('document_sequence', sa.Integer(), nullable=False),
    sa.Column('page_start', sa.Integer(), nullable=True),
    sa.Column('page_end', sa.Integer(), nullable=True),
    sa.Column('page_status', sa.String(length=10), nullable=False),
    sa.Column('token_count', sa.Integer(), nullable=False),
    sa.Column('start_offset', sa.Integer(), nullable=True),
    sa.Column('end_offset', sa.Integer(), nullable=True),
    sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("page_status IN ('exact', 'partial', 'unknown')", name=op.f('ck_passages_page_status_valid')),
    sa.CheckConstraint('page_end IS NULL OR page_start IS NULL OR page_end >= page_start', name=op.f('ck_passages_page_order')),
    sa.CheckConstraint('page_start IS NULL OR page_start >= 1', name=op.f('ck_passages_page_start_positive')),
    sa.CheckConstraint('token_count >= 0', name=op.f('ck_passages_token_count_non_negative')),
    sa.ForeignKeyConstraint(['book_id'], ['books.id'], name=op.f('fk_passages_book_id_books'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_passages_document_id_documents'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['structural_node_id'], ['structural_nodes.id'], name=op.f('fk_passages_structural_node_id_structural_nodes'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_passages')),
    sa.UniqueConstraint('document_id', 'document_sequence', name='uq_passages_document_sequence'),
    sa.UniqueConstraint('structural_node_id', 'sequence', name='uq_passages_node_sequence')
    )
    op.create_index(op.f('ix_passages_book_id'), 'passages', ['book_id'], unique=False)
    op.create_table('passage_paragraphs',
    sa.Column('passage_id', sa.Uuid(), nullable=False),
    sa.Column('position', sa.Integer(), nullable=False),
    sa.Column('paragraph_id', sa.Uuid(), nullable=False),
    sa.Column('is_overlap', sa.Boolean(), server_default='false', nullable=False),
    sa.Column('text_start', sa.Integer(), nullable=True),
    sa.Column('text_end', sa.Integer(), nullable=True),
    sa.CheckConstraint('position >= 0', name=op.f('ck_passage_paragraphs_position_non_negative')),
    sa.CheckConstraint('text_start IS NULL OR text_end IS NULL OR text_end > text_start', name=op.f('ck_passage_paragraphs_text_range_valid')),
    sa.ForeignKeyConstraint(['paragraph_id'], ['paragraphs.id'], name=op.f('fk_passage_paragraphs_paragraph_id_paragraphs'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['passage_id'], ['passages.id'], name=op.f('fk_passage_paragraphs_passage_id_passages'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('passage_id', 'position', name=op.f('pk_passage_paragraphs'))
    )
    op.create_index(op.f('ix_passage_paragraphs_paragraph_id'), 'passage_paragraphs', ['paragraph_id'], unique=False)


def downgrade() -> None:
    """Drop the structure, paragraph and passage tables."""
    op.drop_index(op.f('ix_passage_paragraphs_paragraph_id'), table_name='passage_paragraphs')
    op.drop_table('passage_paragraphs')
    op.drop_index(op.f('ix_passages_book_id'), table_name='passages')
    op.drop_table('passages')
    op.drop_index(op.f('ix_paragraphs_book_id'), table_name='paragraphs')
    op.drop_table('paragraphs')
    op.drop_table('structure_analyses')
    op.drop_index(op.f('ix_structural_nodes_parent_id'), table_name='structural_nodes')
    op.drop_index('ix_structural_nodes_document_type', table_name='structural_nodes')
    op.drop_index('ix_structural_nodes_document_parent', table_name='structural_nodes')
    op.drop_index(op.f('ix_structural_nodes_book_id'), table_name='structural_nodes')
    op.drop_table('structural_nodes')

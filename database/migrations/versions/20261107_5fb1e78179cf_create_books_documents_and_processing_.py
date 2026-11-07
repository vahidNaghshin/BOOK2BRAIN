"""create books documents and processing tables

Revision ID: 5fb1e78179cf
Revises: 
Create Date: 2026-11-07 10:45:19.135359

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '5fb1e78179cf'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the initial schema."""
    # Embeddings are not implemented yet; the extension is enabled for later phases.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table('books',
    sa.Column('title', sa.String(length=500), nullable=False),
    sa.Column('subtitle', sa.String(length=500), nullable=True),
    sa.Column('author', sa.String(length=500), nullable=True),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('language', sa.String(length=64), nullable=True),
    sa.Column('isbn', sa.String(length=20), nullable=True),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_books'))
    )
    op.create_index(op.f('ix_books_isbn'), 'books', ['isbn'], unique=False)
    op.create_index('uq_books_title_author', 'books', [sa.literal_column('lower(title)'), sa.literal_column("lower(coalesce(author, ''))")], unique=True)
    op.create_table('documents',
    sa.Column('book_id', sa.Uuid(), nullable=False),
    sa.Column('original_filename', sa.String(length=255), nullable=False),
    sa.Column('stored_filename', sa.String(length=255), nullable=False),
    sa.Column('file_path', sa.String(length=1024), nullable=False),
    sa.Column('file_format', sa.String(length=16), nullable=False),
    sa.Column('mime_type', sa.String(length=255), nullable=False),
    sa.Column('file_size', sa.BigInteger(), nullable=False),
    sa.Column('sha256_hash', sa.String(length=64), nullable=False),
    sa.Column('processing_status', sa.String(length=20), server_default='uploaded', nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("processing_status IN ('uploaded', 'pending', 'processing', 'completed', 'failed')", name=op.f('ck_documents_processing_status_valid')),
    sa.CheckConstraint('file_size > 0', name=op.f('ck_documents_file_size_positive')),
    sa.ForeignKeyConstraint(['book_id'], ['books.id'], name=op.f('fk_documents_book_id_books'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_documents')),
    sa.UniqueConstraint('file_path', name=op.f('uq_documents_file_path'))
    )
    op.create_index(op.f('ix_documents_book_id'), 'documents', ['book_id'], unique=False)
    op.create_index(op.f('ix_documents_processing_status'), 'documents', ['processing_status'], unique=False)
    op.create_index(op.f('ix_documents_sha256_hash'), 'documents', ['sha256_hash'], unique=True)
    op.create_table('processing_runs',
    sa.Column('document_id', sa.Uuid(), nullable=False),
    sa.Column('status', sa.String(length=20), server_default='pending', nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.CheckConstraint("status IN ('pending', 'processing', 'completed', 'failed')", name=op.f('ck_processing_runs_status_valid')),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_processing_runs_document_id_documents'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_processing_runs'))
    )
    op.create_index(op.f('ix_processing_runs_document_id'), 'processing_runs', ['document_id'], unique=False)
    op.create_index(op.f('ix_processing_runs_status'), 'processing_runs', ['status'], unique=False)
    op.create_table('processing_errors',
    sa.Column('document_id', sa.Uuid(), nullable=False),
    sa.Column('processing_run_id', sa.Uuid(), nullable=True),
    sa.Column('error_type', sa.String(length=255), nullable=False),
    sa.Column('error_message', sa.Text(), nullable=False),
    sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_processing_errors_document_id_documents'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['processing_run_id'], ['processing_runs.id'], name=op.f('fk_processing_errors_processing_run_id_processing_runs'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_processing_errors'))
    )
    op.create_index(op.f('ix_processing_errors_document_id'), 'processing_errors', ['document_id'], unique=False)
    op.create_index(op.f('ix_processing_errors_processing_run_id'), 'processing_errors', ['processing_run_id'], unique=False)


def downgrade() -> None:
    """Drop the initial schema. The vector extension is kept: other objects may use it."""
    op.drop_index(op.f('ix_processing_errors_processing_run_id'), table_name='processing_errors')
    op.drop_index(op.f('ix_processing_errors_document_id'), table_name='processing_errors')
    op.drop_table('processing_errors')
    op.drop_index(op.f('ix_processing_runs_status'), table_name='processing_runs')
    op.drop_index(op.f('ix_processing_runs_document_id'), table_name='processing_runs')
    op.drop_table('processing_runs')
    op.drop_index(op.f('ix_documents_sha256_hash'), table_name='documents')
    op.drop_index(op.f('ix_documents_processing_status'), table_name='documents')
    op.drop_index(op.f('ix_documents_book_id'), table_name='documents')
    op.drop_table('documents')
    op.drop_index('uq_books_title_author', table_name='books')
    op.drop_index(op.f('ix_books_isbn'), table_name='books')
    op.drop_table('books')

"""Add cached_suggestions table for caching AI-generated prompts and competitors.

Revision ID: 002
Revises: 001
Create Date: 2026-01-26

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create cached_suggestions table
    op.create_table(
        'cached_suggestions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('brand', sa.String(length=255), nullable=False),
        sa.Column('search_type', sa.String(length=50), nullable=False, server_default='brand'),
        sa.Column('industry', sa.String(length=255), nullable=True),
        sa.Column('prompts', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('competitors', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    # Index for fast lookups by brand + search_type + industry
    op.create_index('ix_cached_suggestions_lookup', 'cached_suggestions', ['brand', 'search_type', 'industry'], unique=False)
    # Index for cleaning up expired entries
    op.create_index('ix_cached_suggestions_expires_at', 'cached_suggestions', ['expires_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_cached_suggestions_expires_at', table_name='cached_suggestions')
    op.drop_index('ix_cached_suggestions_lookup', table_name='cached_suggestions')
    op.drop_table('cached_suggestions')

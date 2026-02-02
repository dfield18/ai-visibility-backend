"""Add cached_categories table for domain classification caching.

Revision ID: 005
Revises: 004
Create Date: 2026-02-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '005'
down_revision: Union[str, None] = '004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create cached_categories table
    op.create_table(
        'cached_categories',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('domain', sa.String(length=255), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Create unique index on domain for fast lookups
    op.create_index('ix_cached_categories_domain', 'cached_categories', ['domain'], unique=True)


def downgrade() -> None:
    op.drop_index('ix_cached_categories_domain', table_name='cached_categories')
    op.drop_table('cached_categories')

"""Add scheduled_reports table.

Revision ID: 006
Revises: 005
Create Date: 2026-02-02

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '006'
down_revision: Union[str, None] = '005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create scheduled_reports table
    op.create_table(
        'scheduled_reports',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('brand', sa.String(length=255), nullable=False),
        sa.Column('search_type', sa.String(length=20), nullable=False, server_default='brand'),
        sa.Column('prompts', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('competitors', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('providers', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('temperatures', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('repeats', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('frequency', sa.String(length=20), nullable=False, server_default='weekly'),
        sa.Column('day_of_week', sa.Integer(), nullable=True),
        sa.Column('hour', sa.Integer(), nullable=False, server_default='9'),
        sa.Column('timezone', sa.String(length=50), nullable=False, server_default='UTC'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('last_run_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('next_run_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_scheduled_reports_user_id', 'scheduled_reports', ['user_id'], unique=False)
    op.create_index('ix_scheduled_reports_user_active', 'scheduled_reports', ['user_id', 'is_active'], unique=False)
    op.create_index('ix_scheduled_reports_next_run', 'scheduled_reports', ['next_run_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_scheduled_reports_next_run', table_name='scheduled_reports')
    op.drop_index('ix_scheduled_reports_user_active', table_name='scheduled_reports')
    op.drop_index('ix_scheduled_reports_user_id', table_name='scheduled_reports')
    op.drop_table('scheduled_reports')

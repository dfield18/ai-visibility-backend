"""Initial schema with sessions, runs, and results tables.

Revision ID: 001
Revises:
Create Date: 2026-01-20

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create sessions table
    op.create_table(
        'sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', sa.String(length=255), nullable=False),
        sa.Column('total_spent', sa.Numeric(precision=10, scale=2), nullable=False, server_default='0.00'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id')
    )
    op.create_index('ix_sessions_session_id', 'sessions', ['session_id'], unique=True)
    op.create_index('ix_sessions_expires_at', 'sessions', ['expires_at'], unique=False)

    # Create runs table
    op.create_table(
        'runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='queued'),
        sa.Column('brand', sa.String(length=255), nullable=False),
        sa.Column('config', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('total_calls', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('completed_calls', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('failed_calls', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('estimated_cost', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('actual_cost', sa.Numeric(precision=10, scale=4), nullable=False, server_default='0.0000'),
        sa.Column('cancelled', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_runs_session_status', 'runs', ['session_id', 'status'], unique=False)
    op.create_index('ix_runs_created_at', 'runs', ['created_at'], unique=False)

    # Create results table
    op.create_table(
        'results',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('provider', sa.String(length=50), nullable=False),
        sa.Column('model', sa.String(length=100), nullable=False),
        sa.Column('temperature', sa.Numeric(precision=3, scale=2), nullable=False),
        sa.Column('repeat_index', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('response_text', sa.Text(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('brand_mentioned', sa.Boolean(), nullable=True),
        sa.Column('competitors_mentioned', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('response_type', sa.String(length=20), nullable=True),
        sa.Column('tokens', sa.Integer(), nullable=True),
        sa.Column('cost', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_results_run_id', 'results', ['run_id'], unique=False)
    op.create_index('ix_results_created_at', 'results', ['created_at'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order due to foreign key constraints
    op.drop_index('ix_results_created_at', table_name='results')
    op.drop_index('ix_results_run_id', table_name='results')
    op.drop_table('results')

    op.drop_index('ix_runs_created_at', table_name='runs')
    op.drop_index('ix_runs_session_status', table_name='runs')
    op.drop_table('runs')

    op.drop_index('ix_sessions_expires_at', table_name='sessions')
    op.drop_index('ix_sessions_session_id', table_name='sessions')
    op.drop_table('sessions')

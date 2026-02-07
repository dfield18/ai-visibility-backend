"""Add site_audits table.

Revision ID: 008
Revises: 007
Create Date: 2026-02-07

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '008'
down_revision: Union[str, None] = '007'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create site_audits table
    op.create_table(
        'site_audits',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('url', sa.String(length=2048), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='queued'),
        sa.Column('results', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('overall_score', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_site_audits_session_id', 'site_audits', ['session_id'], unique=False)
    op.create_index('ix_site_audits_user_id', 'site_audits', ['user_id'], unique=False)
    op.create_index('ix_site_audits_url', 'site_audits', ['url'], unique=False)
    op.create_index('ix_site_audits_created_at', 'site_audits', ['created_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_site_audits_created_at', table_name='site_audits')
    op.drop_index('ix_site_audits_url', table_name='site_audits')
    op.drop_index('ix_site_audits_user_id', table_name='site_audits')
    op.drop_index('ix_site_audits_session_id', table_name='site_audits')
    op.drop_table('site_audits')

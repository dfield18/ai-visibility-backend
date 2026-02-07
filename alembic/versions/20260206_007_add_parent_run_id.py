"""Add parent_run_id to runs table for run extension feature.

Revision ID: 007
Revises: 006
Create Date: 2026-02-06

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '007'
down_revision: Union[str, None] = '006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add parent_run_id column to runs table
    op.add_column(
        'runs',
        sa.Column(
            'parent_run_id',
            postgresql.UUID(as_uuid=True),
            nullable=True,
        )
    )

    # Add foreign key constraint
    op.create_foreign_key(
        'fk_runs_parent_run_id',
        'runs',
        'runs',
        ['parent_run_id'],
        ['id'],
        ondelete='SET NULL'
    )

    # Add index for efficient lookup of child runs
    op.create_index(
        'ix_runs_parent_run_id',
        'runs',
        ['parent_run_id'],
        unique=False
    )


def downgrade() -> None:
    op.drop_index('ix_runs_parent_run_id', table_name='runs')
    op.drop_constraint('fk_runs_parent_run_id', 'runs', type_='foreignkey')
    op.drop_column('runs', 'parent_run_id')

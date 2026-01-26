"""Add sentiment classification fields to results table.

Revision ID: 003
Revises: 002
Create Date: 2026-01-26

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add brand_sentiment column
    op.add_column(
        'results',
        sa.Column('brand_sentiment', sa.String(length=50), nullable=True)
    )
    # Add competitor_sentiments column (JSON dict mapping competitor name to sentiment)
    op.add_column(
        'results',
        sa.Column('competitor_sentiments', postgresql.JSON(astext_type=sa.Text()), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('results', 'competitor_sentiments')
    op.drop_column('results', 'brand_sentiment')

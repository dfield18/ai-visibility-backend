"""Add users, auth, and billing tables.

Revision ID: 004
Revises: 003
Create Date: 2026-02-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '004'
down_revision: Union[str, None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('avatar_url', sa.String(length=500), nullable=True),
        sa.Column('email_verified', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('stripe_customer_id', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('stripe_customer_id')
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=True)
    op.create_index('ix_users_stripe_customer_id', 'users', ['stripe_customer_id'], unique=True)

    # Create auth_accounts table
    op.create_table(
        'auth_accounts',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('provider', sa.String(length=50), nullable=False),
        sa.Column('provider_account_id', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_auth_accounts_user_id', 'auth_accounts', ['user_id'], unique=False)
    op.create_index(
        'ix_auth_accounts_provider_account',
        'auth_accounts',
        ['provider', 'provider_account_id'],
        unique=True,
        postgresql_where=sa.text('provider_account_id IS NOT NULL')
    )

    # Create magic_link_tokens table
    op.create_table(
        'magic_link_tokens',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('token_hash', sa.String(length=64), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('token_hash')
    )
    op.create_index('ix_magic_link_tokens_email', 'magic_link_tokens', ['email'], unique=False)
    op.create_index('ix_magic_link_tokens_token_hash', 'magic_link_tokens', ['token_hash'], unique=True)

    # Create subscriptions table
    op.create_table(
        'subscriptions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('stripe_subscription_id', sa.String(length=255), nullable=False),
        sa.Column('stripe_price_id', sa.String(length=255), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='incomplete'),
        sa.Column('current_period_start', sa.DateTime(timezone=True), nullable=True),
        sa.Column('current_period_end', sa.DateTime(timezone=True), nullable=True),
        sa.Column('cancel_at_period_end', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('canceled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('stripe_subscription_id')
    )
    op.create_index('ix_subscriptions_user_id', 'subscriptions', ['user_id'], unique=False)
    op.create_index('ix_subscriptions_status', 'subscriptions', ['status'], unique=False)
    op.create_index('ix_subscriptions_stripe_subscription_id', 'subscriptions', ['stripe_subscription_id'], unique=True)

    # Create usage_records table
    op.create_table(
        'usage_records',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('credits_used', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('cost', sa.Numeric(precision=10, scale=4), nullable=False, server_default='0.0000'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_usage_records_user_id', 'usage_records', ['user_id'], unique=False)
    op.create_index('ix_usage_records_created_at', 'usage_records', ['created_at'], unique=False)
    op.create_index('ix_usage_records_user_created', 'usage_records', ['user_id', 'created_at'], unique=False)

    # Add user_id column to runs table
    op.add_column('runs', sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        'fk_runs_user_id',
        'runs',
        'users',
        ['user_id'],
        ['id'],
        ondelete='SET NULL'
    )
    op.create_index('ix_runs_user_id', 'runs', ['user_id'], unique=False)


def downgrade() -> None:
    # Remove user_id from runs table
    op.drop_index('ix_runs_user_id', table_name='runs')
    op.drop_constraint('fk_runs_user_id', 'runs', type_='foreignkey')
    op.drop_column('runs', 'user_id')

    # Drop usage_records table
    op.drop_index('ix_usage_records_user_created', table_name='usage_records')
    op.drop_index('ix_usage_records_created_at', table_name='usage_records')
    op.drop_index('ix_usage_records_user_id', table_name='usage_records')
    op.drop_table('usage_records')

    # Drop subscriptions table
    op.drop_index('ix_subscriptions_stripe_subscription_id', table_name='subscriptions')
    op.drop_index('ix_subscriptions_status', table_name='subscriptions')
    op.drop_index('ix_subscriptions_user_id', table_name='subscriptions')
    op.drop_table('subscriptions')

    # Drop magic_link_tokens table
    op.drop_index('ix_magic_link_tokens_token_hash', table_name='magic_link_tokens')
    op.drop_index('ix_magic_link_tokens_email', table_name='magic_link_tokens')
    op.drop_table('magic_link_tokens')

    # Drop auth_accounts table
    op.drop_index('ix_auth_accounts_provider_account', table_name='auth_accounts')
    op.drop_index('ix_auth_accounts_user_id', table_name='auth_accounts')
    op.drop_table('auth_accounts')

    # Drop users table
    op.drop_index('ix_users_stripe_customer_id', table_name='users')
    op.drop_index('ix_users_email', table_name='users')
    op.drop_table('users')

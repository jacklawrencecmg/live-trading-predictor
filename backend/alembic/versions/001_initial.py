"""initial schema

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'trades',
        sa.Column('id', sa.Integer, primary_key=True, index=True),
        sa.Column('symbol', sa.String(20), nullable=False, index=True),
        sa.Column('option_symbol', sa.String(50), nullable=True),
        sa.Column('action', sa.String(10), nullable=False),
        sa.Column('quantity', sa.Integer, nullable=False),
        sa.Column('price', sa.Float, nullable=False),
        sa.Column('underlying_price', sa.Float, nullable=True),
        sa.Column('strike', sa.Float, nullable=True),
        sa.Column('expiry', sa.String(12), nullable=True),
        sa.Column('option_type', sa.String(4), nullable=True),
        sa.Column('delta', sa.Float, nullable=True),
        sa.Column('iv', sa.Float, nullable=True),
        sa.Column('model_prob_up', sa.Float, nullable=True),
        sa.Column('model_prob_down', sa.Float, nullable=True),
        sa.Column('model_confidence', sa.Float, nullable=True),
        sa.Column('executed_at', sa.DateTime, nullable=True),
        sa.Column('is_paper', sa.Boolean, default=True),
    )
    op.create_table(
        'positions',
        sa.Column('id', sa.Integer, primary_key=True, index=True),
        sa.Column('symbol', sa.String(20), nullable=False, index=True),
        sa.Column('option_symbol', sa.String(50), nullable=True),
        sa.Column('quantity', sa.Integer, nullable=False),
        sa.Column('avg_cost', sa.Float, nullable=False),
        sa.Column('current_price', sa.Float, nullable=True),
        sa.Column('unrealized_pnl', sa.Float, default=0.0),
        sa.Column('realized_pnl', sa.Float, default=0.0),
        sa.Column('opened_at', sa.DateTime, nullable=True),
        sa.Column('closed_at', sa.DateTime, nullable=True),
        sa.Column('is_open', sa.Boolean, default=True),
        sa.Column('strike', sa.Float, nullable=True),
        sa.Column('expiry', sa.String(12), nullable=True),
        sa.Column('option_type', sa.String(4), nullable=True),
    )
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer, primary_key=True, index=True),
        sa.Column('event_type', sa.String(50), nullable=False, index=True),
        sa.Column('symbol', sa.String(20), nullable=True, index=True),
        sa.Column('details', sa.JSON, nullable=True),
        sa.Column('message', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=True, index=True),
    )
    op.create_table(
        'backtest_results',
        sa.Column('id', sa.Integer, primary_key=True, index=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('interval', sa.String(10), nullable=False),
        sa.Column('start_date', sa.String(12), nullable=False),
        sa.Column('end_date', sa.String(12), nullable=False),
        sa.Column('n_folds', sa.Integer, nullable=False),
        sa.Column('train_size', sa.Integer, nullable=False),
        sa.Column('test_size', sa.Integer, nullable=False),
        sa.Column('accuracy', sa.Float, nullable=True),
        sa.Column('brier_score', sa.Float, nullable=True),
        sa.Column('log_loss', sa.Float, nullable=True),
        sa.Column('magnitude_mae', sa.Float, nullable=True),
        sa.Column('sharpe_ratio', sa.Float, nullable=True),
        sa.Column('total_return', sa.Float, nullable=True),
        sa.Column('n_trades', sa.Integer, nullable=True),
        sa.Column('fold_results', sa.JSON, nullable=True),
        sa.Column('calibration_data', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=True),
    )


def downgrade():
    op.drop_table('backtest_results')
    op.drop_table('audit_logs')
    op.drop_table('positions')
    op.drop_table('trades')

"""initial_migration_document_table_and_pgvector

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-01-19

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create document table for PDF metadata
    op.create_table(
        "document",
        sa.Column("document_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("page_count", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("chunk_count", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column(
            "status",
            sa.String(length=50),
            nullable=False,
            server_default="processing",
        ),
        sa.Column(
            "create_time",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "update_time",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("document_id"),
    )


def downgrade() -> None:
    op.drop_table("document")
    op.execute("DROP EXTENSION IF EXISTS vector")

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core import database_session


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with database_session.get_async_session() as session:
        yield session

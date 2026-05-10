from app.db.session import AsyncSessionLocal, engine, Base
from app.rag.seed import ingest_docs_txt, seed_corpus_if_empty


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSessionLocal() as session:
        await seed_corpus_if_empty(session)
        await ingest_docs_txt(session)

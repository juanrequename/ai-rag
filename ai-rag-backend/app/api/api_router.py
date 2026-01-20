from fastapi import APIRouter

from app.api.endpoints import rag

api_router = APIRouter()
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])

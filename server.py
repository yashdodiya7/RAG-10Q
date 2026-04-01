"""
FastAPI server entry point.
Run with: uvicorn server:app --reload
"""

from src.api.app import create_app
from src.api.routes import router

app = create_app()

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    from src.config import get_settings
    
    settings = get_settings()
    
    uvicorn.run(
        "server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.ENV == "development",
        workers=settings.API_WORKERS,
    )

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .core.config import get_settings
from .core.metrics import setup_metrics
from .services.cache import get_cache
from .services.sentiment import get_sentiment_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - load model on startup."""
    settings = get_settings()
    print(f"Starting {settings.app_name} v{settings.app_version}")

    setup_metrics()

    try:
        service = get_sentiment_service()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Warning: Failed to load model on startup: {e}")

    yield

    try:
        await get_cache().close()
    except Exception:
        pass
    print("Shutting down application")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Financial sentiment analysis API using ONNX model",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    workers = 1 if settings.debug else settings.workers
    uvicorn.run(
        "sentiment_api.main:app",
        host=settings.host,
        port=settings.port,
        workers=workers,
        reload=settings.debug,
    )

"""
NeuroPulse AI — FastAPI Backend Main Entry Point
Registers all routers, middleware, and WebSocket endpoints.
Run: uvicorn backend.app.main:app --reload --port 8001
"""

import sys
sys.path.append('.')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from backend.app.api.routes import (
    patients, signals, inference,
    alerts, benchmark, xai, analyze, records
)
from backend.app.api.websocket.manager import websocket_router
from backend.app.database import create_tables

# ── App initialization ─────────────────────────────────────────────────────────
app = FastAPI(
    title="NeuroPulse AI API",
    description="""
    NeuroPulse AI — Neuromorphic Platform for Multi-Modal
    Biosignal Monitoring and Anomaly Detection Using SNNs.
    
    Provides REST APIs and WebSocket endpoints for:
    - Patient monitoring
    - Biosignal streaming
    - SNN inference
    - Anomaly alerts
    - XAI attribution
    - Model benchmarking
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── CORS Middleware ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(patients.router,
                   prefix="/api/patients",
                   tags=["Patients"])

app.include_router(signals.router,
                   prefix="/api/signals",
                   tags=["Signals"])

app.include_router(inference.router,
                   prefix="/api/inference",
                   tags=["Inference"])

app.include_router(alerts.router,
                   prefix="/api/alerts",
                   tags=["Alerts"])

app.include_router(benchmark.router,
                   prefix="/api/benchmark",
                   tags=["Benchmarking"])

app.include_router(xai.router,
                   prefix="/api/xai",
                   tags=["XAI"])

app.include_router(websocket_router)

app.include_router(analyze.router,
                   prefix="/api/analyze",
                   tags=["Analysis Engine"])

app.include_router(records.router,
                   prefix="/api/records",
                   tags=["Raw Records"])

# ── Startup Event ──────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup."""
    print("NeuroPulse AI Backend starting...")
    await create_tables()
    print("Database tables created.")
    print("API docs available at: http://localhost:8001/docs")


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "status":  "running",
        "project": "NeuroPulse AI",
        "version": "1.0.0",
        "docs":    "http://localhost:8001/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status":   "healthy",
        "database": "connected",
        "redis":    "connected",
        "models":   "loaded"
    }


if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )
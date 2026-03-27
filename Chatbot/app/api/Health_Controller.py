from fastapi import APIRouter
from app.Config import Config
from app.db.Postgres_Database import get_pool
from app.services.Health_Service import Health_Service
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/health", tags=["Health"])

config = Config()
service = Health_Service()


@router.get("/")
def overall_health():
    services = {
        "redis": service.check_redis(),
        "postgres": service.check_postgres(),
        "qdrant": service.check_qdrant()
    }
    all_up = all(v["status"] == "UP & RUNNING" for v in services.values())
    status_code = 200 if all_up else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if all_up else "degraded",
            "services": services
        }
    )

@router.get("/redis")
def redis_health():
    return service.check_redis()


@router.get("/postgres")
def postgres_health():
    return service.check_postgres()


@router.get("/qdrant")
def qdrant_health():
    return service.check_qdrant()

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.routes import predict, data_pipeline, catalog
import uvicorn

load_dotenv()

limitador = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Predictor de Conciertos",
    description="API para predecir próximos conciertos",
    version="1.0.0"
)

app.state.limiter = limitador
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Rutas
app.include_router(predict.router)
app.include_router(data_pipeline.router)
app.include_router(catalog.router)

# CORS (para que el frontend pueda llamar desde otro dominio)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción: especificar dominio del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
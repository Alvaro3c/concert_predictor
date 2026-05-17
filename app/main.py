from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import predict, data_pipeline
import uvicorn

load_dotenv()

app = FastAPI(
    title="Predictor de Conciertos",
    description="API para predecir próximos conciertos",
    version="1.0.0"
)

# Rutas
app.include_router(predict.router)
app.include_router(data_pipeline.router)

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
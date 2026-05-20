"""Dependencia FastAPI para proteger endpoints de administración con API key."""

import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

CABECERA_ADMIN = APIKeyHeader(name="X-Admin-Key", auto_error=False)


def verificar_clave_admin(clave_recibida: str = Security(CABECERA_ADMIN)) -> None:
    """
    Verifica que el header X-Admin-Key coincida con ADMIN_API_KEY del entorno.

    Lanza 403 si la clave es incorrecta o está ausente, sin revelar si el error
    es de ausencia o de valor incorrecto (misma respuesta para ambos casos).
    """
    clave_esperada = os.getenv("ADMIN_API_KEY")
    if not clave_esperada:
        raise HTTPException(status_code=500, detail="Error interno del servidor.")
    if not clave_recibida or clave_recibida != clave_esperada:
        raise HTTPException(status_code=403, detail="Acceso no autorizado.")

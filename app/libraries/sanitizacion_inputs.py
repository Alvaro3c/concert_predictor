"""Funciones de sanitización de inputs de usuario antes de inyectarlos en prompts de LLM."""

import re

LONGITUD_MAXIMA_ARTISTA = 150
LONGITUD_MAXIMA_PAIS = 100


def _sanitizar_texto_para_prompt(texto: str, longitud_maxima: int) -> str:
    """
    Elimina caracteres de control que permiten inyectar instrucciones en un prompt.

    Los saltos de línea, tabuladores y caracteres nulos son el vector principal de
    prompt injection: permiten añadir bloques de instrucciones nuevas tras el valor
    legítimo. Se mantienen letras acentuadas, puntuación y símbolos propios de
    nombres de artistas (AC/DC, P!nk, will.i.am, etc.).
    """
    texto_limpio = re.sub(r'[\x00-\x1f\x7f]', '', texto)
    return texto_limpio[:longitud_maxima].strip()


def sanitizar_nombre_artista(nombre_artista: str) -> str:
    """Sanitiza el nombre de artista recibido del usuario para uso seguro en prompts."""
    return _sanitizar_texto_para_prompt(nombre_artista, LONGITUD_MAXIMA_ARTISTA)


def sanitizar_nombre_pais(nombre_pais: str) -> str:
    """Sanitiza el nombre de país recibido del usuario para uso seguro en prompts."""
    return _sanitizar_texto_para_prompt(nombre_pais, LONGITUD_MAXIMA_PAIS)

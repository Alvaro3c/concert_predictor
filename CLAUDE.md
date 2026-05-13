# Proyecto: Concert Return Predictor

## Reglas de código
- NO usar programación orientada a objetos
- NO usar Pydantic, solo type hints
- Funciones simples y planas
- Comentarios en español
- las variables no deben de ser de 1 letra o abreviaturas

## Arquitectura
- routes/ → endpoints FastAPI
- modules/ → orquestadores
- libraries/ → funciones básicas reutilizables
- Funciones dentro de routes/ solo llaman a funciones de modules
- funciones dentro de modules/ solo llaman a funciones de libraries

## Stack
- Python 3.11
- FastAPI
- LastFM API para stats
- Scraping de concertsarchive.com
- Hugging Face para ML

## Convenciones
- Datos crudos en data/raw/
- Datos procesados en data/processed/
- JSONL como formato de almacenamiento
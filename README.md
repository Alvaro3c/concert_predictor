# Concerts predictor backend

## Entorno de desarrollo

### Descripción del proyecto

Concert Return Predictor es un backend desarrollado en Python que predice si un artista musical volverá a tocar en un país determinado, y en qué tramo temporal lo hará (A: rápido, B: medio, C: largo, D: sin historial suficiente). El sistema combina scraping de datos históricos de conciertos, ingeniería de features y un clasificador LightGBM, completado con una explicación en lenguaje natural generada por un LLM.

### Instalación y arranque

https://concert-predictor-client.vercel.app/

---

### Verificación del entorno

Para confirmar que todas las piezas responden correctamente antes de empezar a desarrollar:

1. **HuggingFace** — ejecutar `POST /data-pipeline/upload-dataset` con el dataset local. Si devuelve `{ "subido": true }` la conexión funciona.
2. **Groq** — ejecutar `GET /predict?artist=Green Day&country=Spain`. Si la respuesta incluye el campo `explicacion` con texto coherente, el LLM está operativo.
3. **Scraper** — ejecutar `POST /data-pipeline/scrape?artists_names=Radiohead` y verificar que `nuevos_registros > 0`.

## Selección de entorno y proveedor de IA

### Stack
- **Python 3.11 + FastAPI** como núcleo del backend
- **Visual Studio Code** como editor, con las extensiones Python y Pylance
- **LightGBM** para el modelo de clasificación por tramos
- **GitHub** como repositorio (usuario `Alvaro3c`, rama principal `main`, desarrollo en `develop`)
- **HuggingFace Hub** como almacenamiento del dataset enriquecido, accesible desde cualquier entorno sin necesidad de subir ficheros manualmente

### Proveedor de IA seleccionado

Se eligió **Groq** como proveedor LLM, usando el modelo **Llama 3** para generar la explicación en lenguaje natural que acompaña cada predicción en el endpoint `/predict`.

Los motivos de la elección frente a OpenAI o Gemini:

- **Velocidad** — Groq ofrece inferencia notablemente más rápida gracias a su hardware LPU, lo que mantiene el tiempo de respuesta del endpoint por debajo de 2 segundos incluso incluyendo la llamada al LLM
- **Coste** — capa gratuita suficiente para el volumen de peticiones del proyecto
- **Integración sencilla** — API compatible con el estándar OpenAI, lo que reduce el código de integración a unas pocas líneas

### Validación del entorno

Se realizó una petición real al endpoint de predicción una vez el modelo estaba entrenado:

```
GET /predict?artist=Green-Day&country=Spain
```

Respuesta obtenida:

```json
{
  "artista": "Green Day",
  "pais": "Spain",
  "tramo_predicho": "A",
  "probabilidades": { "A": 0.71, "B": 0.19, "C": 0.07, "D": 0.03 },
  "ultimo_concierto_conocido": "2024-06-15",
  "gap_medio_historico_dias": 398,
  "total_visitas_registradas": 9,
  "explicacion": "Green Day ha visitado España 9 veces con un intervalo medio de 398 días entre visitas. Su patrón histórico y actividad reciente sugieren un retorno en menos de 18 meses."
}
```

El campo `tramo_predicho: "A"` confirma que el modelo clasificó correctamente (retorno en menos de 18 meses), las probabilidades suman 1.0, y el campo `explicacion` demuestra que la llamada a Groq devolvió texto coherente con los datos reales del historial. El entorno funciona de extremo a extremo.

## Información y conocimiento del sistema (2.3)

### Qué información necesita el sistema

El predictor opera con dos fuentes de conocimiento:

- **Dataset de conciertos históricos** — 113.507 registros de conciertos entre 2001 y 2026 para los 95 artistas del catálogo. Cada registro contiene artista, país, fecha y las features calculadas (gaps históricos, frecuencia de visitas, contexto de gira, estación, etc.). Es la única fuente de verdad sobre el comportamiento pasado de cada artista.
- **Modelo entrenado + encoders** — el clasificador LightGBM y los encoders de artista y país persistidos en disco. Sin ellos el sistema no puede producir probabilidades por tramo.

### Cómo está almacenada y cómo se accede

| Recurso | Dónde vive | Cómo se accede |
|---|---|---|
| Dataset enriquecido | HuggingFace Hub (`Alvaro3c/alvaro-3c-concerts-dataset-from-2001-2026`) | Se descarga automáticamente al disco local en la primera petición si no existe (`data/processed/conciertos_enriquecido.jsonl`) |
| Modelo LightGBM | Disco local (`models/model_predict/model.joblib`) | Cargado en memoria en cada petición a `/predict` |
| Encoders | Disco local (`models/model_predict/`) | Cargados junto al modelo |

La búsqueda dentro del dataset no usa índice: en cada predicción se carga el JSONL completo en un DataFrame y se filtra por artista y país con `groupby`. Funciona para el volumen actual (~81k registros) pero no escalaría a catálogos grandes.

### Huecos actuales y soluciones que los cubrirían

1. **Sin historial de predicciones** — el sistema no registra nada de lo que predice. No hay forma de saber si el modelo acierta o se degrada con el tiempo sin revisión manual. Una tabla en base de datos (PostgreSQL o SQLite) que registre cada petición, el tramo predicho y —cuando se conozca— el resultado real, permitiría evaluar el modelo en producción.

2. **Búsqueda no indexada** — cargar 81k registros en memoria para filtrar por artista y país es ineficiente bajo carga concurrente. Migrar el dataset a una base de datos con índices sobre `artista` y `pais` reduciría la búsqueda de lineal a logarítmica.

3. **Dataset estático** — el catálogo no se actualiza automáticamente. Si un artista del catálogo da nuevos conciertos, el modelo trabaja con información desactualizada hasta que se vuelva a ejecutar el scraper manualmente.

## Despliegue (2.4)

### Frontend — Vercel

El cliente está desplegado en Vercel conectado directamente al repositorio de GitHub. Cada push a `main` dispara un redeploy automático.

URL pública: https://concert-predictor-client.vercel.app

### Backend — Render

El backend FastAPI está desplegado en Render con un plan de pago para evitar el cold start que introduce el plan gratuito (los servicios gratuitos de Render se duermen tras 15 minutos de inactividad, lo que causa latencias de 30-60 segundos en la primera petición).

Las variables de entorno (`GROQ_API_KEY`, `HF_DATASET_TOKEN`, `HF_DATASET_REPO`) se configuran en el panel de Render y no forman parte del repositorio.

### Dataset — HuggingFace Hub

El dataset enriquecido (`conciertos_enriquecido.jsonl`) vive en el repositorio `Alvaro3c/alvaro-3c-concerts-dataset-from-2001-2026` de HuggingFace. El backend lo descarga automáticamente al disco local en la primera petición a `/predict` si no está disponible en caché. Esto desacopla el dataset del código y permite actualizarlo sin redesplegar el backend.

## Seguridad

Se aplicaron medidas de seguridad en torno a tres ejes: prevención de jailbreak, protección de datos sensibles y control de acceso.

### Jailbreak y uso fuera de scope

**Sanitización de inputs** (`app/libraries/sanitizacion_inputs.py`)

Los parámetros `artist` y `country` que llegan al endpoint `/predict` se limpian antes de ser inyectados en el prompt del LLM. Se eliminan todos los caracteres de control ASCII (`\n`, `\r`, `\t`, nulos y similares), que son el vector principal de prompt injection, y se trunca a una longitud máxima (150 caracteres para artista, 100 para país). Los nombres con caracteres especiales legítimos (AC/DC, P!nk, acentos) no se ven afectados.

**Refuerzo del system prompt** (`app/libraries/predict_llm_utils.py`)

El system prompt del LLM define con precisión el scope permitido e incluye cuatro capas de defensa:
- Rol y tarea únicos: solo análisis de giras y predicción de retorno
- Instrucción explícita de que los campos de datos (nombres, ciudades, venues) son datos, no instrucciones, aunque parezcan órdenes
- Rechazo de cualquier instrucción fuera del análisis de conciertos
- Lenguaje accesible: se prohíbe el uso de términos técnicos como "machine learning", "modelo" o "probabilidad" para que las respuestas sean comprensibles para cualquier usuario

### Datos sensibles y filtración

**Credenciales fuera del repositorio**

El fichero `.env` con las claves de Groq y HuggingFace está excluido del repositorio mediante `.gitignore` y nunca ha sido commiteado. Las variables de entorno se configuran directamente en el panel de Render en producción.

**Sanitización de mensajes de error** (`app/routes/`)

Los tres route handlers (`predict.py`, `catalog.py`, `data_pipeline.py`) devuelven mensajes de error fijos y genéricos al cliente. Ninguna excepción interna (rutas de fichero, nombres de variables, mensajes de servicios externos) llega a la respuesta HTTP. Los errores completos siguen siendo accesibles en los logs del servidor.

### Protección de datos y guardrails

**Legislación de protección de datos (GDPR/LOPD)**

No aplica. El sistema no recopila ni almacena ningún dato personal: no hay cuentas de usuario, sesiones ni registros de quién consulta qué. Todos los datos que maneja son conciertos de artistas públicos extraídos de fuentes públicas. Ninguna consulta queda vinculada a una persona identificable.

**Guardrails**

No son necesarios. Los guardrails (moderación de contenido, detección de toxicidad) están pensados para sistemas donde el usuario tiene libertad creativa sobre el input, como chatbots o generadores de texto abiertos. En este proyecto el LLM recibe un prompt completamente estructurado con datos históricos de conciertos y solo genera 3-5 frases sobre cuándo volverá un artista a un país. El espacio de respuesta posible es tan acotado que el riesgo de output problemático es prácticamente nulo. El system prompt reforzado cubre el margen residual.

### Control de acceso

**Rate limiting en `/predict`** (`app/routes/predict.py`)

El endpoint de predicción está limitado a **15 peticiones por minuto por IP** mediante `slowapi`. Al superar el límite se devuelve `429 Too Many Requests` con el header `Retry-After`. El límite permite un uso intensivo normal sin restricciones pero bloquea bucles automatizados desde el primer segundo.

**API key en endpoints de administración** (`app/libraries/admin_auth.py`)

Los cinco endpoints de `/data-pipeline` (scraping, normalización, features, subida a HuggingFace y entrenamiento) requieren el header `X-Admin-Key` con el valor de la variable de entorno `ADMIN_API_KEY`. Sin él se devuelve `403 Forbidden`. La respuesta es idéntica tanto si la clave está ausente como si es incorrecta, para no revelar información sobre el fallo.

## Validación del modelo

El clasificador LightGBM se evalúa automáticamente al final de cada entrenamiento sobre un conjunto de test que nunca ha visto. El split es **temporal por artista** (80% de los conciertos más antiguos para train, 20% más recientes para test), lo que garantiza que el modelo no evalúa sobre datos que podría haber visto durante el entrenamiento.

Las métricas se persisten en `models/model_predict/metrics.json` tras cada ejecución de `POST /data-pipeline/train/model-predict`.

### Métricas del último entrenamiento (2026-05-17)

El test set contiene **4.475 registros reales** distribuidos entre los cuatro tramos:

| Tramo | Significado | Casos reales | Recall | Precisión | F1 |
|---|---|---|---|---|---|
| A | Retorno en menos de 1 año | 3.453 | 74,2% | 92,7% | 82,4% |
| B | Retorno entre 1 y 2 años | 642 | 39,7% | 26,5% | 31,8% |
| C | Retorno entre 2 y 4 años | 313 | 55,3% | 25,4% | 34,8% |
| D | Sin historial suficiente (regla explícita) | 67 | 100% | 100% | 100% |

**Accuracy global: 68,3%**

### Interpretación

El modelo tiene un comportamiento asimétrico que refleja el desequilibrio del dataset: el 77% de los casos son Tramo A, por lo que el modelo aprende bien ese patrón. El Tramo B es el más difícil de predecir (F1 31,8%), algo esperable dado que es la clase intermedia con mayor ambigüedad en los límites.

El Tramo D tiene métricas perfectas porque no lo aprende el modelo — se asigna por regla determinista antes de llamar al clasificador: si el artista tiene 0 visitas previas al país o un gap medio histórico superior a 4 años, el sistema lo clasifica directamente como D sin consultar al modelo.

## Inicializar
- uvicorn app.main:app --reload 

## Rutas

### Catálogo (`/catalog`)

1. **GET /catalog/countries**
   - Devuelve los países únicos disponibles en el dataset
   - Response: `["Spain", "United States", "Germany", ...]`

2. **GET /catalog/artists**
   - Devuelve los artistas únicos disponibles en el dataset
   - Response: `["Green Day", "Metallica", "Radiohead", ...]`

---

### Pipeline de datos (`/data-pipeline`)

3. **POST /data-pipeline/scrape**
   - Query params: `artists_names` (lista de strings, requerido), `until_year` (int, opcional)
   - Scrapea conciertos de concertarchives.org para cada artista y los guarda en `data/raw/concerts.jsonl`
   - Response:
     ```json
     {
       "hasta_año": 2010,
       "archivo": "data/raw/concerts.jsonl",
       "artistas_procesados": [
         {
           "artista": "Green Day",
           "slug": "green-day",
           "paginas_scrapeadas": 12,
           "nuevos_registros": 340,
           "duplicados_evitados_esta_sesion": 5
         }
       ],
       "artistas_fallidos": [],
       "total_nuevos_registros": 340
     }
     ```

4. **POST /data-pipeline/process/normalise**
   - Normaliza fechas, filtra registros inválidos y deduplica el JSONL crudo
   - Lee de `data/raw/concerts.jsonl`, escribe en `data/interim/normalized_concerts.jsonl`
   - Response:
     ```json
     {
       "total_procesados": 81000,
       "correctos_sin_cambio": 70000,
       "normalizados": 9000,
       "errores": 200,
       "eliminados": 800,
       "duplicados": 150
     }
     ```

5. **POST /data-pipeline/process/enrich-features**
   - Ejecuta el pipeline completo de features en orden: global → tour → country → country_global → context
   - Sube el resultado a HuggingFace al finalizar
   - Lee de `data/interim/normalized_concerts.jsonl`, escribe en `data/processed/conciertos_enriquecido.jsonl`
   - Response:
     ```json
     {
       "global":         { "registros_procesados": 81000, "errores_fecha": 0 },
       "tour":           { "registros_procesados": 81000 },
       "country":        { "registros_procesados": 81000 },
       "country_global": { "registros_procesados": 81000 },
       "context":        { "registros_procesados": 81000 },
       "subida_hf":      { "subido": true, "repo": "user/dataset", "fichero": "conciertos_enriquecido.jsonl" }
     }
     ```

6. **POST /data-pipeline/upload-dataset**
   - Sube el dataset enriquecido existente a HuggingFace sin regenerarlo
   - Response:
     ```json
     { "subido": true, "repo": "user/dataset", "fichero": "conciertos_enriquecido.jsonl" }
     ```

7. **POST /data-pipeline/train/model-predict**
   - Entrena el modelo LightGBM con los datos procesados y lo guarda en `models/model_predict/`
   - Response:
     ```json
     {
       "metricas": {
         "bucket_accuracy_global": 0.78,
         "bucket_accuracy_por_tramo": { "A": 0.82, "B": 0.65, "C": 0.71, "D": 1.0 }
       },
       "ruta_modelo": "models/model_predict/model.joblib"
     }
     ```

---

### Predicción (`/predict`)

8. **GET /predict**
   - Query params: `artist` (str), `country` (str)
   - Predice el tramo de retorno de un artista a un país basándose en su historial
   - Tramos: **A** (retorno rápido) · **B** (retorno medio) · **C** (retorno largo) · **D** (sin historial suficiente)
   - Response:
     ```json
     {
       "artista": "Green Day",
       "pais": "Spain",
       "tramo_predicho": "A",
       "probabilidades": { "A": 0.72, "B": 0.18, "C": 0.07, "D": 0.03 },
       "ultimo_concierto_conocido": "2023-06-15",
       "gap_medio_historico_dias": 420,
       "total_visitas_registradas": 8,
       "explicacion": "Green Day ha visitado Spain 8 veces con un gap medio de 420 días..."
     }
     ```

## features para entrenar el modelo
### Artist Global Features

Calculadas con los conciertos previos del artista en cualquier país.

- **conciertos_ultimo_año** — Conciertos en los 365 días previos a la fecha del registro.
- **conciertos_ultimos_3_años** — Conciertos en los 1095 días previos.
- **conciertos_ultimos_5_años** — Conciertos en los 1825 días previos.
- **dias_desde_ultimo_concierto_global** — Días desde el concierto más reciente del artista en cualquier país.
- **tendencia_actividad** — Ratio entre conciertos del último año y la media anual de los 5
anteriores. Detecta si el artista está acelerando o desacelerando su
actividad.
- **años_observados_artista** — Años desde el primer concierto del artista en el dataset hasta la
fecha del registro. Proxy de antigüedad e indicador de fiabilidad de las demás features.

### Artist Tour Features

Calculadas para detectar si el artista está en medio de una gira activa.

- **conciertos_artista_ultimos_30_dias** — Conciertos en los últimos 30 días. Si es alto, indica gira activa.
- **conciertos_artista_ultimos_90_dias** — Conciertos en los últimos 90 días.
- **en_gira_activa** — Binaria. `True` si `conciertos_ultimos_30_dias >= 3`.
- **dias_desde_concierto_anterior** — Gap en días con el concierto inmediatamente anterior del artista en cualquier país. Si es 1-3 días, está en mitad de un tour.
- **paises_distintos_ultimos_30_dias** — Número de países distintos visitados en los últimos 30 días. Confirma si es gira internacional.

### Artist × Country Features

Calculadas para el par artista + país concreto.

- **visitas_previas_al_pais** — Número de veces que el artista ha tocado en ese país antes de la fecha del registro.
- **es_primera_visita_observada** — Binaria. `True` si `visitas_previas_al_pais == 0`.
- **dias_desde_ultima_visita_pais** — Días desde el último concierto del artista en ese país.
- **gap_medio_pais** — Media de los gaps históricos entre visitas del artista a ese país (en días).
- **gap_mediano_pais** — Mediana de los gaps. Más robusta a outliers que la media.
- **gap_std_pais** — Desviación estándar de los gaps. Mide la regularidad de las visitas.
- **gap_min_pais** — Gap mínimo registrado entre visitas. Límite inferior del comportamiento histórico.
- **gap_max_pais** — Gap máximo registrado entre visitas. Límite superior del comportamiento histórico.
- **ultimo_gap_pais** — El gap más reciente registrado (entre las dos últimas visitas anteriores al registro).
- **tendencia_gap_pais** — Compara el último gap con la media histórica. Indica si los gaps se están acortando o alargando.
- **proporcion_visitas_pais** — `visitas_previas_al_pais / conciertos_totales_previos`. Mide qué tan importante es ese país para el artista.
- **rank_pais_para_artista** — Posición del país en el ranking de países más visitados por el
artista (1 = más visitado). Si está en el top 3, es probable visita
frecuente.

### Country Global Features

Calculadas considerando todos los artistas del dataset para ese país.

- **paises_unicos_visitados_total** — Países únicos visitados por el artista hasta la fecha del registro.
- **paises_unicos_visitados_ultimos_3_años** — Países únicos visitados en los 1095 días previos.
- **paises_unicos_visitados_ultimos_5_años** — Países únicos visitados en los 1825 días previos.
- **ciudades_unicas_ultimos_5_años** — Ciudades únicas visitadas en los 1825 días previos.
- **ratio_paises_conciertos_5y** — `países_únicos / conciertos_totales` en los últimos 5 años. Distingue giras distribuidas de giras concentradas en pocos países.
- **conciertos_totales_pais_previos** — Total de conciertos de todos los artistas del dataset en ese país antes de la fecha. Proxy del tamaño del mercado.
- **gap_medio_global_pais** — Gap medio entre visitas a ese país considerando todos los artistas.
Captura cada cuánto vuelven los artistas en general a ese país.

### Context Features

Calculadas a partir de la fecha del registro, sin necesidad de historial.

- **estacion** — Estación del año en la fecha del registro. Categórica: `invierno`, `primavera`, `verano`, `otoño`.
- **es_post_covid** — Binaria. `True` si `fecha > 2022-01-01`.
- **es_periodo_covid** — Binaria. `True` si `fecha` está entre `2020-03` y `2021-12`. Permite al modelo tratar ese período de forma separada.
import re
import json
import os
import time
import random
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from config import (
    EXTRA_STEALTH_MODE,
    STEALTH_DELAY_MIN_S,
    STEALTH_DELAY_MAX_S,
    STEALTH_PROBABILIDAD_PAUSA_FANTASMA,
    STEALTH_PAUSA_LARGA_MIN_S,
    STEALTH_PAUSA_LARGA_MAX_S,
)


# Separadores de rango de fechas: en dash (–) y em dash (—)
SEPARADOR_RANGO = re.compile(r'\s*[–—]\s*')

FORMATOS_FECHA = ["%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]


def _convertir_fecha_simple(texto: str) -> str:
    # Convierte un string de fecha limpio (sin rangos) a ISO 8601
    texto = texto.strip()
    for formato in FORMATOS_FECHA:
        try:
            return datetime.strptime(texto, formato).strftime("%Y-%m-%d")
        except ValueError:
            continue
    print(f"[debug] No se pudo convertir la fecha '{texto}' a ISO 8601, se conserva el valor original.")
    return texto


def normalizar_rango_fecha(fecha_raw: str) -> tuple[str, str | None]:
    # Colapsa saltos de línea y espacios múltiples, luego separa rango si existe
    texto_limpio = re.sub(r'\s+', ' ', fecha_raw).strip()
    partes = SEPARADOR_RANGO.split(texto_limpio, maxsplit=1)
    if len(partes) == 2 and partes[1].strip():
        return _convertir_fecha_simple(partes[0]), _convertir_fecha_simple(partes[1])
    return _convertir_fecha_simple(texto_limpio), None


def convertir_fecha_iso(fecha_raw: str) -> str:
    # Compatibilidad: devuelve solo la fecha de inicio del rango (o la fecha única)
    fecha_inicio, _ = normalizar_rango_fecha(fecha_raw)
    return fecha_inicio


def artist_to_slug(artist_name: str) -> str:
    slug = artist_name.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug.strip())
    return slug


HEADERS_NAVEGADOR = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def crear_sesion() -> requests.Session:
    sesion = requests.Session()
    sesion.headers.update(HEADERS_NAVEGADOR)
    return sesion


def fetch_page(url: str, sesion: requests.Session | None = None) -> str:
    cliente = sesion if sesion is not None else requests
    resp = cliente.get(url, headers=HEADERS_NAVEGADOR if sesion is None else {}, timeout=15)
    resp.raise_for_status()
    print(f"[debug] Fetched {url} - Content Length: {len(resp.text)}")
    return resp.text


def get_total_pages(html: str) -> int:
    # Extrae el número total de páginas del paginador de concertarchives
    soup = BeautifulSoup(html, "html.parser")
    paginas = soup.select("ul.pagination li a")
    numeros = []
    for a in paginas:
        texto = a.get_text(strip=True)
        if texto.isdigit():
            numeros.append(int(texto))
    return max(numeros) if numeros else 1


def extraer_url_base_real(html: str, url_original: str) -> str:
    # El slug generado puede diferir del slug real del sitio (ej. blink-182 → blink-182--9)
    # Los links de paginación siempre usan el slug correcto
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.select("ul.pagination li a"):
        href = a.get("href", "")
        coincidencia = re.match(r"(/bands/[^?#]+)\?page=\d+", href)
        if coincidencia:
            return "https://www.concertarchives.org" + coincidencia.group(1)
    return url_original


def parse_conciertos_por_año(html: str) -> dict[int, int]:
    # Extrae {año: nº_conciertos} del sidebar de concertarchives
    soup = BeautifulSoup(html, "html.parser")
    celdas = soup.find_all("td", class_="table-cell-no-stretch", align="right")
    resultado = {}
    for celda in celdas:
        a = celda.find("a")
        if not a:
            continue
        match_año = re.search(r"year=(\d{4})", a.get("href", ""))
        match_num = re.search(r"(\d+)\s+concert", a.get_text())
        if match_año and match_num:
            resultado[int(match_año.group(1))] = int(match_num.group(1))
    return resultado


def _es_tabla_conciertos(table) -> bool:
    # Descarta tablas de estadísticas y tablas de duplicados anidadas
    clases = table.get("class", [])
    if "tops_table" in clases or "duplicate-concerts-table" in clases:
        return False
    if table.find_parent(class_="duplicate-concerts-container"):
        return False
    return True


def _filas_directas(table) -> list:
    # Devuelve solo las filas que pertenecen directamente a esta tabla,
    # excluyendo filas de tablas anidadas dentro de ella
    return [row for row in table.find_all("tr") if row.find_parent("table") is table]


def parse_concerts_from_table(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")

    # Recopila todas las tablas de conciertos de la página (upcoming + past)
    tablas_conciertos = [
        tabla for tabla in soup.find_all("table", class_="table")
        if _es_tabla_conciertos(tabla)
    ]

    if not tablas_conciertos:
        print("[debug] Error: no se encontraron tablas de conciertos.")
        return []

    conciertos = []
    for table in tablas_conciertos:
        for row in _filas_directas(table):
            # Salta filas que contienen tablas anidadas de duplicados
            if row.find("table"):
                continue

            cols = row.find_all("td")

            # La tabla puede tener 3 columnas (Fecha, Venue, Lugar)
            # o 4 columnas (Fecha, Gira/Tour, Venue, Lugar) para conciertos con gira nombrada
            if len(cols) >= 4:
                fecha_raw = cols[0].get_text(strip=True)
                venue = cols[2].get_text(strip=True)
                location = cols[3].get_text(strip=True)
            elif len(cols) == 3:
                fecha_raw = cols[0].get_text(strip=True)
                venue = cols[1].get_text(strip=True)
                location = cols[2].get_text(strip=True)
            else:
                continue

            if "Date" in fecha_raw or not fecha_raw:
                continue

            # Elimina etiquetas como "Upcoming" que aparecen pegadas a la fecha
            fecha_raw = re.sub(r'\s*(Upcoming|Past|TBA)\s*', ' ', fecha_raw).strip()

            fecha_inicio, fecha_fin = normalizar_rango_fecha(fecha_raw)
            loc_parts = [p.strip() for p in location.split(",") if p.strip()]

            conciertos.append({
                "fecha": fecha_inicio,
                "fecha_fin": fecha_fin,
                "venue": venue,
                "ciudad": loc_parts[0] if loc_parts else "",
                "pais": loc_parts[-1] if len(loc_parts) > 1 else "",
            })

    print(f"[debug] Successfully parsed {len(conciertos)} concerts.")
    return conciertos


def extraer_año(fecha: str) -> int | None:
    # Busca cualquier número de 4 dígitos que empiece por 19 o 20
    match = re.search(r"(19|20)\d{2}", fecha)
    if match:
        return int(match.group(0))
    print(f"[debug] Regex failed to find year in string: '{fecha}'")
    return None


def acumular_hasta_año(acumulado: list[dict], conciertos: list[dict], until_year: int | None) -> tuple[list[dict], bool]:
    if until_year is None:
        acumulado.extend(conciertos)
        return acumulado, False

    print(f"[debug] Revisando lote de {len(conciertos)} conciertos...")
    for i, concierto in enumerate(conciertos):
        año = extraer_año(concierto["fecha"])
        print(f"[debug] Fila {i}: fecha='{concierto['fecha']}' -> año={año}")

        if año is None:
            print(f"[debug] Fila {i}: omitida (año no encontrado)")
            continue

        if año < until_year:
            print(f"[scraper] Límite de año alcanzado: {año} < {until_year}. Deteniendo.")
            return acumulado, True

        acumulado.append(concierto)

    print(f"[debug] Lote procesado. Señal de parada: False")
    return acumulado, False


def human_delay(min_s: float = 6.0, max_s: float = 20.0) -> None:
    # Pausa aleatoria para simular comportamiento humano entre peticiones
    if EXTRA_STEALTH_MODE:
        min_s = STEALTH_DELAY_MIN_S
        max_s = STEALTH_DELAY_MAX_S
    pausa = random.uniform(min_s, max_s)
    time.sleep(pausa)


def _maybe_pausa_larga() -> None:
    # Dispara una pausa de 8-12 min con la probabilidad configurada en STEALTH_PROBABILIDAD_PAUSA_FANTASMA
    if not EXTRA_STEALTH_MODE:
        return
    if random.random() < STEALTH_PROBABILIDAD_PAUSA_FANTASMA:
        pausa = random.uniform(STEALTH_PAUSA_LARGA_MIN_S, STEALTH_PAUSA_LARGA_MAX_S)
        minutos = round(pausa / 60, 1)
        print(f"[scraper] Modo fantasma: pausa larga de {minutos} min.")
        time.sleep(pausa)


def load_existing_keys(filepath: str) -> set[tuple]:
    existentes = set()
    if not os.path.exists(filepath):
        return existentes
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                c = json.loads(line)
                # Incluye el artista para no filtrar bandas distintas en el mismo evento
                clave = (c.get("fecha"), c.get("venue"), c.get("ciudad"), c.get("pais"), c.get("artista"))
                existentes.add(clave)
            except json.JSONDecodeError:
                pass
    return existentes


def añadir_conciertos_previos(conciertos: list[dict], conciertos_por_año: dict[int, int]) -> list[dict]:
    # El más reciente recibe el total histórico; cada concierto anterior recibe total-1, total-2...
    total = sum(conciertos_por_año.values())
    ordenados = sorted(conciertos, key=lambda c: c.get("fecha", ""))
    n = len(ordenados)
    for i, c in enumerate(ordenados):
        c["conciertos_previos_hasta_la_fecha"] = total - (n - 1 - i)
    return ordenados


def save_to_jsonl(conciertos: list[dict], filepath: str) -> int:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        for c in conciertos:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return len(conciertos)


def añadir_nombre_artista(conciertos: list[dict], nombre_artista: str) -> list[dict]:
    for concierto in conciertos:
        concierto["artista"] = nombre_artista
    return conciertos


def filtrar_duplicados(todos_los_conciertos: list[dict], vistos: set[tuple]) -> tuple[list[dict], int]:
    conciertos_sin_duplicados = []
    for concierto in todos_los_conciertos:
        clave = (
            concierto.get("fecha"),
            concierto.get("venue"),
            concierto.get("ciudad"),
            concierto.get("pais"),
            concierto.get("artista"),
        )
        if clave not in vistos:
            vistos.add(clave)
            conciertos_sin_duplicados.append(concierto)
    duplicados_omitidos = len(todos_los_conciertos) - len(conciertos_sin_duplicados)
    return conciertos_sin_duplicados, duplicados_omitidos


def scrapear_paginas(
    url_base: str,
    nombre_artista: str,
    total_paginas: int,
    until_year: int | None,
    acumulado: list[dict],
    parar: bool,
    primer_concierto_pagina_anterior: dict | None,
    sesion: requests.Session | None = None,
) -> tuple[list[dict], int]:
    paginas_scrapeadas = 1
    for pagina in range(2, total_paginas + 1):
        if parar:
            print(f"[scraper] Límite de año alcanzado. Deteniendo en página {pagina - 1}.")
            break

        human_delay()
        _maybe_pausa_larga()
        url = f"{url_base}?page={pagina}"
        print(f"[scraper] Página {pagina}/{total_paginas} → {url}")

        html = fetch_page(url, sesion)
        batch = añadir_nombre_artista(parse_concerts_from_table(html), nombre_artista)

        if not batch:
            print(f"[debug] Sin conciertos en página {pagina}. Deteniendo.")
            break

        # El servidor a veces ignora el parámetro ?page= y devuelve siempre la primera página
        if primer_concierto_pagina_anterior and batch[0] == primer_concierto_pagina_anterior:
            print(f"[scraper] Datos duplicados en página {pagina}. El servidor ignoró la paginación.")
            break

        primer_concierto_pagina_anterior = batch[0]
        acumulado, parar = acumular_hasta_año(acumulado, batch, until_year)
        paginas_scrapeadas += 1

    return acumulado, paginas_scrapeadas

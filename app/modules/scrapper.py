from app.libraries.scrapper_utils import (
    artist_to_slug,
    crear_sesion,
    extraer_url_base_real,
    fetch_page,
    get_total_pages,
    parse_conciertos_por_año,
    parse_concerts_from_table,
    acumular_hasta_año,
    load_existing_keys,
    añadir_conciertos_previos,
    save_to_jsonl,
    añadir_nombre_artista,
    filtrar_duplicados,
    scrapear_paginas,
)

BASE_URL = "https://www.concertarchives.org/bands"


def run(artist_name: str, until_year: int | None = None) -> dict:
    slug = artist_to_slug(artist_name)
    url_base = f"{BASE_URL}/{slug}"
    filepath = "data/raw/concerts.jsonl"
    sesion = crear_sesion()

    print(f"[scraper] Iniciando scraping: {artist_name} (hasta_año: {until_year})")
    print(f"[scraper] Página 1 → {url_base}")
    html = fetch_page(url_base, sesion)
    url_base = extraer_url_base_real(html, url_base)
    print(f"[scraper] URL base real: {url_base}")
    total_paginas = get_total_pages(html)
    conciertos_por_año = parse_conciertos_por_año(html)

    batch_pagina_1 = añadir_nombre_artista(parse_concerts_from_table(html), artist_name)
    primer_concierto = batch_pagina_1[0] if batch_pagina_1 else None
    todos_los_conciertos, parar = acumular_hasta_año([], batch_pagina_1, until_year)

    todos_los_conciertos, paginas_scrapeadas = scrapear_paginas(
        url_base, artist_name, total_paginas, until_year, todos_los_conciertos, parar, primer_concierto, sesion
    )

    vistos = load_existing_keys(filepath)
    conciertos_sin_duplicados, duplicados_omitidos = filtrar_duplicados(todos_los_conciertos, vistos)
    print(f"[scraper] Deduplicación completa: {len(conciertos_sin_duplicados)} registros nuevos únicos.")

    if conciertos_sin_duplicados:
        conciertos_sin_duplicados = añadir_conciertos_previos(conciertos_sin_duplicados, conciertos_por_año)
        total_guardado = save_to_jsonl(conciertos_sin_duplicados, filepath)
    else:
        total_guardado = 0

    return {
        "artista": artist_name,
        "slug": slug,
        "hasta_año": until_year,
        "paginas_scrapeadas": paginas_scrapeadas,
        "nuevos_registros": total_guardado,
        "duplicados_evitados_esta_sesion": duplicados_omitidos,
        "archivo": filepath,
    }

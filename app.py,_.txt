import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Predictor Probabilístico y Value Bets",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Predictor Probabilístico + Detector de Value Bets")
st.markdown("""
Esta herramienta combina el **Modelo de Poisson** alimentado por la API de **Football-Data.org** con un **Detector de Apuestas de Valor (*Value Bets*)** para identificar oportunidades donde las casas de apuestas pagan cuotas superiores al riesgo real.
""")

# ==========================================
# 2. BARRA LATERAL: CONFIGURACIÓN
# ==========================================
st.sidebar.header("🔑 Configuración de API")
api_key = st.sidebar.text_input(
    "API Key (Football-Data.org)", 
    type="password", 
    help="Consigue tu API Key gratuita en https://www.football-data.org/"
)

LIGAS_DISPONIBLES = {
    "Premier League (Inglaterra)": "PL",
    "LaLiga (España)": "PD",
    "Serie A (Italia)": "SA",
    "Bundesliga (Alemania)": "BL1",
    "Ligue 1 (Francia)": "FL1",
    "Primeira Liga (Portugal)": "PPL",
    "Eredivisie (Países Bajos)": "DED",
    "UEFA Champions League": "CL"
}

liga_seleccionada = st.sidebar.selectbox("Seleccionar Competición", list(LIGAS_DISPONIBLES.keys()))
code_liga = LIGAS_DISPONIBLES[liga_seleccionada]

# ==========================================
# 3. CONEXIÓN CON LA API DE DATOS
# ==========================================
@st.cache_data(ttl=1800)
def obtener_partidos_proximos(api_key, competition_code):
    url = f"https://api.football-data.org/v4/competitions/{competition_code}/matches?status=SCHEDULED"
    headers = {"X-Auth-Token": api_key}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get("matches", [])
        return []
    except Exception:
        return []

@st.cache_data(ttl=3600)
def obtener_estadisticas_equipos(api_key, competition_code):
    url = f"https://api.football-data.org/v4/competitions/{competition_code}/standings"
    headers = {"X-Auth-Token": api_key}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            table = response.json()["standings"][0]["table"]
            stats, total_goles, total_partidos = {}, 0, 0
            for row in table:
                team_id = row["team"]["id"]
                played, gf, ga = row["playedGames"], row["goalsFor"], row["goalsAgainst"]
                if played > 0:
                    stats[team_id] = {
                        "name": row["team"]["name"],
                        "prom_gf": gf / played,
                        "prom_ga": ga / played
                    }
                    total_goles += gf
                    total_partidos += played
            promedio_liga = (total_goles / total_partidos) if total_partidos > 0 else 1.35
            return stats, promedio_liga
        return {}, 1.35
    except Exception:
        return {}, 1.35

# ==========================================
# 4. MOTOR MATEMÁTICO & VALUE BETS
# ==========================================
def calcular_xg_poisson(stats_local, stats_vis, promedio_liga):
    fuerza_ataque_local = stats_local["prom_gf"] / promedio_liga
    fuerza_defensa_vis = stats_vis["prom_ga"] / promedio_liga
    fuerza_ataque_vis = stats_vis["prom_gf"] / promedio_liga
    fuerza_defensa_local = stats_local["prom_ga"] / promedio_liga
    
    FACTOR_LOCALIA = 1.10
    xg_local = fuerza_ataque_local * fuerza_defensa_vis * promedio_liga * FACTOR_LOCALIA
    xg_vis = fuerza_ataque_vis * fuerza_defensa_local * promedio_liga
    return max(0.2, round(xg_local, 2)), max(0.2, round(xg_vis, 2))

def generar_matriz_probabilidades(lambda_local, lambda_vis, max_goles=5):
    prob_l = [poisson.pmf(i, lambda_local) for i in range(max_goles + 1)]
    prob_v = [poisson.pmf(j, lambda_vis) for j in range(max_goles + 1)]
    return np.outer(prob_l, prob_v)

def calcular_valor_apuesta(probabilidad_estimada, cuota_casa):
    p = probabilidad_estimada / 100.0
    ev = (p * cuota_casa) - 1.0
    ev_porcentaje = ev * 100.0
    
    q = 1.0 - p
    b = cuota_casa - 1.0
    kelly_fraction = max(0.0, (b * p - q) / b) if b > 0 else 0.0
    kelly_recomendado = (kelly_fraction * 0.25) * 100.0
    
    return ev_porcentaje, kelly_recomendado

# ==========================================
# 5. DESPLIEGUE PRINCIPAL
# ==========================================
if not api_key:
    st.info("👈 Para empezar, introduce tu **API Key de Football-Data.org** en la barra lateral.")
else:
    partidos = obtener_partidos_proximos(api_key, code_liga)
    stats_equipos, prom_liga = obtener_estadisticas_equipos(api_key, code_liga)

    if partidos and stats_equipos:
        opciones_partidos = {f"{p['homeTeam']['name']} vs {p['awayTeam']['name']} ({p['utcDate'][:10]})": p for p in partidos}
        partido_elegido_str = st.selectbox("📅 Selecciona un partido:", list(opciones_partidos.keys()))
        partido_datos = opciones_partidos[partido_elegido_str]
        
        id_h, id_a = partido_datos["homeTeam"]["id"], partido_datos["awayTeam"]["id"]
        
        if id_h in stats_equipos and id_a in stats_equipos:
            stats_h, stats_a = stats_equipos[id_h], stats_equipos[id_a]
            xg_h, xg_a = calcular_xg_poisson(stats_h, stats_a, prom_liga)
            matriz = generar_matriz_probabilidades(xg_h, xg_a)
            
            p1 = np.sum(np.tril(matriz, -1)) * 100
            px = np.sum(np.diag(matriz)) * 100
            p2 = np.sum(np.triu(matriz, 1)) * 100
            
            c1_fair = round(100 / p1, 2) if p1 > 0 else 0
            cx_fair = round(100 / px, 2) if px > 0 else 0
            c2_fair = round(100 / p2, 2) if p2 > 0 else 0

            # SECCIÓN APUESTAS DE VALOR (COMPARADOR)
            st.markdown("---")
            st.subheader("🤑 Detector de Apuestas de Valor (Value Bets)")
            st.markdown("Ingresa las cuotas ofrecidas por tu casa de apuestas favorita para evaluar el **Valor Esperado (+EV)**:")
            
            col_inp1, col_inp2, col_inp3 = st.columns(3)
            with col_inp1:
                cuota_bookie_1 = st.number_input(f"Cuota Casa para {stats_h['name']} (1)", value=float(c1_fair), step=0.05)
            with col_inp2:
                cuota_bookie_x = st.number_input("Cuota Casa para Empate (X)", value=float(cx_fair), step=0.05)
            with col_inp3:
                cuota_bookie_2 = st.number_input(f"Cuota Casa para {stats_a['name']} (2)", value=float(c2_fair), step=0.05)
            
            ev1, kelly1 = calcular_valor_apuesta(p1, cuota_bookie_1)
            evx, kellyx = calcular_valor_apuesta(px, cuota_bookie_x)
            ev2, kelly2 = calcular_valor_apuesta(p2, cuota_bookie_2)
            
            res_col1, res_col2, res_col3 = st.columns(3)
            
            def mostrar_evaluacion(col, opcion, p_est, c_fair, c_bookie, ev, kelly):
                with col:
                    st.markdown(f"### {opcion}")
                    st.write(f"• Probabilidad Estimada: **{p_est:.1f}%**")
                    st.write(f"• Cuota Justa (Fair): **{c_fair}**")
                    st.write(f"• Cuota Casa de Apuestas: **{c_bookie}**")
                    
                    if ev > 0:
                        st.success(f"🔥 **¡VALUE BET DETECTADA!**\n\n• Valor Esperado (EV): **+{ev:.2f}%**\n• Stake recomendado (Kelly 1/4): **{kelly:.2f}% de tu Bankroll**")
                    else:
                        st.error(f"❌ **SIN VALOR DE APUESTA**\n\n• Valor Esperado (EV): **{ev:.2f}%**\n• No recomendable apostar.")

            mostrar_evaluacion(res_col1, f"Local: {stats_h['name']}", p1, c1_fair, cuota_bookie_1, ev1, kelly1)
            mostrar_evaluacion(res_col2, "Empate (X)", px, cx_fair, cuota_bookie_x, evx, kellyx)
            mostrar_evaluacion(res_col3, f"Visitante: {stats_a['name']}", p2, c2_fair, cuota_bookie_2, ev2, kelly2)

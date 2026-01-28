import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
from pytrends.request import TrendReq
import time
import random

st.set_page_config(
    page_title="Analizator Trendów Kryptowalut - Google Trends",
    page_icon="📊",
    layout="wide"
)

MAX_COUNTRIES_FOR_API = 3
API_REQUEST_DELAY = 3
MAX_KEYWORDS_FOR_API = 5

TRENDS_KEYWORDS = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "BNB": "Binance Coin", "SOL": "Solana",
    "XRP": "XRP", "ADA": "Cardano", "DOGE": "Dogecoin", "DOT": "Polkadot",
    "AVAX": "Avalanche", "LINK": "Chainlink", "MATIC": "Polygon", "LTC": "Litecoin",
    "SHIB": "Shiba Inu", "UNI": "Uniswap", "TRX": "TRON", "ATOM": "Cosmos",
    "ETC": "Ethereum Classic", "XLM": "Stellar", "VET": "VeChain", "ALGO": "Algorand",
    "ICP": "Internet Computer", "FIL": "Filecoin", "XTZ": "Tezos", "AAVE": "Aave",
    "EOS": "EOS", "AXS": "Axie Infinity", "THETA": "Theta Network", "SAND": "The Sandbox",
    "MANA": "Decentraland", "CHZ": "Chiliz", "NEAR": "NEAR Protocol", "QNT": "Quant",
    "GRT": "The Graph", "FTM": "Fantom", "APE": "ApeCoin", "FLOW": "Flow",
    "EGLD": "MultiversX", "IMX": "Immutable X", "RUNE": "THORChain", "MKR": "Maker",
    "SNX": "Synthetix", "CRV": "Curve DAO Token", "COMP": "Compound"
}

COINS = {
    "Bitcoin": "BTC", "Ethereum": "ETH", "Binance Coin": "BNB", "Solana": "SOL",
    "XRP": "XRP", "Cardano": "ADA", "Dogecoin": "DOGE", "Polkadot": "DOT",
    "Avalanche": "AVAX", "Chainlink": "LINK", "Polygon": "MATIC", "Litecoin": "LTC",
    "Shiba Inu": "SHIB", "Uniswap": "UNI", "TRON": "TRX", "Cosmos": "ATOM",
    "Ethereum Classic": "ETC", "Stellar": "XLM", "VeChain": "VET", "Algorand": "ALGO",
    "Internet Computer": "ICP", "Filecoin": "FIL", "Tezos": "XTZ", "Aave": "AAVE",
    "EOS": "EOS", "Axie Infinity": "AXS", "Theta Network": "THETA", "The Sandbox": "SAND",
    "Decentraland": "MANA", "Chiliz": "CHZ", "NEAR Protocol": "NEAR", "Quant": "QNT",
    "The Graph": "GRT", "Fantom": "FTM", "ApeCoin": "APE", "Flow": "FLOW",
    "MultiversX": "EGLD", "Immutable X": "IMX", "THORChain": "RUNE", "Maker": "MKR",
    "Synthetix": "SNX", "Curve DAO Token": "CRV", "Compound": "COMP"
}

COUNTRIES = {
    "USA": "US", "Niemcy": "DE", "Polska": "PL", "Wielka Brytania": "GB",
    "Francja": "FR", "Kanada": "CA", "Australia": "AU", "Brazylia": "BR",
    "Japonia": "JP", "Korea Południowa": "KR", "Indie": "IN", "Rosja": "RU",
    "Chiny": "CN", "Singapur": "SG", "Szwajcaria": "CH", "Holandia": "NL",
    "Włochy": "IT", "Hiszpania": "ES", "Szwecja": "SE", "Norwegia": "NO",
    "Dania": "DK", "Finlandia": "FI", "Turcja": "TR", "Ukraina": "UA",
    "Zjednoczone Emiraty Arabskie": "AE", "Arabia Saudyjska": "SA", "RPA": "ZA",
    "Meksyk": "MX", "Argentyna": "AR", "Chile": "CL", "Malezja": "MY",
    "Indonezja": "ID", "Wietnam": "VN", "Tajlandia": "TH", "Filipiny": "PH",
    "Izrael": "IL", "Czechy": "CZ", "Węgry": "HU", "Austria": "AT",
    "Belgia": "BE", "Portugalia": "PT", "Irlandia": "IE", "Nowa Zelandia": "NZ"
}

COLORS = {
    "price": "#1f77b4", "trend": "#ff7f0e", "forecast": "#2ca02c",
    "primary": "#1f77b4", "secondary": "#ff7f0e", "accent": "#2ca02c"
}

if 'trends_data' not in st.session_state:
    st.session_state.trends_data = None
if 'fact_table' not in st.session_state:
    st.session_state.fact_table = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'last_coin' not in st.session_state:
    st.session_state.last_coin = None
if 'last_countries' not in st.session_state:
    st.session_state.last_countries = []
if 'trends_cache_key' not in st.session_state:
    st.session_state.trends_cache_key = None
if 'data_source_mode' not in st.session_state:
    st.session_state.data_source_mode = "Auto (Google Trends z fallback)"
if 'selected_single_coin' not in st.session_state:
    st.session_state.selected_single_coin = "Bitcoin"
if 'api_status' not in st.session_state:
    st.session_state.api_status = "unknown"


@st.cache_data(ttl=3600, show_spinner=False)
def load_google_trends(keywords, countries, timeframe='today 5-y', force_fallback=False):
    if force_fallback:
        st.info("Używam symulowanych danych (wybrano ręcznie)")
        return create_fallback_trends_data(keywords, countries), "simulated"

    if len(countries) > MAX_COUNTRIES_FOR_API:
        st.info(f"Używam tylko {MAX_COUNTRIES_FOR_API} krajów dla Google Trends API")
        api_countries = countries[:MAX_COUNTRIES_FOR_API]
    else:
        api_countries = countries

    if len(keywords) > MAX_KEYWORDS_FOR_API:
        api_keywords = keywords[:MAX_KEYWORDS_FOR_API]
        st.info(f"Używam {MAX_KEYWORDS_FOR_API} głównych kryptowalut dla API")
    else:
        api_keywords = keywords

    trend_keywords = []
    for kw in api_keywords:
        if kw in TRENDS_KEYWORDS:
            trend_keywords.append(TRENDS_KEYWORDS[kw])
        else:
            trend_keywords.append(kw)

    st.info(f"Próba połączenia z Google Trends API ({len(api_countries)} kraje, {len(trend_keywords)} kryptowaluty)...")

    all_data = []
    api_success = False
    errors_count = 0
    successful_countries = 0

    try:
        pytrends = TrendReq(hl='en', tz=0, timeout=(10, 15))
        progress_bar = st.progress(0, text="Pobieranie danych z Google Trends...")
        total_countries = len(api_countries)

        for idx, country in enumerate(api_countries):
            try:
                progress = (idx + 1) / total_countries
                progress_bar.progress(progress, text=f"Pobieranie dla {country}... ({idx + 1}/{total_countries})")

                if errors_count >= 2:
                    st.warning("Zbyt wiele błędów API, przełączam na dane symulowane")
                    progress_bar.empty()
                    return create_fallback_trends_data(keywords, countries), "simulated_after_errors"

                pytrends.build_payload(kw_list=trend_keywords, cat=0, timeframe=timeframe, geo=country, gprop='')
                trends_df = pytrends.interest_over_time()

                if not trends_df.empty:
                    if 'isPartial' in trends_df.columns:
                        trends_df = trends_df[~trends_df['isPartial']]
                        trends_df = trends_df.drop(columns=['isPartial'], errors='ignore')
                    else:
                        trends_df = trends_df.drop(columns=['isPartial'], errors='ignore')

                    for trend_keyword in trend_keywords:
                        if trend_keyword in trends_df.columns:
                            original_keyword = None
                            for key, value in TRENDS_KEYWORDS.items():
                                if value == trend_keyword:
                                    original_keyword = key
                                    break
                            if not original_keyword:
                                original_keyword = trend_keyword
                            for date, value in trends_df[trend_keyword].items():
                                all_data.append({
                                    'date': date, 'keyword': original_keyword, 'trend_keyword': trend_keyword,
                                    'value': int(value), 'country': country, 'source': 'Google Trends API'
                                })
                    api_success = True
                    successful_countries += 1
                    time.sleep(API_REQUEST_DELAY + random.uniform(0.5, 1.5))

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Too Many Requests" in error_msg:
                    progress_bar.empty()
                    st.error(f"Osiągnięto limit zapytań Google Trends API dla kraju {country}")
                    st.warning("Przełączam na symulowane dane")
                    return create_fallback_trends_data(keywords, countries), "rate_limited"
                elif "400" in error_msg:
                    st.warning(f"Pomijam kraj {country} - błąd zapytania (400)")
                    errors_count += 1
                    continue
                elif "500" in error_msg or "503" in error_msg:
                    st.warning(f"Serwer Google Trends niedostępny dla {country}")
                    errors_count += 1
                    continue
                else:
                    st.warning(f"Błąd dla kraju {country}: {error_msg[:100]}...")
                    errors_count += 1
                    continue

        progress_bar.empty()
        if not all_data:
            st.warning("Brak danych z Google Trends, używam symulowanych")
            return create_fallback_trends_data(keywords, countries), "no_data"

        fact_table = pd.DataFrame(all_data)
        if not fact_table.empty:
            fact_table['date'] = pd.to_datetime(fact_table['date'])
            fact_table = fact_table.sort_values('date')

        if api_success:
            st.success(f"Pobrano dane z Google Trends API dla {successful_countries}/{total_countries} krajów")
            return fact_table, "success"
        else:
            st.warning("Brak udanych zapytań do API, używam symulowanych danych")
            return create_fallback_trends_data(keywords, countries), "all_failed"

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "Too Many Requests" in error_msg:
            st.error("Google Trends API: Limit zapytań przekroczony (Error 429)")
            st.info("Używam zaawansowanych symulowanych danych")
        else:
            st.error(f"Błąd połączenia z Google Trends: {error_msg[:150]}")
        return create_fallback_trends_data(keywords, countries), "connection_error"


def create_fallback_trends_data(keywords, countries):
    st.info("Generowanie zaawansowanych symulowanych danych...")
    dates = pd.date_range(end=datetime.now(), periods=60, freq='ME')
    all_data = []

    crypto_profiles = {
        "BTC": {"base": 80, "trend": 0.2, "season": 15, "volatility": 8, "spikes": True},
        "ETH": {"base": 65, "trend": 0.1, "season": 12, "volatility": 6, "spikes": True},
        "BNB": {"base": 50, "trend": 0.3, "season": 20, "volatility": 10, "spikes": False},
        "SOL": {"base": 70, "trend": 0.15, "season": 10, "volatility": 7, "spikes": True},
        "XRP": {"base": 60, "trend": 0.1, "season": 13, "volatility": 9, "spikes": False},
        "ADA": {"base": 75, "trend": 0.18, "season": 14, "volatility": 8, "spikes": True},
        "DOGE": {"base": 68, "trend": 0.12, "season": 16, "volatility": 12, "spikes": True},
        "DOT": {"base": 55, "trend": 0.08, "season": 11, "volatility": 5, "spikes": False},
        "AVAX": {"base": 62, "trend": 0.25, "season": 18, "volatility": 11, "spikes": True},
        "LINK": {"base": 58, "trend": 0.14, "season": 12, "volatility": 6, "spikes": False},
        "MATIC": {"base": 52, "trend": 0.22, "season": 15, "volatility": 9, "spikes": True},
        "LTC": {"base": 45, "trend": 0.05, "season": 10, "volatility": 4, "spikes": False},
        "SHIB": {"base": 40, "trend": 0.35, "season": 25, "volatility": 15, "spikes": True},
        "UNI": {"base": 55, "trend": 0.2, "season": 14, "volatility": 8, "spikes": False},
        "TRX": {"base": 48, "trend": 0.08, "season": 12, "volatility": 6, "spikes": False},
        "ATOM": {"base": 58, "trend": 0.18, "season": 13, "volatility": 7, "spikes": True},
        "ETC": {"base": 42, "trend": 0.06, "season": 10, "volatility": 5, "spikes": False},
        "XLM": {"base": 52, "trend": 0.1, "season": 11, "volatility": 6, "spikes": False},
        "VET": {"base": 46, "trend": 0.15, "season": 14, "volatility": 8, "spikes": False},
        "ALGO": {"base": 53, "trend": 0.12, "season": 12, "volatility": 7, "spikes": False},
        "ICP": {"base": 35, "trend": 0.25, "season": 18, "volatility": 12, "spikes": True},
        "FIL": {"base": 47, "trend": 0.14, "season": 13, "volatility": 8, "spikes": False},
        "XTZ": {"base": 44, "trend": 0.09, "season": 11, "volatility": 6, "spikes": False},
        "AAVE": {"base": 56, "trend": 0.16, "season": 14, "volatility": 9, "spikes": False},
        "EOS": {"base": 41, "trend": 0.07, "season": 10, "volatility": 5, "spikes": False},
        "AXS": {"base": 38, "trend": 0.3, "season": 20, "volatility": 14, "spikes": True},
        "THETA": {"base": 50, "trend": 0.22, "season": 16, "volatility": 10, "spikes": True},
        "SAND": {"base": 45, "trend": 0.28, "season": 18, "volatility": 13, "spikes": True},
        "MANA": {"base": 47, "trend": 0.26, "season": 17, "volatility": 12, "spikes": True},
        "CHZ": {"base": 43, "trend": 0.2, "season": 15, "volatility": 11, "spikes": True},
        "NEAR": {"base": 51, "trend": 0.24, "season": 14, "volatility": 9, "spikes": True},
        "QNT": {"base": 49, "trend": 0.19, "season": 13, "volatility": 8, "spikes": False},
        "GRT": {"base": 44, "trend": 0.17, "season": 12, "volatility": 7, "spikes": False},
        "FTM": {"base": 47, "trend": 0.21, "season": 15, "volatility": 10, "spikes": True},
        "APE": {"base": 42, "trend": 0.27, "season": 16, "volatility": 12, "spikes": True},
        "FLOW": {"base": 46, "trend": 0.13, "season": 11, "volatility": 6, "spikes": False},
        "EGLD": {"base": 53, "trend": 0.18, "season": 14, "volatility": 9, "spikes": False},
        "IMX": {"base": 45, "trend": 0.23, "season": 15, "volatility": 11, "spikes": True},
        "RUNE": {"base": 48, "trend": 0.16, "season": 13, "volatility": 8, "spikes": False},
        "MKR": {"base": 54, "trend": 0.14, "season": 12, "volatility": 7, "spikes": False},
        "SNX": {"base": 47, "trend": 0.15, "season": 13, "volatility": 8, "spikes": False},
        "CRV": {"base": 43, "trend": 0.12, "season": 11, "volatility": 6, "spikes": False},
        "COMP": {"base": 52, "trend": 0.17, "season": 14, "volatility": 9, "spikes": False}
    }

    country_profiles = {
        "US": {"multiplier": 1.0, "trend_boost": 0.1, "season_factor": 1.0},
        "DE": {"multiplier": 0.8, "trend_boost": 0.05, "season_factor": 0.9},
        "PL": {"multiplier": 0.6, "trend_boost": 0.15, "season_factor": 1.2},
        "GB": {"multiplier": 0.9, "trend_boost": 0.08, "season_factor": 0.95},
        "FR": {"multiplier": 0.7, "trend_boost": 0.06, "season_factor": 0.85},
        "CA": {"multiplier": 0.85, "trend_boost": 0.09, "season_factor": 1.1},
        "AU": {"multiplier": 0.75, "trend_boost": 0.07, "season_factor": 1.05},
        "BR": {"multiplier": 0.8, "trend_boost": 0.12, "season_factor": 1.3},
        "JP": {"multiplier": 0.7, "trend_boost": 0.08, "season_factor": 0.9},
        "KR": {"multiplier": 0.9, "trend_boost": 0.2, "season_factor": 1.4},
        "IN": {"multiplier": 0.85, "trend_boost": 0.18, "season_factor": 1.5},
        "RU": {"multiplier": 0.5, "trend_boost": 0.04, "season_factor": 0.7},
        "CN": {"multiplier": 0.4, "trend_boost": 0.05, "season_factor": 0.6},
        "SG": {"multiplier": 0.8, "trend_boost": 0.1, "season_factor": 1.1},
        "CH": {"multiplier": 0.75, "trend_boost": 0.07, "season_factor": 0.95},
        "NL": {"multiplier": 0.8, "trend_boost": 0.09, "season_factor": 1.0},
        "IT": {"multiplier": 0.65, "trend_boost": 0.06, "season_factor": 0.85},
        "ES": {"multiplier": 0.7, "trend_boost": 0.08, "season_factor": 0.9},
        "SE": {"multiplier": 0.8, "trend_boost": 0.09, "season_factor": 1.0},
        "NO": {"multiplier": 0.75, "trend_boost": 0.07, "season_factor": 0.95},
        "DK": {"multiplier": 0.75, "trend_boost": 0.07, "season_factor": 0.95},
        "FI": {"multiplier": 0.7, "trend_boost": 0.06, "season_factor": 0.9},
        "TR": {"multiplier": 0.85, "trend_boost": 0.15, "season_factor": 1.3},
        "UA": {"multiplier": 0.6, "trend_boost": 0.12, "season_factor": 1.1},
        "AE": {"multiplier": 0.8, "trend_boost": 0.1, "season_factor": 1.1},
        "SA": {"multiplier": 0.7, "trend_boost": 0.09, "season_factor": 1.0},
        "ZA": {"multiplier": 0.7, "trend_boost": 0.1, "season_factor": 1.1},
        "MX": {"multiplier": 0.75, "trend_boost": 0.11, "season_factor": 1.2},
        "AR": {"multiplier": 0.8, "trend_boost": 0.14, "season_factor": 1.4},
        "CL": {"multiplier": 0.7, "trend_boost": 0.09, "season_factor": 1.0},
        "MY": {"multiplier": 0.75, "trend_boost": 0.12, "season_factor": 1.2},
        "ID": {"multiplier": 0.8, "trend_boost": 0.16, "season_factor": 1.5},
        "VN": {"multiplier": 0.78, "trend_boost": 0.18, "season_factor": 1.6},
        "TH": {"multiplier": 0.72, "trend_boost": 0.13, "season_factor": 1.3},
        "PH": {"multiplier": 0.79, "trend_boost": 0.17, "season_factor": 1.5},
        "IL": {"multiplier": 0.82, "trend_boost": 0.11, "season_factor": 1.1},
        "CZ": {"multiplier": 0.68, "trend_boost": 0.08, "season_factor": 0.9},
        "HU": {"multiplier": 0.65, "trend_boost": 0.09, "season_factor": 1.0},
        "AT": {"multiplier": 0.73, "trend_boost": 0.07, "season_factor": 0.95},
        "BE": {"multiplier": 0.71, "trend_boost": 0.08, "season_factor": 1.0},
        "PT": {"multiplier": 0.69, "trend_boost": 0.09, "season_factor": 1.0},
        "IE": {"multiplier": 0.76, "trend_boost": 0.1, "season_factor": 1.1},
        "NZ": {"multiplier": 0.74, "trend_boost": 0.08, "season_factor": 1.0}
    }

    progress_bar = st.progress(0, text="Generowanie zaawansowanych symulowanych danych...")
    total_combinations = len(countries) * len(keywords)
    current = 0

    for country in countries:
        country_profile = country_profiles.get(country, {
            "multiplier": 0.7,
            "trend_boost": 0.05,
            "season_factor": 1.0
        })

        for keyword in keywords:
            current += 1
            progress = current / total_combinations

            if current % 5 == 0 or current == total_combinations:
                progress_bar.progress(progress,
                                      text=f"Generowanie: {keyword} - {country}... ({current}/{total_combinations})")

            profile = crypto_profiles.get(keyword, {
                "base": 60,
                "trend": 0.1,
                "season": 15,
                "volatility": 5,
                "spikes": False
            })

            time_idx = np.arange(len(dates))
            adjusted_trend = profile["trend"] + country_profile["trend_boost"]
            trend_line = (profile["base"] * country_profile["multiplier"] +
                          adjusted_trend * time_idx)

            seasonal = (profile["season"] * country_profile["season_factor"] *
                        np.sin(2 * np.pi * time_idx / 12 +
                               random.uniform(0, np.pi)))

            noise = np.random.randn(len(dates)) * profile["volatility"]

            events = np.zeros(len(dates))
            if profile["spikes"] and random.random() > 0.5:
                for _ in range(random.randint(1, 3)):
                    event_idx = random.randint(10, 50)
                    event_duration = random.randint(1, 3)
                    event_strength = random.uniform(20, 40)
                    for i in range(event_duration):
                        if event_idx + i < len(dates):
                            events[event_idx + i] += event_strength / (i + 1)

            values = trend_line + seasonal + noise + events
            values = np.clip(values, 0, 100).round().astype(int)

            for i, date in enumerate(dates):
                all_data.append({
                    'date': date,
                    'keyword': keyword,
                    'trend_keyword': TRENDS_KEYWORDS.get(keyword, keyword),
                    'value': values[i],
                    'country': country,
                    'source': 'Symulowane dane (zaawansowane)'
                })

    progress_bar.empty()
    fact_table = pd.DataFrame(all_data)
    fact_table['date'] = pd.to_datetime(fact_table['date'])
    fact_table = fact_table.sort_values('date')

    st.success(f"Wygenerowano {len(fact_table):,} rekordów zaawansowanych symulowanych danych")
    return fact_table


def create_correlation_matrix(fact_table, selected_coins):
    if fact_table is None or fact_table.empty:
        return None

    try:
        selected_trend_keywords = []
        for coin in selected_coins:
            if coin in TRENDS_KEYWORDS:
                selected_trend_keywords.append(TRENDS_KEYWORDS[coin])

        if len(selected_trend_keywords) < 2:
            return None

        pivot_data = pd.pivot_table(
            fact_table[fact_table['trend_keyword'].isin(selected_trend_keywords)],
            index='date',
            columns='trend_keyword',
            values='value',
            aggfunc='mean'
        ).fillna(0)

        available_columns = [col for col in selected_trend_keywords if col in pivot_data.columns]

        if len(available_columns) < 2:
            return None

        correlation_matrix = pivot_data[available_columns].corr()

        coin_names_mapping = {}
        for coin_symbol in selected_coins:
            if coin_symbol in TRENDS_KEYWORDS:
                trend_name = TRENDS_KEYWORDS[coin_symbol]
                for name, symbol in COINS.items():
                    if symbol == coin_symbol:
                        coin_names_mapping[trend_name] = name
                        break

        correlation_matrix = correlation_matrix.rename(index=coin_names_mapping, columns=coin_names_mapping)
        return correlation_matrix

    except Exception as e:
        st.warning(f"Błąd tworzenia macierzy korelacji: {str(e)[:100]}")
        return None


def analyze_seasonality(fact_table, keyword):
    if fact_table is None or fact_table.empty:
        return None

    try:
        trend_keyword = TRENDS_KEYWORDS.get(keyword, keyword)
        keyword_data = fact_table[fact_table['trend_keyword'] == trend_keyword].copy()

        if keyword_data.empty:
            return None

        keyword_data['month'] = keyword_data['date'].dt.month
        monthly_avg = keyword_data.groupby('month')['value'].mean().reset_index()
        return monthly_avg

    except Exception as e:
        st.warning(f"Błąd analizy sezonowości: {str(e)[:100]}")
        return None


def aggregate_trends_time(fact_table, freq="M"):
    if fact_table is None or fact_table.empty:
        return None

    try:
        agg_df = (fact_table.set_index("date")
                  .groupby(["trend_keyword", "country"])
                  .resample(freq)["value"]
                  .mean()
                  .reset_index())
        return agg_df
    except Exception as e:
        st.warning(f"Błąd agregacji czasowej: {str(e)[:100]}")
        return None


def trend_direction_analysis(fact_table, keyword):
    if fact_table is None or fact_table.empty:
        return None

    try:
        trend_keyword = TRENDS_KEYWORDS.get(keyword, keyword)
        keyword_data = fact_table[fact_table['trend_keyword'] == trend_keyword].copy()

        if keyword_data.empty:
            return None

        keyword_data = keyword_data.sort_values('date')
        series = keyword_data.groupby('date')['value'].mean()

        if len(series) < 2:
            return None

        x = np.arange(len(series))
        y = series.values
        coef = np.polyfit(x, y, 1)[0]

        if coef > 0.5:
            return "Silnie wzrostowy", coef
        elif coef > 0.1:
            return "Wzrostowy", coef
        elif coef < -0.5:
            return "Silnie spadkowy", coef
        elif coef < -0.1:
            return "Spadkowy", coef
        else:
            return "Stabilny", coef

    except Exception as e:
        st.warning(f"Błąd analizy trendu: {str(e)[:100]}")
        return None


def create_heatmap_data(fact_table, keyword=None):
    if fact_table is None or fact_table.empty:
        return None

    try:
        heat_df = fact_table.copy()
        if keyword:
            trend_keyword = TRENDS_KEYWORDS.get(keyword, keyword)
            heat_df = heat_df[heat_df['trend_keyword'] == trend_keyword].copy()
            if heat_df.empty:
                return None

        heat_df['year'] = heat_df['date'].dt.year
        pivot_heat = heat_df.pivot_table(
            index='country',
            columns='year',
            values='value',
            aggfunc='mean'
        ).fillna(0)
        return pivot_heat

    except Exception as e:
        st.warning(f"Błąd tworzenia mapy cieplnej: {str(e)[:100]}")
        return None


def calculate_data_statistics(fact_table, keyword=None):
    if fact_table is None or fact_table.empty:
        return None

    try:
        stats_df = fact_table.copy()
        if keyword:
            trend_keyword = TRENDS_KEYWORDS.get(keyword, keyword)
            stats_df = stats_df[stats_df['trend_keyword'] == trend_keyword].copy()
            if stats_df.empty:
                return None

        stats = {
            'total_records': len(stats_df),
            'date_range': f"{stats_df['date'].min().strftime('%Y-%m-%d')} - {stats_df['date'].max().strftime('%Y-%m-%d')}",
            'unique_keywords': stats_df['keyword'].nunique(),
            'unique_countries': stats_df['country'].nunique(),
            'avg_value': round(stats_df['value'].mean(), 2),
            'max_value': stats_df['value'].max(),
            'min_value': stats_df['value'].min(),
            'data_source': stats_df['source'].iloc[0] if not stats_df.empty else 'Brak danych',
            'months_covered': stats_df['date'].dt.to_period('M').nunique()
        }
        return stats

    except Exception as e:
        st.warning(f"Błąd obliczania statystyk: {str(e)[:100]}")
        return None


def calculate_price_changes(price_df):
    if price_df is None or price_df.empty or 'Close' not in price_df.columns:
        return None

    try:
        latest_price = price_df['Close'].iloc[-1]
        price_df = price_df.copy()
        price_df['date'] = price_df.index

        def pct_change_years(years):
            past_date = price_df['date'].max() - pd.DateOffset(years=years)
            past_prices = price_df[price_df['date'] <= past_date]
            if past_prices.empty:
                return None
            past_price = past_prices['Close'].iloc[-1]
            return ((latest_price - past_price) / past_price) * 100

        return {
            "current": latest_price,
            "1y": pct_change_years(1),
            "5y": pct_change_years(5),
            "10y": pct_change_years(10)
        }

    except Exception as e:
        st.warning(f"Błąd obliczania zmian cen: {str(e)[:100]}")
        return None


def predict_future_price(price_df, years_ahead=5):
    if price_df is None or price_df.empty or len(price_df) < 30:
        return None, None

    try:
        df = price_df.reset_index()
        df['t'] = np.arange(len(df))
        X = df[['t']]
        y = df['Close']
        model = LinearRegression()
        model.fit(X, y)

        future_days = years_ahead * 365
        future_t = np.array([[df['t'].iloc[-1] + future_days]])
        predicted_price = model.predict(future_t)[0]
        daily_trend = model.coef_[0]

        return predicted_price, daily_trend

    except Exception as e:
        st.warning(f"Błąd predykcji ceny: {str(e)[:100]}")
        return None, None


def create_forecast_plot(price_df, forecast_years=5):
    if price_df is None or price_df.empty or len(price_df) < 30:
        return None

    try:
        df = price_df.reset_index()
        df['t'] = np.arange(len(df))
        X = df[['t']]
        y = df['Close']
        model = LinearRegression()
        model.fit(X, y)

        last_date = price_df.index[-1]
        future_dates = pd.date_range(start=last_date, periods=365 * forecast_years + 1, freq='D')[1:]
        future_t = np.arange(len(df), len(df) + len(future_dates)).reshape(-1, 1)
        future_prices = model.predict(future_t)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=price_df.index,
            y=price_df['Close'],
            mode='lines',
            name='Cena historyczna',
            line=dict(color=COLORS['price'], width=2),
            hovertemplate="<b>Data:</b> %{x|%Y-%m-%d}<br><b>Cena:</b> $%{y:,.2f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_prices,
            mode='lines',
            name=f'Prognoza {forecast_years}-letnia (stabilne warunki)',
            line=dict(color=COLORS['forecast'], width=2, dash='dash'),
            hovertemplate="<b>Data:</b> %{x|%Y-%m-%d}<br><b>Prognoza:</b> $%{y:,.2f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=[last_date],
            y=[price_df['Close'].iloc[-1]],
            mode='markers',
            name='Punkt startowy prognozy',
            marker=dict(color=COLORS['accent'], size=10),
            hovertemplate="<b>Start prognozy</b><br><b>Cena:</b> $%{y:,.2f}<extra></extra>"
        ))

        fig.update_layout(
            title=f'Prognoza ceny (model regresji liniowej) - horyzont {forecast_years} lat',
            xaxis_title='Data',
            yaxis_title='Cena (USD)',
            height=500,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='rgba(240, 240, 240, 0.8)'
        )
        return fig

    except Exception as e:
        st.warning(f"Błąd tworzenia wykresu prognozy: {str(e)[:100]}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_price_data(symbol):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10 * 365)

        st.info(f"Pobieranie danych cenowych dla {symbol} z Yahoo Finance...")

        ticker = yf.Ticker(f"{symbol}-USD")
        df = ticker.history(start=start_date, end=end_date, interval="1d")

        if df.empty or len(df) < 100:
            st.warning(f"Brak lub ograniczone dane dla {symbol}, generuję symulowane...")

            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            base_price = 30000 if symbol == "BTC" else 2000 if symbol == "ETH" else 100

            n_days = len(dates)
            daily_growth = 0.0003
            trend = np.exp(daily_growth * np.arange(n_days))
            seasonal = 0.15 * np.sin(2 * np.pi * np.arange(n_days) / 30)
            noise = np.random.randn(n_days) * 0.03
            events = np.zeros(n_days)

            for _ in range(3):
                idx = np.random.randint(100, n_days - 100)
                events[idx:idx + 30] += np.random.uniform(0.2, 0.4)

            prices = base_price * trend * np.exp(seasonal + noise + events)
            prices = np.maximum(prices, base_price * 0.3)

            df = pd.DataFrame({
                "Close": prices,
                "Open": prices * (1 + np.random.randn(n_days) * 0.01),
                "High": prices * (1 + np.random.uniform(0.01, 0.05, n_days)),
                "Low": prices * (1 - np.random.uniform(0.01, 0.05, n_days)),
                "Volume": np.random.lognormal(14, 1.2, n_days)
            }, index=dates)

        else:
            if 'Close' not in df.columns and len(df.columns) > 0:
                df = df.rename(columns={df.columns[0]: 'Close'})

        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])
        df['Year'] = df.index.year
        yearly = df.groupby('Year')['Close'].mean().reset_index().round(2)

        st.success(f"Załadowano {len(df):,} dni danych cenowych dla {symbol}")
        return df, yearly

    except Exception as e:
        st.error(f"Błąd ładowania danych cenowych: {str(e)[:200]}")

        dates = pd.date_range(end=datetime.now(), periods=365 * 5, freq='D')
        base_price = 30000 if symbol == "BTC" else 2000 if symbol == "ETH" else 100
        prices = base_price * (1 + 0.0003 * np.arange(len(dates)))

        df = pd.DataFrame({"Close": prices}, index=dates)
        df['Year'] = df.index.year
        yearly = df.groupby('Year')['Close'].mean().reset_index().round(2)

        return df, yearly


st.title("Analizator Trendów Kryptowalut - Google Trends")
st.markdown("""
### Projekt: Analiza trendów w wyszukiwaniu tematów w Google Trends  
**Przedmiot:** Big Data i Hurtownie Danych  
**Temat:** Analiza popularności kryptowalut na podstawie Google Trends

**Opis projektu:** Implementacja kompletnego pipeline'u ETL do pozyskiwania, przetwarzania i analizy danych z Google Trends 
z wykorzystaniem metodologii hurtowni danych i analizy Big Data.
""")

st.sidebar.header("Ustawienia")
if st.sidebar.button("Odśwież wszystkie dane", use_container_width=True, type="primary"):
    st.cache_data.clear()
    st.session_state.trends_data = None
    st.session_state.fact_table = None
    st.session_state.data_loaded = False
    st.session_state.last_coin = None
    st.session_state.last_countries = []
    st.session_state.trends_cache_key = None
    st.session_state.selected_single_coin = "Bitcoin"
    st.session_state.api_status = "unknown"
    st.rerun()

st.sidebar.subheader("Tryb danych")
data_source_mode = st.sidebar.radio(
    "Źródło danych", [
        "Auto (Google Trends z fallback)",
        "Tylko Google Trends (może failować)",
        "Tylko symulowane dane (zalecane dla prezentacji)"
    ],
    index=0,
    help="Google Trends API ma ograniczenia rate limiting. W trybie auto aplikacja automatycznie przełącza się na symulowane dane w przypadku błędów."
)

st.sidebar.subheader("Kryptowaluty do analizy")
coin_names = list(COINS.keys())
selected_coin_names = st.sidebar.multiselect(
    "Wybierz kryptowaluty",
    coin_names,
    default=["Bitcoin", "Ethereum", "Solana", "Cardano", "Dogecoin", "Polkadot", "Avalanche"],
    key="coin_selector",
    help="Wybierz przynajmniej jedną kryptowalutę"
)

selected_coins = [COINS[coin] for coin in selected_coin_names]

st.sidebar.subheader("Kraje do analizy")
st.sidebar.caption(f"Maksymalnie {MAX_COUNTRIES_FOR_API} kraje dla Google Trends API")

selected_countries = []
country_cols = st.sidebar.columns(2)
country_items = list(COUNTRIES.items())
mid_point = len(country_items) // 2

for idx, (country_name, country_code) in enumerate(country_items):
    col_idx = 0 if idx < mid_point else 1
    default_value = country_code in ["US", "DE", "PL"]
    if country_cols[col_idx].checkbox(
            country_name,
            value=default_value,
            key=f"country_{country_code}"
    ):
        selected_countries.append(country_code)

if not selected_countries:
    selected_countries = ["US", "DE", "PL"]

st.sidebar.subheader("Opcje analizy")
show_correlation = st.sidebar.checkbox("Macierz korelacji", value=True)
show_seasonality = st.sidebar.checkbox("Analiza sezonowości", value=True)
show_comparison = st.sidebar.checkbox("Porównanie kryptowalut", value=True)
show_aggregation = st.sidebar.checkbox("Agregacje czasowe", value=True)
show_heatmap = st.sidebar.checkbox("Mapa cieplna", value=True)
show_price_data = st.sidebar.checkbox("Dane cenowe (rozszerzenie)", value=True)

st.sidebar.header("Eksport danych")
if st.sidebar.button("Pobierz dane trendów (CSV)", use_container_width=True):
    if st.session_state.fact_table is not None:
        csv = st.session_state.fact_table.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            "Kliknij aby pobrać",
            csv,
            f"google_trends_kryptowaluty_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            key="download_csv"
        )

st.sidebar.markdown("---")
if 'api_status' in st.session_state:
    status = st.session_state.api_status
    if status == "success":
        st.sidebar.success("Google Trends API: Aktywny")
    elif status == "simulated":
        st.sidebar.info("Używam symulowanych danych")
    elif status == "rate_limited":
        st.sidebar.warning("API Limit, dane symulowane")
    elif status == "connection_error":
        st.sidebar.error("Błąd połączenia, dane symulowane")
    else:
        st.sidebar.info("Stan API: Nieznany")

if not selected_coins:
    st.warning("Wybierz przynajmniej jedną kryptowalutę do analizy")
else:
    current_cache_key = f"{'_'.join(sorted(selected_coins))}_{'_'.join(sorted(selected_countries))}_{data_source_mode}"

    if (st.session_state.fact_table is None or
            st.session_state.trends_cache_key != current_cache_key or
            not st.session_state.data_loaded or
            st.session_state.data_source_mode != data_source_mode):

        with st.spinner("Uruchamianie ETL Pipeline..."):
            if data_source_mode == "Tylko symulowane dane (zalecane dla prezentacji)":
                fact_table = create_fallback_trends_data(selected_coins, selected_countries)
                api_status = "simulated"
            elif data_source_mode == "Tylko Google Trends (może failować)":
                fact_table, api_status = load_google_trends(
                    selected_coins,
                    selected_countries,
                    force_fallback=False
                )
            else:
                fact_table, api_status = load_google_trends(
                    selected_coins,
                    selected_countries,
                    force_fallback=False
                )

            if fact_table is not None and not fact_table.empty:
                st.session_state.fact_table = fact_table
                st.session_state.trends_cache_key = current_cache_key
                st.session_state.data_loaded = True
                st.session_state.data_source_mode = data_source_mode
                st.session_state.api_status = api_status
                st.success("Dane załadowane pomyślnie!")
            else:
                st.error("Nie udało się załadować danych")

    if st.session_state.fact_table is not None and not st.session_state.fact_table.empty:
        fact_table = st.session_state.fact_table

        if st.session_state.selected_single_coin not in selected_coin_names:
            st.session_state.selected_single_coin = selected_coin_names[0]

        st.session_state.selected_single_coin = st.selectbox(
            "Wybierz kryptowalutę do szczegółowej analizy:",
            selected_coin_names,
            index=selected_coin_names.index(st.session_state.selected_single_coin),
            key="single_coin_selector"
        )

        selected_single_coin_symbol = COINS[st.session_state.selected_single_coin]
        fact_table_filtered = fact_table[fact_table['keyword'].isin(selected_coins)]

        st.header("Trendy wyszukiwań – Google Trends")

        data_stats = calculate_data_statistics(fact_table_filtered, selected_single_coin_symbol)

        if data_stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                source_icon = "API" if "Google Trends API" in data_stats['data_source'] else "SYM"
                st.metric("Źródło danych", f"{source_icon}")
            with col2:
                st.metric("Liczba rekordów", f"{data_stats['total_records']:,}")
            with col3:
                st.metric("Okres analizy", f"{data_stats['months_covered']} miesięcy")
            with col4:
                st.metric("Średnie zainteresowanie", f"{data_stats['avg_value']}")

        st.info(f"""
        **Analizowane kryptowaluty:** {', '.join(selected_coin_names)}  
        **Wybrana kryptowaluta do szczegółowej analizy:** **{st.session_state.selected_single_coin}**  
        **Kraje:** {', '.join([name for name, code in COUNTRIES.items() if code in selected_countries])}
        """)

        if selected_single_coin_symbol:
            trend_result = trend_direction_analysis(fact_table_filtered, selected_single_coin_symbol)
            if trend_result:
                direction, coef = trend_result
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Kierunek trendu", direction, delta=f"{coef:.4f}")
                with col2:
                    st.metric("Wybrana kryptowaluta", st.session_state.selected_single_coin)

        if show_comparison and len(selected_coin_names) > 1:
            st.subheader("Porównanie trendów wyszukiwań kryptowalut")

            comparison_df = fact_table_filtered.copy()
            comparison_df = (comparison_df.groupby(['date', 'trend_keyword'], as_index=False)['value'].mean())
            comparison_df['value_norm'] = comparison_df.groupby('trend_keyword')['value'].transform(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )
            comparison_df['ma_3'] = comparison_df.groupby('trend_keyword')['value_norm'].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )

            comparison_df['Kryptowaluta'] = comparison_df['trend_keyword'].map({v: k for k, v in COINS.items()})
            comparison_df['Kryptowaluta'] = comparison_df.apply(
                lambda row: row['trend_keyword'] if pd.isna(row['Kryptowaluta']) else row['Kryptowaluta'],
                axis=1
            )

            fig_comp = go.Figure()
            for coin in comparison_df['Kryptowaluta'].unique():
                df_coin = comparison_df[comparison_df['Kryptowaluta'] == coin]
                fig_comp.add_trace(go.Scatter(
                    x=df_coin['date'],
                    y=df_coin['value_norm'],
                    mode='lines',
                    name=f"{coin} (surowe)",
                    opacity=0.3,
                    hovertemplate="<b>%{x|%Y-%m}</b><br>Zainteresowanie: %{y:.2f}<extra></extra>"
                ))
                fig_comp.add_trace(go.Scatter(
                    x=df_coin['date'],
                    y=df_coin['ma_3'],
                    mode='lines',
                    name=f"{coin} (trend)",
                    line=dict(width=3),
                    hovertemplate="<b>%{x|%Y-%m}</b><br>Trend: %{y:.2f}<extra></extra>"
                ))

            fig_comp.update_layout(
                title="Porównanie trendów wyszukiwań (normalizacja + trend)",
                yaxis_title="Znormalizowany poziom zainteresowania (0–1)",
                xaxis_title="Data",
                height=550,
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        if show_aggregation:
            st.subheader("Agregacje czasowe – analiza Big Data")
            aggregation_freq = st.selectbox(
                "Okres agregacji",
                ["Miesięczna (M)", "Kwartalna (Q)", "Roczna (Y)"],
                key="agg_freq"
            )

            freq_map = {"Miesięczna (M)": "M", "Kwartalna (Q)": "Q", "Roczna (Y)": "Y"}
            selected_freq = freq_map[aggregation_freq]
            agg_df = aggregate_trends_time(fact_table_filtered, selected_freq)
            freq_label = {"M": "Miesięczna", "Q": "Kwartalna", "Y": "Roczna"}

            if agg_df is not None:
                col1, col2, col3 = st.columns(3)
                col1.metric("Rekordy surowe", f"{len(fact_table_filtered):,}")
                col2.metric("Rekordy po agregacji", f"{len(agg_df):,}")
                col3.metric("Stopień kompresji", f"{len(fact_table_filtered) / len(agg_df):.1f}:1")

                agg_df['Kryptowaluta'] = agg_df['trend_keyword'].map({v: k for k, v in COINS.items()})
                agg_df['Kryptowaluta'] = agg_df.apply(
                    lambda row: row['trend_keyword'] if pd.isna(row['Kryptowaluta']) else row['Kryptowaluta'],
                    axis=1
                )

                fig_agg = px.line(
                    agg_df,
                    x='date',
                    y='value',
                    color='Kryptowaluta',
                    markers=True,
                    title=f"Agregacja {freq_label[selected_freq].lower()} trendów wyszukiwań",
                    labels={'value': 'Średnie zainteresowanie (0–100)', 'date': 'Data'}
                )

                fig_agg.update_traces(line=dict(width=3))
                fig_agg.update_layout(
                    height=650,
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
                    margin=dict(l=40, r=40, t=80, b=40)
                )
                st.plotly_chart(fig_agg, use_container_width=True)

                if not fact_table_filtered.empty and agg_df is not None:
                    raw_std = fact_table_filtered.groupby('trend_keyword')['value'].std().mean()
                    agg_std = agg_df.groupby('trend_keyword')['value'].std().mean()

                    c1, c2 = st.columns(2)
                    c1.metric("Zmienność (dane surowe)", f"{raw_std:.2f}")
                    c2.metric("Zmienność (po agregacji)", f"{agg_std:.2f}")

                    if raw_std > 0:
                        reduction = ((raw_std - agg_std) / raw_std) * 100
                        st.info(f"Redukcja zmienności: {reduction:.1f}%")

        if show_correlation and len(selected_coins) > 1:
            st.subheader("Macierz korelacji trendów wyszukiwań")
            correlation_matrix = create_correlation_matrix(fact_table_filtered, selected_coins)

            if correlation_matrix is not None:
                fig_corr = px.imshow(
                    correlation_matrix,
                    text_auto='.2f',
                    aspect="auto",
                    color_continuous_scale='RdBu',
                    title='Korelacja między trendami wyszukiwań kryptowalut',
                    labels=dict(color="Współczynnik korelacji"),
                    zmin=-1,
                    zmax=1
                )

                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)

                correlation_values = correlation_matrix.values
                mask = np.triu(np.ones_like(correlation_values, dtype=bool), k=1)
                upper_values = correlation_values[mask]

                if len(upper_values) > 0:
                    max_corr = upper_values.max()
                    min_corr = upper_values.min()
                    avg_corr = upper_values.mean()

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Najwyższa korelacja", f"{max_corr:.3f}")
                    with col2:
                        st.metric("Najniższa korelacja", f"{min_corr:.3f}")
                    with col3:
                        st.metric("Średnia korelacja", f"{avg_corr:.3f}")

        if show_seasonality:
            st.subheader("Analiza sezonowości wyszukiwań")
            seasonality_data = analyze_seasonality(fact_table_filtered, selected_single_coin_symbol)

            if seasonality_data is not None:
                month_names = [
                    'Styczeń', 'Luty', 'Marzec', 'Kwiecień', 'Maj', 'Czerwiec',
                    'Lipiec', 'Sierpień', 'Wrzesień', 'Październik', 'Listopad', 'Grudzień'
                ]

                seasonality_data['month_name'] = [month_names[m - 1] for m in seasonality_data['month']]

                fig_season = px.bar(
                    seasonality_data,
                    x='month_name',
                    y='value',
                    title=f'Sezonowość wyszukiwań - {st.session_state.selected_single_coin}',
                    labels={'value': 'Średnie zainteresowanie (0-100)', 'month_name': 'Miesiąc'},
                    color='value',
                    color_continuous_scale='Viridis'
                )

                fig_season.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_season, use_container_width=True)

                if not seasonality_data.empty:
                    max_month_idx = seasonality_data['value'].idxmax()
                    min_month_idx = seasonality_data['value'].idxmin()

                    max_month = seasonality_data.loc[max_month_idx, 'month_name']
                    max_value = seasonality_data.loc[max_month_idx, 'value']

                    min_month = seasonality_data.loc[min_month_idx, 'month_name']
                    min_value = seasonality_data.loc[min_month_idx, 'value']

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Najwyższe zainteresowanie", max_month, f"{max_value:.1f}")
                    with col2:
                        st.metric("Najniższe zainteresowanie", min_month, f"{min_value:.1f}")

        if show_heatmap:
            st.subheader("Mapa cieplna zainteresowania (rok × kraj)")
            heatmap_data = create_heatmap_data(fact_table_filtered, selected_single_coin_symbol)

            if heatmap_data is not None:
                heatmap_data['Kraj'] = heatmap_data.index.map({code: name for name, code in COUNTRIES.items()})

                fig_heat = px.imshow(
                    heatmap_data.drop(columns=['Kraj']),
                    labels=dict(x="Rok", y="Kraj", color="Zainteresowanie"),
                    title=f'Zainteresowanie Google Trends – mapa cieplna (średnie roczne) - {st.session_state.selected_single_coin}',
                    color_continuous_scale='Viridis',
                    aspect="auto"
                )

                fig_heat.update_layout(height=400)
                st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("Analiza regionalna zainteresowania")
        if not fact_table_filtered.empty:
            coin_data = fact_table_filtered[
                fact_table_filtered['trend_keyword'] == TRENDS_KEYWORDS.get(selected_single_coin_symbol,
                                                                            selected_single_coin_symbol)
                ]

            if not coin_data.empty:
                country_overall = coin_data.groupby('country').agg({
                    'value': ['mean', 'max', 'std', 'count']
                }).round(1)

                country_overall.columns = ['Średnie', 'Maksimum', 'Odchylenie', 'Liczba pomiarów']
                country_overall = country_overall.sort_values('Średnie', ascending=False)
                country_overall['Kraj'] = country_overall.index.map({code: name for name, code in COUNTRIES.items()})

                fig_country = px.bar(
                    country_overall,
                    x='Kraj',
                    y='Średnie',
                    title=f'Średnie zainteresowanie wyszukiwaniami wg krajów - {st.session_state.selected_single_coin}',
                    labels={'Średnie': 'Średnie zainteresowanie (0-100)'},
                    color='Średnie',
                    color_continuous_scale='Viridis'
                )

                fig_country.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_country, use_container_width=True)

                st.dataframe(
                    country_overall[['Kraj', 'Średnie', 'Maksimum', 'Odchylenie', 'Liczba pomiarów']],
                    use_container_width=True,
                    height=300
                )

        with st.expander("Podgląd surowych danych (tabela faktów)"):
            display_columns = ['date', 'keyword', 'trend_keyword', 'value', 'country', 'source']

            if not fact_table_filtered.empty:
                display_df = fact_table_filtered[display_columns].copy()
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                display_df['keyword'] = display_df['keyword'].map({code: name for name, code in COINS.items()})
                display_df['country'] = display_df['country'].map({code: name for name, code in COUNTRIES.items()})

                st.dataframe(display_df.head(100), use_container_width=True, height=300)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Liczba rekordów", f"{len(fact_table_filtered):,}")
                with col2:
                    st.metric("Unikalne kraje", f"{fact_table_filtered['country'].nunique()}")
                with col3:
                    st.metric("Średnie zainteresowanie", f"{fact_table_filtered['value'].mean():.1f}")
                with col4:
                    start_date = fact_table_filtered['date'].min().strftime('%Y-%m')
                    end_date = fact_table_filtered['date'].max().strftime('%Y-%m')
                    st.metric("Zakres danych", f"{start_date} - {end_date}")

        if show_price_data:
            st.markdown("---")
            st.header("Analiza cen kryptowalut (rozszerzenie projektu)")

            st.markdown("""
            ### Moduł analizy cenowej i predykcji
            **Cel akademicki:** Demonstracja integracji różnych źródeł danych w systemie Big Data
            oraz implementacja prostego modelu predykcyjnego w stabilnych warunkach rynkowych.

            **Źródła danych:** Yahoo Finance API  
            **Model:** Regresja liniowa (stabilne warunki rynkowe)  

            **Założenia modelu:**
            - Brak szoków rynkowych (black swan events)
            - Kontynuacja średniego trendu historycznego
            - Stabilne warunki makroekonomiczne
            - Brak gwałtownych zmian regulacyjnych
            """)

            selected_price_coin = st.session_state.selected_single_coin
            price_symbol = COINS[selected_price_coin]

            with st.spinner(f"Ładowanie i analiza danych cenowych dla {selected_price_coin}..."):
                price_df, yearly_df = load_price_data(price_symbol)

                if price_df is not None and not price_df.empty:
                    st.subheader("Aktualna cena i zmiany historyczne")

                    price_stats = calculate_price_changes(price_df)

                    if price_stats:
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric(
                                "Aktualna cena",
                                f"${price_stats['current']:,.2f}",
                                help="Ostatnia znana cena zamknięcia"
                            )

                        with col2:
                            if price_stats["1y"] is not None:
                                st.metric(
                                    "Zmiana 1 rok",
                                    f"{price_stats['1y']:+.2f}%",
                                    help="Zmiana procentowa ceny w ciągu ostatniego roku"
                                )
                            else:
                                st.metric("Zmiana 1 rok", "Brak danych")

                        with col3:
                            if price_stats["5y"] is not None:
                                st.metric(
                                    "Zmiana 5 lat",
                                    f"{price_stats['5y']:+.2f}%",
                                    help="Zmiana procentowa ceny w ciągu ostatnich 5 lat"
                                )
                            else:
                                st.metric("Zmiana 5 lat", "Brak danych")

                        with col4:
                            if price_stats["10y"] is not None:
                                st.metric(
                                    "Zmiana 10 lat",
                                    f"{price_stats['10y']:+.2f}%",
                                    help="Zmiana procentowa ceny w ciągu ostatnich 10 lat"
                                )
                            else:
                                st.metric("Zmiana 10 lat", "Brak danych")

                    st.subheader("Historyczny wykres ceny")

                    col1, col2 = st.columns([3, 1])

                    with col1:
                        fig_price = go.Figure()
                        fig_price.add_trace(go.Scatter(
                            x=price_df.index,
                            y=price_df['Close'].values,
                            mode='lines',
                            name='Cena zamknięcia',
                            line=dict(color=COLORS['price'], width=2),
                            hovertemplate="<b>Data:</b> %{x|%Y-%m-%d}<br><b>Cena:</b> $%{y:,.2f}<extra></extra>"
                        ))

                        if len(price_df) > 50:
                            ma_50 = price_df['Close'].rolling(window=50).mean()
                            fig_price.add_trace(go.Scatter(
                                x=price_df.index,
                                y=ma_50.values,
                                mode='lines',
                                name='Średnia 50-dniowa',
                                line=dict(color=COLORS['secondary'], width=1.5, dash='dash')
                            ))

                        if len(price_df) > 200:
                            ma_200 = price_df['Close'].rolling(window=200).mean()
                            fig_price.add_trace(go.Scatter(
                                x=price_df.index,
                                y=ma_200.values,
                                mode='lines',
                                name='Średnia 200-dniowa',
                                line=dict(color=COLORS['accent'], width=1.5, dash='dot')
                            ))

                        fig_price.update_layout(
                            title=f'Cena {selected_price_coin} (USD) - dane historyczne',
                            xaxis_title='Data',
                            yaxis_title='Cena (USD)',
                            height=500,
                            hovermode='x unified',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            plot_bgcolor='rgba(240, 240, 240, 0.8)'
                        )

                        st.plotly_chart(fig_price, use_container_width=True)

                    with col2:
                        if len(price_df) > 0:
                            current_price = float(price_df['Close'].iloc[-1])
                            min_price = float(price_df['Close'].min())
                            max_price = float(price_df['Close'].max())
                            start_price = float(price_df['Close'].iloc[0])
                            avg_price = float(price_df['Close'].mean())

                            total_change = ((current_price - start_price) / start_price) * 100 if start_price > 0 else 0
                            volatility = float(price_df['Close'].pct_change().std() * np.sqrt(365) * 100)

                            st.metric("Aktualna cena", f"${current_price:,.2f}", f"{total_change:+.1f}%")
                            st.metric("Cena maksymalna", f"${max_price:,.2f}")
                            st.metric("Cena minimalna", f"${min_price:,.2f}")
                            st.metric("Średnia cena", f"${avg_price:,.2f}")
                            st.metric("Volatylność roczna", f"{volatility:.1f}%")
                            st.metric("Dni danych", len(price_df))

                    st.subheader("Predykcja ceny (stabilne warunki rynkowe)")

                    st.info("""
                    **Uwaga akademicka:** Ten model predykcyjny wykorzystuje prostą regresję liniową 
                    i zakłada stabilne warunki rynkowe. Jest to przykład zastosowania analizy predykcyjnej 
                    w systemach Big Data, a nie profesjonalny model inwestycyjny.
                    """)

                    pred_1y, daily_trend = predict_future_price(price_df, 1)
                    pred_3y, _ = predict_future_price(price_df, 3)
                    pred_5y, _ = predict_future_price(price_df, 5)

                    if pred_1y and daily_trend:
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric(
                                "Trend dzienny",
                                f"${daily_trend:.4f}/dzień",
                                help="Średni dzienny trend wzrostu/spadku ceny"
                            )

                        with col2:
                            current_price = price_df['Close'].iloc[-1]
                            change_1y = ((pred_1y - current_price) / current_price) * 100
                            st.metric("Prognoza 1 rok", f"${pred_1y:,.2f}", f"{change_1y:+.1f}%")

                        with col3:
                            change_3y = ((pred_3y - current_price) / current_price) * 100
                            st.metric("Prognoza 3 lata", f"${pred_3y:,.2f}", f"{change_3y:+.1f}%")

                        with col4:
                            change_5y = ((pred_5y - current_price) / current_price) * 100
                            st.metric("Prognoza 5 lat", f"${pred_5y:,.2f}", f"{change_5y:+.1f}%")

                        st.subheader("Wykres: historia + prognoza (5-letni horyzont)")

                        fig_forecast = create_forecast_plot(price_df, forecast_years=5)

                        if fig_forecast:
                            st.plotly_chart(fig_forecast, use_container_width=True)

                            with st.expander("Szczegóły techniczne modelu predykcyjnego"):
                                st.markdown("**Model regresji liniowej:**")

                                df_model = price_df.reset_index()
                                df_model['t'] = np.arange(len(df_model))
                                X = df_model[['t']]
                                y = df_model['Close']
                                model = LinearRegression()
                                model.fit(X, y)

                                col1, col2 = st.columns(2)

                                with col1:
                                    st.metric("Współczynnik a (trend)", f"{model.coef_[0]:.6f}")
                                    st.metric("Trend miesięczny", f"${model.coef_[0] * 30:.2f}/miesiąc")
                                    st.metric("Trend roczny", f"${model.coef_[0] * 365:.2f}/rok")

                                with col2:
                                    st.metric("Współczynnik b", f"{model.intercept_:.2f}")
                                    st.metric("R² (dopasowanie)", f"{model.score(X, y):.4f}")
                                    st.metric("Dopasowanie modelu", f"{model.score(X, y) * 100:.1f}%")

                        st.subheader("Analiza trendu rocznego")

                        if yearly_df is not None and len(yearly_df) > 1:
                            fig_yearly = px.bar(
                                yearly_df,
                                x='Year',
                                y='Close',
                                title=f'Średnia roczna cena {selected_price_coin}',
                                labels={'Close': 'Średnia cena roczna (USD)', 'Year': 'Rok'},
                                color='Close',
                                color_continuous_scale='Viridis'
                            )

                            fig_yearly.update_layout(height=400)
                            st.plotly_chart(fig_yearly, use_container_width=True)

                            yearly_df['Change'] = yearly_df['Close'].pct_change() * 100

                            if 'Change' in yearly_df.columns and not yearly_df['Change'].isna().all():
                                best_year = yearly_df.loc[yearly_df['Change'].idxmax()]
                                worst_year = yearly_df.loc[yearly_df['Change'].idxmin()]

                                col1, col2 = st.columns(2)

                                with col1:
                                    if not pd.isna(best_year['Change']):
                                        st.metric("Najlepszy rok", f"{int(best_year['Year'])}",
                                                  f"{best_year['Change']:+.1f}%")

                                with col2:
                                    if not pd.isna(worst_year['Change']):
                                        st.metric("Najgorszy rok", f"{int(worst_year['Year'])}",
                                                  f"{worst_year['Change']:+.1f}%")

                    else:
                        st.warning("Za mało danych historycznych do budowy modelu predykcyjnego")

st.markdown("---")
st.caption(f"""
**Ostatnia aktualizacja:** {datetime.now().strftime('%d.%m.%Y %H:%M')}  
**Źródła:** Google Trends API, Yahoo Finance  
**Do celów edukacyjnych** | **Projekt:** Analiza trendów w wyszukiwaniu tematów w Google Trends  
**Przedmiot:** Big Data i Hurtownie Danych
""")

st.markdown("---")
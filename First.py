import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as sch

# Настройки страницы Streamlit (Квантовый стиль)
st.set_page_config(page_title="Hedge Fund Quantitative Terminal", layout="wide")
plt.style.use('dark_background')

# --- БЛОК 1. ФУНКЦИИ БЭКЭНДА ---

@st.cache_data(ttl=3600)  # Кэшируем список активов на час, чтобы сайт работал быстро
def get_crypto_assets(limit_assets=25):
    try:
        exchange = ccxt.okx()
        exchange.load_markets()
        usdt_pairs = [symbol for symbol, m in exchange.markets.items() if m['spot'] and m['quote'] == 'USDT' and m['active']]
        return usdt_pairs[:limit_assets]
    except Exception as e:
        # Резервный список на случай сбоя сети/блокировки API
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 'AR/USDT', 'ARKM/USDT', '1INCH/USDT', 'AAVE/USDT']

@st.cache_data(ttl=1800)  # Кэшируем данные котировок на 30 минут
def download_data(symbols, days_back=120):
    exchange = ccxt.okx()
    since = exchange.parse8601((datetime.now() - timedelta(days=days_back)).isoformat())
    combined_df = pd.DataFrame()
    
    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, '1d', since)
            if not ohlcv: continue
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['date', 'close']].rename(columns={'close': symbol})
            df.set_index('date', inplace=True)
            
            combined_df = df if combined_df.empty else combined_df.join(df, how='outer')
        except Exception:
            continue
    return combined_df.ffill().dropna(how='all')

def quasi_diagonalization(cov_matrix, returns_df):
    corr = returns_df.corr() 
    distance_matrix = np.sqrt(0.5 * (1 - corr))
    link = linkage(squareform(distance_matrix), method='ward')
    root = sch.to_tree(link)
    
    def get_sort_order(node):
        if node.is_leaf(): return [node.id]
        return get_sort_order(node.left) + get_sort_order(node.right)
    
    sort_indices = get_sort_order(root)
    sorted_tickers = cov_matrix.columns[sort_indices].tolist()
    ordered_cov = cov_matrix.loc[sorted_tickers, sorted_tickers]
    return ordered_cov, sorted_tickers

def run_pair_backtest(df_prices, asset_A, asset_B, entry_threshold, window=20):
    df = df_prices[[asset_A, asset_B]].dropna().copy()
    hedge_ratio = df[asset_A].mean() / df[asset_B].mean()
    
    df['spread'] = df[asset_A] - (hedge_ratio * df[asset_B])
    df['spread_mean'] = df['spread'].rolling(window=window).mean()
    df['spread_std'] = df['spread'].rolling(window=window).std()
    df['z_score'] = (df['spread'] - df['spread_mean']) / df['spread_std']
    df = df.dropna().copy()
    
    positions = []
    current_pos = 0
    for z in df['z_score']:
        if current_pos == 0:
            if z > entry_threshold: current_pos = -1  # Шорт спреда
            elif z < -entry_threshold: current_pos = 1 # Лонг спреда
        elif current_pos == -1 and z <= 0: current_pos = 0  # Выход в ноль
        elif current_pos == 1 and z >= 0: current_pos = 0   # Выход в ноль
        positions.append(current_pos)
        
    df['position'] = positions
    df['returns_A'] = df[asset_A].pct_change()
    df['returns_B'] = df[asset_B].pct_change()
    df['strategy_returns'] = df['position'].shift(1) * (df['returns_A'] - df['returns_B'])
    df['cum_returns'] = (1 + df['strategy_returns'].fillna(0)).cumprod() - 1
    return df

# --- БЛОК 2. ИНТЕРФЕЙС STREAMLIT ---

st.title("📊 Квантовый Терминал Статистического Арбитража & HRP")
st.markdown("---")

# Боковая панель управления (Sidebar)
st.sidebar.header("🛠 Настройки Системы")
days = st.sidebar.slider("Глубина истории (дней)", 60, 365, 120)
asset_count = st.sidebar.slider("Количество сканируемых активов", 10, 40, 20)

st.sidebar.markdown("---")
st.sidebar.header("📈 Параметры Стратегии")
z_enter = st.sidebar.slider("Порог входа (Z-Score)", 1.5, 3.5, 2.0, step=0.1)
leverage = st.sidebar.slider("Кредитное плечо (Leverage)", 1, 5, 3)

# 1. Загрузка данных
with st.spinner("⚡ Выкачиваем котировки и сканируем рынок ковариаций..."):
    symbols = get_crypto_assets(limit_assets=asset_count)
    # Принудительно добавим наши целевые пары для теста
    for pair in ['AR/USDT', 'ARKM/USDT']:
        if pair not in symbols: symbols.append(pair)
        
    prices_df = download_data(symbols, days_back=days)
    returns_df = prices_df.pct_change().dropna()
    cov_matrix = returns_df.cov() * 365

# Разделяем страницу на две вкладки
tab1, tab2 = st.tabs(["🔍 Макро-Анализ Рынка (HRP)", "💰 Бэктестер Парной Стратегии"])

# ВКЛАДКА 1: Ковариация и Кластеризация
with tab1:
    st.subheader("📦 Иерархическая структура ковариаций рынка")
    st.write("Алгоритм перестраивает матрицу, разбивая хаотичные активы в изолированные секторальные блоки риска.")
    
    ordered_cov, sorted_tickers = quasi_diagonalization(cov_matrix, returns_df)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(ordered_cov, cmap='vlag', center=0, xticklabels=False, yticklabels=True, ax=ax)
        st.pyplot(fig)
    with col2:
        st.markdown("**Оптимизированный порядок активов (по кластерам):**")
        st.dataframe(pd.DataFrame(sorted_tickers, columns=["Ticker"]), height=400)

# ВКЛАДКА 2: Торговля и PnL
with tab2:
    st.subheader("🏎 Симуляция парного арбитража в реальном времени")
    
    col_select1, col_select2 = st.columns(2)
    with col_select1:
        asset_A = st.selectbox("Выбери Актив А (Основной)", options=sorted_tickers, index=sorted_tickers.index('AR/USDT') if 'AR/USDT' in sorted_tickers else 0)
    with col_select2:
        asset_B = st.selectbox("Выбери Актив Б (Хеджирующий)", options=sorted_tickers, index=sorted_tickers.index('ARKM/USDT') if 'ARKM/USDT' in sorted_tickers else 1)
        
    if asset_A == asset_B:
        st.error("❌ Выбери два разных актива для построения спреда!")
    else:
        results = run_pair_backtest(prices_df, asset_A, asset_B, z_enter)
        
        if results is not None:
            # Считаем финальные метрики
            raw_pnl = results['cum_returns'].iloc[-1] * 100
            leveraged_pnl = raw_pnl * leverage
            
            # Выводим виджеты с бабками
            m1, m2, m3 = st.columns(3)
            m1.metric("Доходность без плеча", f"{raw_pnl:.2f}%")
            m2.metric(f"Доходность с плечом х{leverage}", f"{leveraged_pnl:.2f}%", delta=f"Итоговый профит")
            m3.metric("Всего торговых дней", len(results))
            
            # График 1: Сигналы Z-Score
            st.markdown("---")
            st.write("**Текущее состояние спреда и торговые зоны (Z-Score)**")
            fig_z, ax_z = plt.subplots(figsize=(12, 4))
            ax_z.plot(results['z_score'], color='#9933ff', label='Z-Score спреда', lw=1.5)
            ax_z.axhline(0, color='grey', linestyle='--', alpha=0.5)
            ax_z.axhline(z_enter, color='red', linestyle=':', label='Вход в Шорт спреда')
            ax_z.axhline(-z_enter, color='green', linestyle=':', label='Вход в Лонг спреда')
            ax_z.set_ylabel('Сигнал отклонения')
            ax_z.legend(loc='upper left')
            st.pyplot(fig_z)
            
            # График 2: Кривая капитала (Equity Curve)
            st.write("**Динамика роста капитала (Equity Curve %)**")
            fig_eq, ax_eq = plt.subplots(figsize=(12, 4))
            ax_eq.plot(results.index, results['cum_returns'] * 100 * leverage, color='#00ffcc', label='Баланс счета (%)', lw=2)
            ax_eq.axhline(0, color='grey', linestyle='--', alpha=0.5)
            ax_eq.set_ylabel('Прибыль в %')
            ax_eq.legend(loc='upper left')
            st.pyplot(fig_eq)
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as sch

def get_crypto_assets(exchange_id='okx', limit_assets=40):
    """Получает список топ-активов к USDT"""
    exchange = getattr(ccxt, exchange_id)()
    exchange.load_markets()
    usdt_pairs = [symbol for symbol, m in exchange.markets.items() if m['spot'] and m['quote'] == 'USDT' and m['active']]
    return usdt_pairs[:limit_assets]

def download_data(symbols, exchange_id='okx', timeframe='1d', days_back=180):
    """Скачивает исторические данные цен"""
    exchange = getattr(ccxt, exchange_id)()
    since = exchange.parse8601((datetime.now() - timedelta(days=days_back)).isoformat())
    combined_df = pd.DataFrame()
    
    print(f"Выгрузка данных для {len(symbols)} активов...")
    for symbol in tqdm(symbols):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
            if not ohlcv: continue
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['date', 'close']].rename(columns={'close': symbol})
            df.set_index('date', inplace=True)
            
            combined_df = df if combined_df.empty else combined_df.join(df, how='outer')
            time.sleep(0.05)
        except Exception:
            continue
    return combined_df.ffill().dropna(how='all')

def quasi_diagonalization(cov_matrix, returns_df):
    """
    Выполняет квази-диагонализацию матрицы ковариации 
    на основе иерархической кластеризации (HRP подход Марко Лопеса де Прадо)
    """
    # 1. Переводим ковариацию в матрицу расстояний (Distance Matrix)
    corr = returns_df.corr() 
    distance_matrix = np.sqrt(0.5 * (1 - corr))
    
    # 2. Строим связь (linkage) методом Уорда
    link = linkage(squareform(distance_matrix), method='ward')
    
    # 3. Получаем оптимальный порядок активов (сортировка по кластерам)
    def get_sort_order(node):
        if node.is_leaf():
            return [node.id]
        return get_sort_order(node.left) + get_sort_order(node.right)
    
    root = sch.to_tree(link)
    sort_indices = get_sort_order(root)
    sorted_tickers = cov_matrix.columns[sort_indices].tolist()
    
    # 4. Перестраиваем исходную ковариационную матрицу в новом порядке
    ordered_cov = cov_matrix.loc[sorted_tickers, sorted_tickers]
    return ordered_cov, link, sorted_tickers

def run_pair_backtest(df_prices, asset_A, asset_B, entry_threshold=2.0, exit_threshold=0.0):
    """
    Бэктестер стат-арбитража. Симулирует торговлю спредом и рассчитывает PnL.
    """
    print(f"\n🚀 Запуск бэктеста стратегии для пары: {asset_A} / {asset_B}")
    
    # Проверяем, есть ли выбранные активы в базе данных
    if asset_A not in df_prices.columns or asset_B not in df_prices.columns:
        print(f"❌ Ошибка: пары {asset_A} или {asset_B} нет в загруженных данных. Выбираем доступные.")
        return None

    df = df_prices[[asset_A, asset_B]].dropna().copy()
    
    # 1. Расчет коэффициента хеджирования (Hedge Ratio)
    hedge_ratio = df[asset_A].mean() / df[asset_B].mean()
    
    # 2. Спред и динамический Z-Score (окно 20 дней для быстрой крипты)
    df['spread'] = df[asset_A] - (hedge_ratio * df[asset_B])
    window = 20
    df['spread_mean'] = df['spread'].rolling(window=window).mean()
    df['spread_std'] = df['spread'].rolling(window=window).std()
    df['z_score'] = (df['spread'] - df['spread_mean']) / df['spread_std']
    df = df.dropna().copy()
    
    # 3. Эмуляция позиций
    positions = []
    current_pos = 0
    
    for z in df['z_score']:
        if current_pos == 0:
            if z > entry_threshold:
                current_pos = -1  # Шорт спреда (Шорт А, Лонг Б)
            elif z < -entry_threshold:
                current_pos = 1   # Лонг спреда (Лонг А, Шорт Б)
        elif current_pos == -1 and z <= exit_threshold:
            current_pos = 0       # Выход (спреды сошлись)
        elif current_pos == 1 and z >= -exit_threshold:
            current_pos = 0       # Выход (спреды сошлись)
        positions.append(current_pos)
        
    df['position'] = positions
    
    # 4. Расчет доходности (Returns) спреда
    df['returns_A'] = df[asset_A].pct_change()
    df['returns_B'] = df[asset_B].pct_change()
    
    # PnL стратегии: позиция вчера умножается на разность доходностей сегодня
    df['strategy_returns'] = df['position'].shift(1) * (df['returns_A'] - df['returns_B'])
    df['cum_returns'] = (1 + df['strategy_returns'].fillna(0)).cumprod() - 1
    return df

if __name__ == "__main__":
    # 1. Получаем активы и качаем цены
    # Для теста добавим принудительно AR и ARKM в список, если их там не окажется
    symbols = get_crypto_assets(limit_assets=35)
    target_pairs = ['AR/USDT', 'ARKM/USDT']
    for pair in target_pairs:
        if pair not in symbols:
            symbols.append(pair)
            
    prices_df = download_data(symbols, days_back=180)
    
    # 2. Переходим к доходностям
    returns_df = prices_df.pct_change().dropna()
    
    # 3. Считаем матрицу ковариации
    cov_matrix = returns_df.cov() * 365
    
    print("\n" + "="*60)
    print("📊 МАТРИЦА КОВАРИАЦИИ СФОРМИРОВАНА (Фрагмент 5x5):")
    print("="*60)
    print(cov_matrix.iloc[:5, :5])
    print("="*60)
    
    # 4. Поиск активов с максимальной ковариацией
    cov_unstacked = cov_matrix.unstack()
    cov_unstacked = cov_unstacked[cov_unstacked.index.get_level_values(0) != cov_unstacked.index.get_level_values(1)]
    
    print("\n🔥 ТОП ПАР С НАИБОЛЬШЕЙ СВЯЗАННОСТЬЮ ПО КОВАРИАЦИИ:")
    print(cov_unstacked.sort_values(ascending=False).head(10).drop_duplicates())
    
    # 5. Запуск Квази-диагонализации (HRP) внутри __main__
    ordered_cov, link, sorted_tickers = quasi_diagonalization(cov_matrix, returns_df)
    
    print("\n" + "="*60)
    print("🎯 ПОРЯДОК АКТИВОВ ПОСЛЕ КВАЗИ-ДИАГОНАЛИЗАЦИИ:")
    print("="*60)
    print(sorted_tickers[:10], "... и остальные.")
    print("="*60)
    
    # 6. БЭКТЕСТ И РАСЧЕТ ЗАРАБОТКА (Наш PnL модуль)
    # Запускаем симуляцию на паре AR/USDT и ARKM/USDT
    backtest_results = run_pair_backtest(prices_df, 'AR/USDT', 'ARKM/USDT')
    
    if backtest_results is not None:
        final_pnl = backtest_results['cum_returns'].iloc[-1] * 100
        print("\n" + "="*60)
        print(f"💰 ФИНАНСОВЫЙ РЕЗУЛЬТАТ СТРАТЕГИИ:")
        print("="*60)
        print(f"Чистая доходность за 180 дней: {final_pnl:.2f}% (без плеча)")
        print(f"С 3-м кредитным плечом (консервативно): {final_pnl * 3:.2f}%")
        print("="*60)
        
        # Строим график эквити (роста капитала)
        plt.style.use('dark_background')
        plt.figure(figsize=(12, 5))
        plt.plot(backtest_results.index, backtest_results['cum_returns'] * 100, label='Equity Curve (PnL %)', color='#00ffcc', lw=2)
        plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
        plt.title('График роста капитала (Equity Curve) на спреде AR/ARKM', fontsize=12, fontweight='bold')
        plt.ylabel('Прибыль (%)')
        plt.grid(True, alpha=0.1)
        plt.legend()
        plt.tight_layout()
        
    # 7. Визуализация перестроенной ковариационной матрицы HRP
    plt.figure(figsize=(12, 10))
    sns.heatmap(ordered_cov, cmap='vlag', center=0, xticklabels=False, yticklabels=True)
    plt.title("Квази-диагональная матрица ковариации (Блочная структура HRP)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Выводим оба графика на экран
    plt.show()

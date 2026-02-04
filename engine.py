import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm

def get_rolling_metrics(ticker, window=504):
    # Téléchargement des données
    df = yf.download(ticker, period="10y")
    if df.empty: return pd.DataFrame()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Calcul des rendements logarithmiques
    returns_raw = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    
    # Vérification du volume de données
    if len(returns_raw) < window + 252:
        return pd.DataFrame()

    results = []
    test_period = 252 # Backtesting sur 1 an
    
    for i in range(len(returns_raw) - test_period, len(returns_raw)):
        train_window = returns_raw.iloc[i-window:i]
        actual_loss = -returns_raw.iloc[i] # On transforme la perte en valeur positive
        
        # 1. VaR PARAMÉTRIQUE NORMALE 99%
        mu_norm, std_norm = train_window.mean(), train_window.std()
        var_param = -(mu_norm + std_norm * norm.ppf(0.01))
        
        # 2. VaR EMPIRIQUE (HISTORIQUE) 99%
        var_hist = np.percentile(-train_window, 99)
        
        # 3. EXPECTED SHORTFALL 97.5% (Méthode Historique)
        # Moyenne des pertes au-delà du seuil des 2.5% pires rendements
        losses = -train_window
        es_threshold = np.percentile(losses, 97.5)
        expected_shortfall = losses[losses >= es_threshold].mean()

        results.append({
            'Date': returns_raw.index[i],
            'Actual_Loss': actual_loss,
            'VaR_Param': var_param,
            'VaR_Hist': var_hist,
            'ES_975': expected_shortfall
        })
    
    return pd.DataFrame(results).set_index('Date')
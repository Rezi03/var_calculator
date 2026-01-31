import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import t, norm

def get_rolling_metrics(ticker, window=504):
    df = yf.download(ticker, period="10y")
    if df.empty: return pd.DataFrame()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    returns_raw = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    
    if len(returns_raw) < window + 252:
        return pd.DataFrame()

    results = []
    test_period = 252
    
    for i in range(len(returns_raw) - test_period, len(returns_raw)):
        train_window = returns_raw.iloc[i-window:i]
        actual_loss = -returns_raw.iloc[i]
        
        mu_norm, std_norm = train_window.mean(), train_window.std()
        var_param = -(mu_norm + std_norm * norm.ppf(0.01))
        
        var_hist = np.percentile(-train_window, 99)
        
        train_garch = train_window * 100
        model = arch_model(train_garch, p=1, q=1, dist='t', rescale=False)
        res = model.fit(disp="off")
        
        forecast = res.forecast(horizon=1)
        sigma_t = np.sqrt(forecast.variance.values[-1, 0])
        mu_t = forecast.mean.values[-1, 0]
        nu = max(res.params['nu'], 2.1)
        
        var_garch = -(mu_t + sigma_t * t.ppf(0.01, nu)) / 100
        
        sim_returns = t.rvs(nu, loc=mu_t, scale=sigma_t, size=10000)
        tail_threshold = np.percentile(sim_returns, 2.5)
        expected_shortfall = -sim_returns[sim_returns <= tail_threshold].mean() / 100

        results.append({
            'Date': returns_raw.index[i],
            'Actual_Loss': actual_loss,
            'VaR_Param': var_param,
            'VaR_Hist': var_hist,
            'VaR_Garch': var_garch,
            'ES_975': expected_shortfall
        })
    
    return pd.DataFrame(results).set_index('Date')
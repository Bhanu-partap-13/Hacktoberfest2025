# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# ------------------------------------------------------------
# 1. Grab S&P-500 daily close (last 5 years)
# ------------------------------------------------------------
ticker = "^GSPC"
data = yf.download(ticker, period="5y")['Adj Close']
returns = 100 * data.pct_change().dropna()   # % log-returns
print(f"Returns shape: {returns.shape}")

# ------------------------------------------------------------
# 2. Fit GARCH(1,1)
# ------------------------------------------------------------
model = arch_model(returns, p=1, q=1, mean='Zero', vol='GARCH', dist='normal')
res = model.fit(disp='off')
print(res.summary())

# ------------------------------------------------------------
# 3. Key quantities
# ------------------------------------------------------------
ω, α, β = res.params
persistence = α + β
uncond_vol = np.sqrt(ω / (1 - persistence))
print(f"\nPersistence (α+β): {persistence:.4f}")
print(f"Unconditional σ:   {uncond_vol:.4f}%")

# ------------------------------------------------------------
# 4. In-sample volatility plot
# ------------------------------------------------------------
fitted_vol = res.conditional_volatility
plt.figure(figsize=(12,4))
plt.plot(fitted_vol, label='GARCH σₜ')
plt.plot(np.abs(returns), label='|rₜ|', alpha=0.5)
plt.legend()
plt.title('GARCH(1,1) – fitted volatility vs absolute returns')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. 10-day ahead forecast
# ------------------------------------------------------------
horizon = 10
forecasts = res.forecast(horizon=horizon)
future_vol = forecasts.variance.iloc[-1] ** 0.5
print(f"\nAnnualised 10-day forecast σ: {future_vol*np.sqrt(252):.2f}%")

plt.figure(figsize=(6,3))
plt.plot(future_vol, marker='o')
plt.title('10-day ahead volatility forecast (%)')
plt.xlabel('days ahead')
plt.ylabel('σ')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

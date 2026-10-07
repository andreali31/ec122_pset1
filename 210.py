import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(
    "capm4.dat",
    sep=r"\s+",
    header=None
)

# Assign column names
df.columns = [
    "DATE",
    "GE",
    "GM",
    "IBM",
    "DISNEY",
    "MICROSOFT",
    "MOBIL_EXXON",
    "MKT",
    "RISKFREE"
]

# 2. a) 
print(df.head())
print(df.shape)

# b) market risk premium
df["MARKET_PREMIUM"] = df["MKT"] - df["RISKFREE"]

companies = [
    "GE",
    "GM",
    "IBM",
    "DISNEY",
    "MICROSOFT",
    "MOBIL_EXXON"
]

results = []

for company in companies:
    # Stock risk premium
    y = df[company] - df["RISKFREE"]
    x = df["MARKET_PREMIUM"]

    # Means
    x_bar = x.mean()
    y_bar = y.mean()

    # OLS beta (slope)
    beta = ((x - x_bar) * (y - y_bar)).sum() / ((x - x_bar)**2).sum()

    # OLS alpha (intercept)
    alpha = y_bar - beta * x_bar

    results.append({
        "Company": company,
        "Alpha": alpha,
        "Beta": beta
    })

results_df = pd.DataFrame(results)

print("\nCAPM RESULTS:")
print(results_df)

print("\nMost aggressive:")
print(results_df.loc[results_df["Beta"].idxmax()])

print("\nMost defensive:")
print(results_df.loc[results_df["Beta"].idxmin()])

# c) Microsoft market and stock risk premiums
x = df["MARKET_PREMIUM"]
y = df["MICROSOFT"] - df["RISKFREE"]

# Get Microsoft's alpha and beta from part (b)
msft_result = results_df[results_df["Company"] == "MICROSOFT"].iloc[0]

alpha_msft = msft_result["Alpha"]
beta_msft = msft_result["Beta"]

# Predicted Microsoft risk premium
y_hat = alpha_msft + beta_msft * x

print("\nMicrosoft CAPM:")
print(f"Alpha = {alpha_msft:.6f}")
print(f"Beta = {beta_msft:.6f}")
print(
    f"MSFT risk premium = {alpha_msft:.6f} "
    f"+ {beta_msft:.6f}(Market risk premium)"
)

# Plot actual observations
plt.scatter(x, y, alpha=0.6, label="Actual observations")

# Sort x so the fitted line is drawn correctly
order = np.argsort(x)

# Plot fitted regression line
plt.plot(
    x.iloc[order],
    y_hat.iloc[order],
    label="Fitted CAPM line"
)

plt.xlabel("Market Risk Premium (MKT - RISKFREE)")
plt.ylabel("Microsoft Risk Premium (MICROSOFT - RISKFREE)")
plt.title("Microsoft CAPM Regression")

plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)

plt.legend()
plt.show()

# d), a_j = 0

x = df["MARKET_PREMIUM"]

results_no_alpha = []

for company in companies:

    # Company's risk premium
    y = df[company] - df["RISKFREE"]

    # OLS beta when intercept is forced to zero
    beta_no_alpha = (x * y).sum() / (x ** 2).sum()

    results_no_alpha.append({
        "Company": company,
        "Beta (alpha=0)": beta_no_alpha
    })

results_no_alpha_df = pd.DataFrame(results_no_alpha)

print("\nCAPM RESULTS WITH ALPHA = 0:")
print(results_no_alpha_df)
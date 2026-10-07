import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data -- first row contains column names
df = pd.read_csv(
    "cps4_small.dat",
    sep=r"\s+",
    header=0
)

print(df.head())
print(df.shape)
print(df.columns)

# 2.15 a)
print("\nWAGE SUMMARY:")
print(df["wage"].describe())

print("\nEDUC SUMMARY:")
print(df["educ"].describe())

print("\nSKEWNESS:")
print("WAGE:", df["wage"].skew())
print("EDUC:", df["educ"].skew())

# histogram WAGE
plt.hist(df["wage"], bins=30, edgecolor="black")

plt.xlabel("Hourly Wage ($)")
plt.ylabel("Frequency")
plt.title("Histogram of WAGE")

plt.show()

# EDUC WAGE
plt.hist(
    df["educ"],
    bins=range(int(df["educ"].min()), int(df["educ"].max()) + 2),
    edgecolor="black"
)

plt.xlabel("Years of Education")
plt.ylabel("Frequency")
plt.title("Histogram of EDUC")

plt.show()

# b)
x = df["educ"]
y = df["wage"]

# Sample means
x_bar = x.mean()
y_bar = y.mean()

# OLS slope
beta2 = (
    ((x - x_bar) * (y - y_bar)).sum()
    / ((x - x_bar) ** 2).sum()
)

# OLS intercept
beta1 = y_bar - beta2 * x_bar

print("\nWAGE REGRESSION:")
print(f"Intercept (beta1) = {beta1:.6f}")
print(f"EDUC coefficient (beta2) = {beta2:.6f}")

print(
    f"WAGE_hat = {beta1:.6f} + "
    f"{beta2:.6f}(EDUC)"
)

# c) predict wages
df["wage_hat"] = beta1 + beta2 * df["educ"]

# Least squares residuals: actual - predicted
df["residual"] = df["wage"] - df["wage_hat"]

# Display first few
print("\nFIRST 10 RESIDUALS:")
print(df[["wage", "educ", "wage_hat", "residual"]].head(10))

# Plot residuals against EDUC
plt.scatter(df["educ"], df["residual"], alpha=0.5)

# Horizontal line at residual = 0
plt.axhline(y=0, linewidth=1)

plt.xlabel("Years of Education (EDUC)")
plt.ylabel("Residual")
plt.title("Least Squares Residuals vs. Education")

plt.show()

# e)
# Create EDUC squared
df["educ2"] = df["educ"] ** 2

x = df["educ2"]
y = df["wage"]

# Sample means
x_bar = x.mean()
y_bar = y.mean()

# OLS slope: alpha2
alpha2 = (
    ((x - x_bar) * (y - y_bar)).sum()
    / ((x - x_bar) ** 2).sum()
)

# OLS intercept: alpha1
alpha1 = y_bar - alpha2 * x_bar

print("\nQUADRATIC REGRESSION:")
print(f"alpha1 = {alpha1:.6f}")
print(f"alpha2 = {alpha2:.6f}")
print(f"WAGE_hat = {alpha1:.6f} + {alpha2:.6f}(EDUC^2)")

# f)
# Create a smooth range of education values
educ_range = np.linspace(
    df["educ"].min(),
    df["educ"].max(),
    200
)

# Predicted wage from linear model
wage_linear = beta1 + beta2 * educ_range

# Predicted wage from quadratic model
wage_quadratic = alpha1 + alpha2 * educ_range**2

# Actual data
plt.scatter(
    df["educ"],
    df["wage"],
    alpha=0.3,
    label="Actual data"
)

# Linear fitted model
plt.plot(
    educ_range,
    wage_linear,
    linewidth=2,
    label="Linear model"
)

# Quadratic fitted model
plt.plot(
    educ_range,
    wage_quadratic,
    linewidth=2,
    label="Quadratic model"
)

plt.xlabel("Years of Education (EDUC)")
plt.ylabel("Hourly Wage ($)")
plt.title("Linear vs. Quadratic Wage Models")

plt.legend()
plt.show()

# g) 
# Create log wage
df["ln_wage"] = np.log(df["wage"])

# Histogram of ln(WAGE)
plt.hist(df["ln_wage"], bins=30, edgecolor="black")

plt.xlabel("ln(WAGE)")
plt.ylabel("Frequency")
plt.title("Histogram of ln(WAGE)")

plt.show()

# Compare skewness
print("WAGE skewness:", df["wage"].skew())
print("ln(WAGE) skewness:", df["ln_wage"].skew())

# h)
x = df["educ"]
y = df["ln_wage"]

x_bar = x.mean()
y_bar = y.mean()

# OLS slope
gamma2 = (
    ((x - x_bar) * (y - y_bar)).sum()
    / ((x - x_bar) ** 2).sum()
)

# OLS intercept
gamma1 = y_bar - gamma2 * x_bar

print("\nLOG-LINEAR REGRESSION:")
print(f"gamma1 = {gamma1:.6f}")
print(f"gamma2 = {gamma2:.6f}")
print(f"ln(WAGE)_hat = {gamma1:.6f} + {gamma2:.6f}(EDUC)")

# Predicted wages at 12 and 14 years of education
wage_12_log = np.exp(gamma1 + gamma2 * 12)
wage_14_log = np.exp(gamma1 + gamma2 * 14)

# Marginal effects
ME_log_12 = gamma2 * wage_12_log
ME_log_14 = gamma2 * wage_14_log

print("\nLOG-LINEAR MARGINAL EFFECTS:")
print(f"Predicted wage at EDUC=12: ${wage_12_log:.4f}")
print(f"Marginal effect at EDUC=12: ${ME_log_12:.4f}/hour")

print(f"\nPredicted wage at EDUC=14: ${wage_14_log:.4f}")
print(f"Marginal effect at EDUC=14: ${ME_log_14:.4f}/hour")

ME_linear = beta2

ME_quad_12 = 2 * alpha2 * 12
ME_quad_14 = 2 * alpha2 * 14

print("\nMARGINAL EFFECT COMPARISON:")
print(f"{'Model':<15} {'EDUC=12':>12} {'EDUC=14':>12}")
print("-" * 41)

print(
    f"{'Linear':<15} "
    f"{ME_linear:>12.4f} "
    f"{ME_linear:>12.4f}"
)

print(
    f"{'Quadratic':<15} "
    f"{ME_quad_12:>12.4f} "
    f"{ME_quad_14:>12.4f}"
)

print(
    f"{'Log-linear':<15} "
    f"{ME_log_12:>12.4f} "
    f"{ME_log_14:>12.4f}"
)
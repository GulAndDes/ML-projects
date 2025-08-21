# housing_project.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Данные
data = fetch_california_housing(as_frame=True)
df = data.frame
print(df.head())

sns.histplot(df["MedHouseVal"], bins=50, kde=True)
plt.title("Распределение цен на жильё")
plt.show()

corr = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Корреляция признаков")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    df.drop("MedHouseVal", axis=1), df["MedHouseVal"], test_size=0.2, random_state=42
)

# Линейная регрессия
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_lr = linreg.predict(X_test)

# RandomForest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# Оценка
def evaluate(y_true, y_pred, name):
    print(f"\n{name}:")
    print(" MAE:", mean_absolute_error(y_true, y_pred))
    print(" R²:", r2_score(y_true, y_pred))


evaluate(y_test, y_pred_lr, "Линейная регрессия")
evaluate(y_test, y_pred_rf, "RandomForest")

# Визуализация
plt.scatter(y_test, y_pred_rf, alpha=0.3)
plt.xlabel("Фактическая цена")
plt.ylabel("Предсказанная цена")
plt.title("Фактические vs предсказанные цены (RandomForest)")
plt.show()

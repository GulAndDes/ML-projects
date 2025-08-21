import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = "ex1data1.txt"
x, y = np.loadtxt(data, delimiter=",", usecols=(0, 1), unpack=True)
X = np.ones((len(x), 2))
X[:, 1] = x
X[:5]

plt.scatter(X[:, 1], y, marker="x", c="r", s=20)
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")

# Вариант решения 1
model = LinearRegression(fit_intercept=True)

model.fit(X[:, 1][:, np.newaxis], y)

xfit = np.linspace(5, 23, 1000)
yfit = model.predict(xfit[:, np.newaxis])

plt.scatter(X[:, 1], y, marker="x", c="r", s=20, label="data")
plt.plot(xfit, yfit, label="h(x) = %0.2f + %0.2fx" % (model.intercept_, model.coef_[0]))
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
pst = plt.legend(loc="lower right", frameon=True)
pst.get_frame().set_edgecolor("k")
print("Model intercept: ", model.intercept_)
print("Model slope:     ", model.coef_[0])
plt.show()

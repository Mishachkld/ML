import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tabulate import tabulate

# Загрузка данных
diabetes = datasets.load_diabetes()
data = diabetes.data
target = diabetes.target
feature_names = diabetes.feature_names

# Анализ данных и выбор признака
print("Доступные признаки:")
for i, name in enumerate(feature_names):
    print(f"{i}: {name}")

# Выбираем 'bmi' (индекс 2) как наиболее подходящий для линейной регрессии
selected_feature = 2
print(f"\nВыбран признак: {feature_names[selected_feature]}")

X = data[:, selected_feature].reshape(-1, 1)
y = target

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Линейная регрессия с помощью Scikit-Learn
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)
sklearn_coef = sklearn_model.coef_[0]
sklearn_intercept = sklearn_model.intercept_

# 2. Собственная реализация линейной регрессии
def CustomLinearRegression(X, y):
    X_mean = np.mean(X)
    y_mean = np.mean(y)

    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean) ** 2)

    coef = numerator / denominator
    intercept = y_mean - coef * X_mean

    return coef, intercept

custom_coef, custom_intercept = CustomLinearRegression(X_train.ravel(), y_train)

# Вывод коэффициентов
print("\nКоэффициенты регрессии:")
print(f"Scikit-Learn: y = {sklearn_coef:.2f}x + {sklearn_intercept:.2f}")
print(f"Собственный метод: y = {custom_coef:.2f}x + {custom_intercept:.2f}")

# Визуализация
plt.figure(figsize=(12, 6))

# Данные
plt.scatter(X_test, y_test, color='black', label='Реальные значения')

# Предсказания Scikit-Learn
plt.plot(X_test, sklearn_model.predict(X_test),
         color='blue', linewidth=2, label='Scikit-Learn')

# Предсказания собственного метода
x_vals = np.array([X.min(), X.max()])
y_vals = custom_coef * x_vals + custom_intercept
plt.plot(x_vals, y_vals, color='red', linestyle='--',
         linewidth=2, label='Собственный метод')

plt.xlabel(feature_names[selected_feature])
plt.ylabel('Целевая переменная')
plt.title('Сравнение линейной регрессии')
plt.legend()
plt.grid(True)
plt.show()

# Таблица с результатами предсказаний
predictions_sklearn = sklearn_model.predict(X_test)
predictions_custom = custom_coef * X_test.ravel() + custom_intercept

results = []
for i in range(min(10, len(X_test))):
    results.append([
        X_test[i][0],
        y_test[i],
        predictions_sklearn[i],
        predictions_custom[i],
        abs(y_test[i] - predictions_sklearn[i]),
        abs(y_test[i] - predictions_custom[i])
    ])

print("\nТаблица результатов предсказаний (первые 10 записей):")
print(tabulate(results,
               headers=['Признак', 'Реальное значение',
                        'Scikit-Learn', 'Собственный метод',
                        'Ошибка (Scikit)', 'Ошибка (Собств)'],
               floatfmt=".2f"))
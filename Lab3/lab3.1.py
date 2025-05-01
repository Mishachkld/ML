# Импорт необходимых библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Загрузка датасета Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# 1. Визуализация данных с помощью Matplotlib
plt.figure(figsize=(12, 5))

# График sepal length vs sepal width
plt.subplot(1, 2, 1)
for target, color in zip([0, 1, 2], ['red', 'green', 'blue']):
    subset = df[df['target'] == target]
    plt.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'],
                c=color, label=iris.target_names[target])
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Sepal Length vs Sepal Width')
plt.legend()

# График petal length vs petal width
plt.subplot(1, 2, 2)
for target, color in zip([0, 1, 2], ['red', 'green', 'blue']):
    subset = df[df['target'] == target]
    plt.scatter(subset['petal length (cm)'], subset['petal width (cm)'],
                c=color, label=iris.target_names[target])
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('Petal Length vs Petal Width')
plt.legend()

plt.tight_layout()
plt.show()

# 2. Визуализация с помощью pairplot
sns.pairplot(df, hue='target', palette='viridis', markers=['o', 's', 'D'])
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()

# 3. Подготовка двух датасетов
# Первый датасет: setosa (0) и versicolor (1)
df1 = df[df['target'].isin([0, 1])].copy()

# Второй датасет: versicolor (1) и virginica (2)
df2 = df[df['target'].isin([1, 2])].copy()


# Функция для выполнения задач 4-8
def train_and_evaluate(df, title):
    # 4. Разделение на обучающую и тестовую выборки
    X = df.iloc[:, :4].values
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 5. Создание модели логистической регрессии
    clf = LogisticRegression(random_state=0)

    # 6. Обучение модели
    clf.fit(X_train, y_train)

    # 7. Предсказание
    y_pred = clf.predict(X_test)

    # 8. Оценка точности
    accuracy = clf.score(X_test, y_test)
    print(f"{title} - Точность модели: {accuracy:.4f}")

    return accuracy


# Применение функции к обоим датасетам
print("\nРезультаты классификации:")
accuracy1 = train_and_evaluate(df1, "Setosa vs Versicolor")
accuracy2 = train_and_evaluate(df2, "Versicolor vs Virginica")

# 9. Генерация синтетического датасета и классификация
# Генерация данных
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                           n_informative=2, random_state=1, n_clusters_per_class=1)

# Визуализация сгенерированных данных
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
plt.title('Сгенерированный датасет для бинарной классификации')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.colorbar()
plt.show()

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создание и обучение модели
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)

# Предсказание и оценка
y_pred = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
print(f"\nСгенерированный датасет - Точность модели: {accuracy:.4f}")

# Визуализация границы принятия решений
plt.figure(figsize=(8, 6))

# Создание сетки для визуализации
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Предсказание для каждой точки сетки
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Визуализация
plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
plt.title('Граница принятия решений логистической регрессии')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.show()
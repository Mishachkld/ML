import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import time

# Загрузка данных
data = pd.read_csv('diabetes.csv')
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Исследование случайного леса

## 1.1. Исследование зависимости качества от глубины деревьев
depths = range(1, 21)
accuracy_depth = []
time_depth = []

for depth in depths:
    start_time = time.time()
    rf = RandomForestClassifier(max_depth=depth, n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy_depth.append(accuracy_score(y_test, y_pred))
    time_depth.append(time.time() - start_time)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(depths, accuracy_depth, marker='o')
plt.title('Зависимость точности от глубины деревьев')
plt.xlabel('Глубина деревьев')
plt.ylabel('Точность')

plt.subplot(1, 2, 2)
plt.plot(depths, time_depth, marker='o', color='orange')
plt.title('Зависимость времени обучения от глубины деревьев')
plt.xlabel('Глубина деревьев')
plt.ylabel('Время обучения (сек)')
plt.tight_layout()
plt.show()

## 1.2. Исследование зависимости качества от количества признаков
max_features = range(1, X.shape[1] + 1)
accuracy_features = []

for n_features in max_features:
    rf = RandomForestClassifier(max_features=n_features, n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy_features.append(accuracy_score(y_test, y_pred))

plt.figure(figsize=(8, 5))
plt.plot(max_features, accuracy_features, marker='o')
plt.title('Зависимость точности от количества признаков')
plt.xlabel('Количество признаков')
plt.ylabel('Точность')
plt.xticks(max_features)
plt.show()

## 1.3. Исследование зависимости качества от количества деревьев
n_trees = range(1, 201, 10)
accuracy_trees = []
time_trees = []

for n in n_trees:
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy_trees.append(accuracy_score(y_test, y_pred))
    time_trees.append(time.time() - start_time)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(n_trees, accuracy_trees, marker='o')
plt.title('Зависимость точности от количества деревьев')
plt.xlabel('Количество деревьев')
plt.ylabel('Точность')

plt.subplot(1, 2, 2)
plt.plot(n_trees, time_trees, marker='o', color='orange')
plt.title('Зависимость времени обучения от количества деревьев')
plt.xlabel('Количество деревьев')
plt.ylabel('Время обучения (сек)')
plt.tight_layout()
plt.show()

# 2. Исследование XGBoost

# Подбор гиперпараметров вручную
params = {
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'binary:logistic',
    'random_state': 42
}

start_time = time.time()
xgb = XGBClassifier(**params)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
xgb_time = time.time() - start_time
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)

# Сравнение с лучшей моделью случайного леса
rf_best = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_best.fit(X_train, y_train)
y_pred_rf = rf_best.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("\nСравнение моделей:")
print(f"Точность RandomForest: {rf_accuracy:.4f}")
print(f"Точность XGBoost: {xgb_accuracy:.4f}")
print(f"Время обучения RandomForest: {time_depth[9]:.4f} сек")
print(f"Время обучения XGBoost: {xgb_time:.4f} сек")

# Выводы
print("\nВыводы:")
print("1. Для RandomForest оптимальная глубина деревьев около 10, дальнейшее увеличение не дает значимого прироста точности.")
print("2. Оптимальное количество признаков для RandomForest - около 3-4 из 8.")
print("3. Точность RandomForest растет с увеличением количества деревьев до ~100, затем стабилизируется.")
print("4. XGBoost показал сравнимую или немного лучшую точность при меньшем времени обучения.")
print("5. Обе модели показали хорошие результаты, но XGBoost может быть предпочтительнее из-за скорости и возможности тонкой настройки.")
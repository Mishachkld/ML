import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder
import graphviz
from sklearn.tree import export_graphviz

# Загрузка данных
data = pd.read_csv('diabetes.csv')

# Проверка на пропущенные значения
print(data.isnull().sum())

# Разделение данных на признаки и целевую переменную
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Задача 1: Сравнение логистической регрессии и решающего дерева
# Логистическая регрессия
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# Решающее дерево с настройками по умолчанию
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)

# Вывод метрик
def print_metrics(y_true, y_pred, model_name):
    print(f"Метрики для модели {model_name}:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print(f"Precision: {precision_score(y_true, y_pred):.2f}")
    print(f"Recall: {recall_score(y_true, y_pred):.2f}")
    print(f"F1-score: {f1_score(y_true, y_pred):.2f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_pred):.2f}")
    print()

print_metrics(y_test, y_pred_log, "Логистической регрессии")
print_metrics(y_test, y_pred_tree, "Решающего дерева")

# Задача 2: Исследование зависимости метрики от глубины дерева
max_depths = range(1, 21)
train_scores = []
test_scores = []

for depth in max_depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_scores.append(f1_score(y_train, tree.predict(X_train)))
    test_scores.append(f1_score(y_test, tree.predict(X_test)))

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_scores, label='Train F1-score')
plt.plot(max_depths, test_scores, label='Test F1-score')
plt.xlabel('Глубина дерева')
plt.ylabel('F1-score')
plt.title('Зависимость F1-score от глубины дерева')
plt.legend()
plt.grid()
plt.show()

# Оптимальная глубина
optimal_depth = max_depths[np.argmax(test_scores)]
print(f"Оптимальная глубина дерева: {optimal_depth}")

# Задача 3: Визуализация дерева, важность признаков, PR и ROC кривые
# Обучение модели с оптимальной глубиной
optimal_tree = DecisionTreeClassifier(max_depth=optimal_depth, random_state=42)
optimal_tree.fit(X_train, y_train)

# Визуализация дерева
plt.figure(figsize=(20, 10))
plot_tree(optimal_tree, filled=True, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], rounded=True)
plt.show()

# Важность признаков
importances = optimal_tree.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Важность признаков')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45)
plt.tight_layout()
plt.show()

# ROC кривая
y_proba_tree = optimal_tree.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba_tree)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC кривая')
plt.legend()
plt.show()

# PR кривая
precision, recall, _ = precision_recall_curve(y_test, y_proba_tree)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR кривая')
plt.legend()
plt.show()

# Опциональное задание: Исследование зависимости метрики от max_features
max_features_range = range(1, X.shape[1] + 1)
test_scores_features = []

for max_features in max_features_range:
    tree = DecisionTreeClassifier(max_depth=optimal_depth, max_features=max_features, random_state=42)
    tree.fit(X_train, y_train)
    test_scores_features.append(f1_score(y_test, tree.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(max_features_range, test_scores_features, label='Test F1-score')
plt.xlabel('max_features')
plt.ylabel('F1-score')
plt.title('Зависимость F1-score от max_features')
plt.legend()
plt.grid()
plt.show()
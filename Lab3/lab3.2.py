import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('Titanic.csv')

# Часть 1: Предобработка данных

data_cleaned = data.dropna()

non_numeric_cols = ['Name', 'Ticket', 'Cabin']
data_cleaned = data_cleaned.drop(columns=non_numeric_cols)

data_cleaned['Sex'] = data_cleaned['Sex'].map({'male': 0, 'female': 1})
data_cleaned['Embarked'] = data_cleaned['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

data_cleaned = data_cleaned.drop(columns=['PassengerId'])

# 5. Вычисление процента потерянных данных
initial_rows = len(data)
cleaned_rows = len(data_cleaned)
lost_percentage = ((initial_rows - cleaned_rows) / initial_rows) * 100
print(f"Процент потерянных данных после очистки: {lost_percentage:.2f}%")

# Часть 2: Машинное обучение

# 2.1. Разделение данных на обучающую и тестовую выборки
X = data_cleaned.drop(columns=['Survived'])
y = data_cleaned['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2.2. Обучение модели логистической регрессии
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 2.3. Оценка точности модели
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.4f}")

X_no_embarked = X.drop(columns=['Embarked'])
X_train_no_e, X_test_no_e, y_train_no_e, y_test_no_e = train_test_split(X_no_embarked, y, test_size=0.2, random_state=42)

model_no_e = LogisticRegression(max_iter=1000)
model_no_e.fit(X_train_no_e, y_train_no_e)
y_pred_no_e = model_no_e.predict(X_test_no_e)
accuracy_no_e = accuracy_score(y_test_no_e, y_pred_no_e)
print(f"Точность модели без признака Embarked: {accuracy_no_e:.4f}")
print(f"Изменение точности: {accuracy - accuracy_no_e:.4f}")
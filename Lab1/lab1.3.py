import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_linear_regression():
    student_data = pd.read_csv('student_scores.csv')

    X_values = student_data.iloc[:, :-1].values
    y_values = student_data.iloc[:, 1].values

    X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(
        X_values, y_values, test_size=0.2, random_state=42)

    regression_model = LinearRegression()
    regression_model.fit(X_train_set, y_train_set)

    y_predicted = regression_model.predict(X_test_set)

    mae_score = mean_absolute_error(y_test_set, y_predicted)
    r2_score_value = r2_score(y_test_set, y_predicted)
    mape_score = calculate_mape(y_test_set, y_predicted)

    print("\nОценка качества линейной регрессии:")
    print(f"Средняя абсолютная ошибка (MAE): {mae_score:.2f}")
    print(f"Коэффициент детерминации (R²): {r2_score_value:.2f}")
    print(f"Средняя абсолютная процентная ошибка (MAPE): {mape_score:.2f}%")

    print("\nАнализ результатов:")
    if r2_score_value > 0.8:
        print("Хорошо объясняет вариативность данных (R² > 0.8)")
    elif r2_score_value > 0.5:
        print("Объясняет вариативность данных (0.5 < R² ≤ 0.8)")
    else:
        print("Плохо объясняет вариативность данных (R² ≤ 0.5)")

    if mape_score < 10:
        print("Высокая точность (MAPE < 10%)")
    elif mape_score < 20:
        print("Хорошая точность (10% ≤ MAPE < 20%)")
    else:
        print("Низкая точность (MAPE ≥ 20%)")

evaluate_linear_regression()
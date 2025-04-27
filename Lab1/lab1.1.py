import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def ReadCsvFile(fileName):
    #Чтение данных
    with open(fileName, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        data = list(reader)
    return headers, data

def GetStatistics(data, columnIndex):
    #Получение статистики по столбцу
    column = [float(row[columnIndex]) for row in data]
    stats = {
        'Count': len(column),
        'Min': min(column),
        'Max': max(column),
        'Mean': np.mean(column)
    }
    return stats

def PlotData(x, y, xLabel, yLabel, title):
    #Построение графика исходных данных
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(x, y, color='blue')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title('Исходные данные')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.scatter(x, y, color='blue')

    n = len(x)
    sumX = sum(x)
    sumY = sum(y)
    sumXY = sum(xi*yi for xi, yi in zip(x, y))
    sumX2 = sum(xi**2 for xi in x)

    a = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX**2)
    b = (sumY - a * sumX) / n

    xMin, xMax = min(x), max(x)
    xLine = np.linspace(xMin, xMax, 100)
    yLine = a * xLine + b
    plt.plot(xLine, yLine, color='red')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title('Линейная регрессия')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.scatter(x, y, color='blue')
    plt.plot(xLine, yLine, color='red')

    # Добавление квадратов ошибок
    for xi, yi in zip(x, y):
        yPred = a * xi + b
        error = abs(yi - yPred)
        rect = Rectangle((xi, min(yi, yPred)), 0, error,
                         linewidth=1, edgecolor='green',
                         facecolor='green', alpha=0.2)
        plt.gca().add_patch(rect)

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title('Квадраты ошибок')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return a, b

def Main():
    fileName = 'student_scores.csv'
    headers, data = ReadCsvFile(fileName)

    print("Доступные столбцы:")
    for i, header in enumerate(headers):
        print(f"{i}: {header}")

    xCol = int(input("Выберите столбец для X (по умолчанию 0): ") or 0)
    yCol = int(input("Выберите столбец для Y (по умолчанию 1): ") or 1)

    x = [float(row[xCol]) for row in data]
    y = [float(row[yCol]) for row in data]

    print("\nСтатистика по X:")
    xStats = GetStatistics(data, xCol)
    print(f"Количество: {xStats['Count']}")
    print(f"Минимум: {xStats['Min']:.2f}")
    print(f"Максимум: {xStats['Max']:.2f}")
    print(f"Среднее: {xStats['Mean']:.2f}")

    print("\nСтатистика по Y:")
    yStats = GetStatistics(data, yCol)
    print(f"Количество: {yStats['Count']}")
    print(f"Минимум: {yStats['Min']:.2f}")
    print(f"Максимум: {yStats['Max']:.2f}")
    print(f"Среднее: {yStats['Mean']:.2f}")

    a, b = PlotData(x, y, headers[xCol], headers[yCol], 'Линейная регрессия')

    print(f"\nУравнение регрессионной прямой: y = {a:.2f}x + {b:.2f}")

Main()
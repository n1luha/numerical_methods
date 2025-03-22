import numpy as np
import matplotlib.pyplot as plt

a, b = 0, 10
n_val = [4, 7, 10]

"Определение функции 1"
def f1(x):

    "Принимает"
    "х - Число, для которого вычисляется значение функции"

    "Возвращает"
    "Значение функции lg(x + 2) + x для каждого элемента x"

    return np.log10(x + 2) + x


"Определение функции 2"
def f2(x):

    "Принимает"
    "х - Число, для которого вычисляется значение функции"

    "Возвращает"
    "Значение функции x^3 - 0.1x^2 + 0.4|x| + 2 для каждого элемента x"

    return np.power(x, 3) - 0.1 * np.power(x, 2) + 0.4 * np.abs(x) + 2


"Создание равномерной сетки"
def uniform_grid(a, b, n):

    "Принимает"
    "a - начало интервала"
    "b - конец интервала"
    "n - количество узлов на интервале"

    "Возвращает"
    "Массив из равномерно распределенных точек на интервале [a, b]"

    return np.linspace(a, b, n)


"Создание Чебышевской сетки"
def chebyshev_grid(a, b, n):

    "Принимает"
    "a - начало интервала"
    "b - конец интервала"
    "n - количество узлов на интервале"

    "Возвращает"
    "Массив из точек, распределенных по узлам Чебышева на интервале [a, b]"

    k = np.arange(n)
    nodes = np.cos((2 * k + 1) * np.pi / (2 * n))
    return 0.5 * (a + b) + 0.5 * (b - a) * nodes


"Базисные полиномы Лагранжа"
def lagrange_basis(x, xt, k):

    "Принимает"
    "х - точка, в которой вычисляется базисный полином"
    "xt - массив узлов интерполяции"
    "k - индекс базисного полинома / номер узла"

    "Возвращает"
    "Значение k-ого базисного полинома Лагранжа в точке х"

    return np.prod([(x - xt[i]) / (xt[k] - xt[i]) for i in range(len(xt)) if i != k], axis=0)


"Интерполяционный полином Лагранжа"
def lagrange_polinom(x, xt, yt):

    "Принимает"
    "х - точка, в которой вычисляется базисный полином"
    "xt - массив узлов интерполяции"
    "yt - массив значений функции в узлах интерполяции"

    "Возвращает"
    "Значение интерполяционного полинома Лагранжа в точке х"

    return sum(yt[k] * lagrange_basis(x, xt, k) for k in range(len(xt)))


"Вспомогательная функция для выбора сетки"
def select_grid(grid_type, a, b, n):

    "Принимает"
    "grid_type - тип сетки"
    "a - начало интервала"
    "b - конец интервала"
    "n - количество узлов на интервале"

    "Возвращает"
    "Массив узлов сетки в зависимости от выбранного типа"

    if grid_type == 'uniform':
        return uniform_grid(a, b, n)
    elif grid_type == 'chebyshev':
        return chebyshev_grid(a, b, n)
    else:
        raise ValueError(f"Неверный тип сетки. Используйте 'uniform' или 'chebyshev'.")


"Построение графиков интерполяции"
def plot_int(f, a, b, n_val, grid_type):

    "Принимает"
    "f - функция, которую нужно интерполировать"
    "a - начало интервала"
    "b - конец интервала"
    "n_val - список значений n (количество узлов интерполяции)"
    "grid_type - тип сетки"

    "Строит график исходной функции и интерполяционных полиномов для каждого значения n из n_val"

    x_plot = np.linspace(a, b, 1000)
    y_plot = f(x_plot)

    plt.figure(figsize=(10, 5))
    plt.plot(x_plot, y_plot, label='Исходная функция', color='black', linewidth=2)

    colors = ['blue', 'green', 'red']
    for i, n in enumerate(n_val):
        xt, yt = select_grid(grid_type, a, b, n), f(select_grid(grid_type, a, b, n))
        y_int = np.array([lagrange_polinom(x, xt, yt) for x in x_plot])

        plt.plot(x_plot, y_int, color=colors[i], linestyle='--', label=f'n = {n}')
        plt.scatter(xt, yt, color=colors[i])
    plt.legend()
    plt.title(f'Интерполяция полиномом Лагранжа ({grid_type} сетка)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

"Цикл, строящий интерполяционные полиномы для двух различных сеток и функций"
for f in [f1, f2]:
    for grid in ['uniform', 'chebyshev']:
        plot_int(f, a, b, n_val, grid)

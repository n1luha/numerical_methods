import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.log10(x + 2) + x

def g(x):
    return np.power(x, 3) - 0.1 * np.power(x, 2) + 0.4 * np.abs(x) + 2

# Генерация узлов и значений функции
def nodes(a, b, n, f):
    # a, b — границы отрезка
    # n — количество узлов
    # f — функция
    # Возвращает: массив узлов и значения функции в этих узлах
    x_nodes = np.linspace(a, b, n)
    y_nodes = f(x_nodes)
    return x_nodes, y_nodes

# Построение квадратичного сплайна
def spline(x_nodes, y_nodes, init):
    # x_nodes, y_nodes — массивы узлов и значений функции
    # init — начальное значение первой производной
    # Возвращает: список функций-квадратичных сплайнов на каждом интервале
    n = len(x_nodes) - 1
    h = np.diff(x_nodes)
    a = np.zeros(n)
    b = np.zeros(n)
    c = y_nodes[:-1]

    b[0] = init
    for i in range(n):
        if i > 0:
            b[i] = 2 * (y_nodes[i] - y_nodes[i - 1]) / h[i - 1] - b[i - 1]
        a[i] = (y_nodes[i + 1] - y_nodes[i] - b[i] * h[i]) / h[i] ** 2

    splines = []
    for i in range(n):
        a_i, b_i, c_i, x_i = a[i], b[i], c[i], x_nodes[i]
        def spline_fn(x, a=a_i, b=b_i, c=c_i, x0=x_i):
            return a * (x - x0) ** 2 + b * (x - x0) + c
        splines.append(spline_fn)

    return splines

# Построение графика интерполяции
def plot_interpolation(f, x_nodes, y_nodes, init, title):
    # f — оригинальная функция
    # x_nodes, y_nodes — узлы и значения
    # init — начальная производная
    # title — заголовок графика
    splines = spline(x_nodes, y_nodes, init)

    x_full = np.linspace(x_nodes[0], x_nodes[-1], 1000)
    y_true = f(x_full)

    y_interp = np.zeros_like(x_full)
    for i in range(len(x_nodes) - 1):
        idx = np.where((x_full >= x_nodes[i]) & (x_full <= x_nodes[i + 1]))
        y_interp[idx] = splines[i](x_full[idx])

    plt.figure(figsize=(10, 6))
    plt.plot(x_full, y_true, label='Оригинальная функция', linewidth=2)
    plt.plot(x_full, y_interp, label='Интерполяция сплайном', linestyle='--')
    plt.plot(x_nodes, y_nodes, 'ro', label='Узлы')
    plt.grid(True)
    plt.title(title)
    plt.legend()
    plt.show()

# Построение графика ошибки
def plot_error(f, x_nodes, y_nodes, init, title):
    # f — функция
    # x_nodes, y_nodes — узлы и значения
    # init — значение производной
    # title — заголовок графика
    x_err, y_err = error(x_nodes, y_nodes, f, init)
    plt.figure(figsize=(10, 6))
    plt.plot(x_err, y_err, label='Абсолютная ошибка')
    plt.grid(True)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.show()

# Вычисление поточечной ошибки
def error(x_nodes, y_nodes, f, init, num=100):
    # x_nodes, y_nodes — узлы и значения
    # f — функция
    # init — значение производной
    # num — число точек на интервал
    # Возвращает: x и соответствующие ошибки
    splines = spline(x_nodes, y_nodes, init)
    x_error = np.array([])
    y_error = np.array([])

    for i in range(len(x_nodes) - 1):
        x = np.linspace(x_nodes[i], x_nodes[i + 1], num)
        y_true = f(x)
        y_int = splines[i](x)
        err = np.abs(y_true - y_int)
        x_error = np.concatenate((x_error, x))
        y_error = np.concatenate((y_error, err))

    return x_error, y_error

# Возмущение значений функции и оценка ошибки
def perturbation_error(x_nodes, y_nodes, f, init, perturbation_percent, num=100):
    # x_nodes, y_nodes — узлы и значения
    # f — функция
    # init — производная
    # perturbation_percent — процент возмущения
    # num — точек интерполяции
    # Возвращает: массив ошибок
    y_nodes_perturbed = y_nodes + y_nodes * np.random.uniform(-perturbation_percent, perturbation_percent, len(y_nodes))
    splines = spline(x_nodes, y_nodes_perturbed, init)

    x_error = np.array([])
    y_error = np.array([])

    for i in range(len(x_nodes) - 1):
        x = np.linspace(x_nodes[i], x_nodes[i + 1], num)
        y_true = f(x)
        y_int = splines[i](x)
        err = np.abs(y_true - y_int)
        x_error = np.concatenate((x_error, x))
        y_error = np.concatenate((y_error, err))

    return y_error

# Несколько запусков эксперимента с возмущениями
def perturbation_experiment(f, a, b, nod, perturbations, init, num=100, trials=20):
    # f — функция
    # a, b — границы
    # nod — число узлов
    # perturbations — список значений возмущения
    # init — производная
    # trials — число повторов
    # Возвращает: словарь с максимальными ошибками
    perturbation_errors = {p: [] for p in perturbations}
    for _ in range(trials):
        for p in perturbations:
            x_nodes, y_nodes = nodes(a, b, nod, f)
            err = perturbation_error(x_nodes, y_nodes, f, init, p, num)
            perturbation_errors[p].append(np.max(err))
    return perturbation_errors

# График ошибок при разных уровнях возмущения
def plot_perturbation(f, a, b, init, title):
    nod = 20
    perturbations = [0.01, 0.02, 0.03, 0.04, 0.05]
    perturbation_errors = perturbation_experiment(f, a, b, nod, perturbations, init)
    plt.figure(figsize=(10, 6))
    plt.boxplot([perturbation_errors[p] for p in perturbations],
                labels=[f'{int(p * 100)}%' for p in perturbations],
                showfliers=False)
    plt.grid(True)
    plt.xlabel('Максимальное возмущение данных')
    plt.ylabel('Относительная ошибка интерполяции, %')
    plt.title(title)
    plt.show()


a, b = -1.5, 1.5
init = 0
n = 20

x_nodes, y_nodes = nodes(a, b, n, f)
plot_interpolation(f, x_nodes, y_nodes, init, 'Интерполяция функции f(x)')
plot_error(f, x_nodes, y_nodes, init, 'Ошибка интерполяции функции f(x)')
plot_perturbation(f, a, b, init, 'Возмущение данных для f(x)')

x_nodes, y_nodes = nodes(a, b, n, g)
plot_interpolation(g, x_nodes, y_nodes, init, 'Интерполяция функции g(x)')
plot_error(g, x_nodes, y_nodes, init, 'Ошибка интерполяции функции g(x)')
plot_perturbation(g, a, b, init, 'Возмущение данных для g(x)')

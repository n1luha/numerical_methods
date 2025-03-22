import numpy as np
import matplotlib.pyplot as plt

a = 0
b = 2
integral = -25.4666666667
k = 9.2

"Определение функции"
def f(x):

    "Принимает"
    "х - Число, для которого вычисляется значение функции"

    "Возвращает"
    "Значение функции x^5 - 9.2x^3 + 5.5x^2 - 7x для каждого элемента x"

    return np.power(x, 5) - k * np.power(x, 3) + 5.5 * np.power(x, 2) - 7 * x

"Метод средних прямоугольников"
def midpoint_rule(a, b, n):

    "Принимает"
    "a - начало интервала интегрирования"
    "b - конец интервала интегрирования"
    "n - количество разбиений интервала"

    "Возвращает"
    "Приближенное значение интеграла"

    h = (b - a) / n
    integral = 0
    for i in range(n):
        x_mid = a + (i + 0.5) * h
        integral += f(x_mid)

    integral *= h
    return integral

"Программа для вычисления интеграла с заданной точностью с использованием правила Рунге"
def midpoint_rule_with_rung(a, b, eps, max_iter=1000):

    "Принимает"
    "a - начало интервала интегрирования"
    "b = конец интервала интегрирования"
    "eps - требуемая точность вычисления интеграла"
    "max_iter - максимальное число итераций"

    "Возвращает"
    "int_new - приближенное значение интеграла, вычисленное с заданной точностью"

    n = 5
    int_prev = midpoint_rule(a, b, n)

    for it in range(1, max_iter + 1):
        n *= 2
        int_new = midpoint_rule(a, b, n)
        error = np.abs(int_new - int_prev) / 3

        if error < eps:
            return int_new, it, error

        int_prev = int_new

    return int_new, max_iter, error

epsilons = [10 ** (-i) for i in range(1, 11)]

"Вычисление интеграла с заданной точностью"
for eps in epsilons:
    int = midpoint_rule_with_rung(a, b, eps, max_iter=1000)
    print(int)

"Внесение погрешности в константу и вычисление относительной ошибки"
ns = [5 * (2 ** i) for i in range(10)]
hs = [(b - a) / n for n in ns]
errors_h = []
for n in ns:
    int = midpoint_rule(a, b, n)
    errors_h.append(np.abs(int - integral))

"Логарифмирование ошибок и шагов"
log_hs = np.log2(hs)
log_errors = np.log2(errors_h)

"Определение порядка точности и константы"
order, const = np.polyfit(log_hs, log_errors, 1)

"Функция для внесения возмущений"
def add_perturbation(k, perturbation_percent):

    "Принимает"
    "k - значение константы"
    "perturbation_percent - процент возмущения, который нужно внести в константу"

    "Возвращает"
    "Новое значение константы, полученное после добавления случайного возмущения"

    perturbation = k * (perturbation_percent / 100)
    return k + np.random.uniform(-perturbation, perturbation)

"Уровни возмущений"
perturbation_levels = [1, 2, 3]
num = 20

"Результаты"
results = {level: [] for level in perturbation_levels}

"Проведение экспериментов"
for level in perturbation_levels:
    for _ in range(num):
        perturbed_const = add_perturbation(const, level)
        relative_error = abs(perturbed_const - const) / const
        results[level].append(np.abs(relative_error * 100))

plt.figure(figsize=(10, 6))
plt.boxplot([results[level] for level in perturbation_levels], labels=[f'{level}%' for level in perturbation_levels])
plt.xlabel('Среднее фактическое возмущение данных, %')
plt.ylabel('Относительная ошибка, %')
plt.title('Зависимость относительной ошибки от уровня возмущения')
plt.grid()
plt.show()

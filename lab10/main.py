import math
import numpy as np
import matplotlib.pyplot as plt

#     y' = y - x^2 + 1
#     y(0) = 0.5
#     точний розв'язок: y = (x + 1)^2 - 0.5 * e^x

def f(x, y):
    return y - x**2 + 1


def exact_solution(x):
    return (x + 1)**2 - 0.5 * math.exp(x)


a = 0.0           # початок відрізка
b = 2.0           # кінець відрізка
y0 = 0.5          # початкова умова y(a)=y0
h = 0.1           # крок для методу Адамса
h_rk = 1e-2       # за методичкою для Рунге-Кутта: h = 10^-2
eps = 1e-5        # задана точність для автоматичного вибору кроку

# Допоміжні функції
def make_grid(a, b, h):
    n = int(round((b - a) / h))
    x = np.array([a + i * h for i in range(n + 1)], dtype=float)
    if x[-1] < b:
        x = np.append(x, b)
    else:
        x[-1] = b
    return x


def exact_array(x_values):
    return np.array([exact_solution(float(x)) for x in x_values], dtype=float)


def rk4_one_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h * k1 / 2)
    k3 = f(x + h / 2, y + h * k2 / 2)
    k4 = f(x + h, y + h * k3)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def rk4_fixed(a, b, y0, h):
    x_values = make_grid(a, b, h)
    y_values = np.zeros(len(x_values), dtype=float)
    y_values[0] = y0

    for i in range(len(x_values) - 1):
        hi = x_values[i + 1] - x_values[i]
        y_values[i + 1] = rk4_one_step(x_values[i], y_values[i], hi)

    return x_values, y_values

# Ч.1. Метод прогнозу та корекції Адамса 2-го порядку
def adams2_predict_correct_fixed(a, b, y0, h, correction_iterations=2):
    x_values = make_grid(a, b, h)
    y_values = np.zeros(len(x_values), dtype=float)
    y_pred_values = np.full(len(x_values), np.nan, dtype=float)
    pc_diff_values = np.full(len(x_values), np.nan, dtype=float)
    err_est_values = np.full(len(x_values), np.nan, dtype=float)

    y_values[0] = y0

    # Для методу Адамса 2-го порядку потрібно знати y1.
    # Його знаходимо методом Рунге-Кутта 4-го порядку.
    y_values[1] = rk4_one_step(x_values[0], y_values[0], x_values[1] - x_values[0])

    for i in range(1, len(x_values) - 1):
        hi = x_values[i + 1] - x_values[i]
        # Для рівномірної сітки hi = h. Якщо останній крок трохи інший,
        # формула все одно працює як наближення.
        f_i = f(x_values[i], y_values[i])
        f_im1 = f(x_values[i - 1], y_values[i - 1])

        # Прогноз Адамса-Башфорта 2-го порядку:
        y_pred = y_values[i] + hi / 2 * (3 * f_i - f_im1)

        # Корекція Адамса-Мултона 2-го порядку:
        y_corr = y_pred
        for _ in range(correction_iterations):
            y_corr = y_values[i] + hi / 2 * (f(x_values[i + 1], y_corr) + f_i)

        y_values[i + 1] = y_corr
        y_pred_values[i + 1] = y_pred
        pc_diff_values[i + 1] = y_corr - y_pred

        # Оцінка локальної похибки для скоригованого значення
        # через різницю між корекцією і прогнозом.
        # Для Адамса 2-го порядку часто беруть |y_corr - y_pred| / 6.
        err_est_values[i + 1] = abs(y_corr - y_pred) / 6

    return x_values, y_values, y_pred_values, pc_diff_values, err_est_values


def adams2_adaptive(a, b, y0, h_start, eps, h_min=1e-6, h_max=0.25):
    x_values = [a]
    y_values = [y0]
    h_values = []
    err_values = []

    x = a
    y = y0
    h_current = h_start

    # Перший крок робимо методом Рунге-Кутта, щоб мати дві точки для Адамса.
    if x + h_current > b:
        h_current = b - x
    y_next = rk4_one_step(x, y, h_current)
    x_prev, y_prev = x, y
    x = x + h_current
    y = y_next
    x_values.append(x)
    y_values.append(y)
    h_values.append(h_current)
    err_values.append(0.0)

    while x < b - 1e-14:
        if x + h_current > b:
            h_current = b - x

        accepted = False
        while not accepted:
            f_i = f(x, y)
            f_im1 = f(x_prev, y_prev)

            y_pred = y + h_current / 2 * (3 * f_i - f_im1)
            y_corr = y_pred
            for _ in range(2):
                y_corr = y + h_current / 2 * (f(x + h_current, y_corr) + f_i)

            err = abs(y_corr - y_pred) / 6

            if err > eps and h_current / 2 >= h_min:
                h_current /= 2
                # після зміни кроку історію безпечніше перезапустити через RK4
                # з поточної точки, тому повторюємо спробу з меншим кроком
            else:
                accepted = True

        x_prev, y_prev = x, y
        x = x + h_current
        y = y_corr
        x_values.append(x)
        y_values.append(y)
        h_values.append(h_current)
        err_values.append(err)

        if err < eps / 8 and h_current * 2 <= h_max:
            h_current *= 2
            # щоб не ламати багатокрокову формулу, після збільшення кроку
            # наступний крок все одно використовує останню прийняту історію

    return np.array(x_values), np.array(y_values), np.array(h_values), np.array(err_values)

# Ч.2. Метод Рунге-Кутта 4-го порядку
def rk4_local_error_exact(x_values, y_values):
    return y_values - exact_array(x_values)


def rk4_runge_error_on_grid(a, b, y0, h):
    x_h, y_h = rk4_fixed(a, b, y0, h)

    errors = np.zeros(len(x_h), dtype=float)
    errors[0] = 0.0

    # Для кожного вузла x_{n+1}: порівнюємо один крок h і два кроки h/2.
    for i in range(len(x_h) - 1):
        hi = x_h[i + 1] - x_h[i]
        y_one = rk4_one_step(x_h[i], y_h[i], hi)
        y_half = rk4_one_step(x_h[i], y_h[i], hi / 2)
        y_two_half = rk4_one_step(x_h[i] + hi / 2, y_half, hi / 2)

        errors[i + 1] = abs((16 / 15) * (y_two_half - y_one))

    return x_h, errors


def rk4_adaptive(a, b, y0, h_start, eps, h_min=1e-7, h_max=0.25):
    x_values = [a]
    y_values = [y0]
    h_values = []
    err_values = []

    x = a
    y = y0
    h_current = h_start

    while x < b - 1e-14:
        if x + h_current > b:
            h_current = b - x

        while True:
            y_one = rk4_one_step(x, y, h_current)
            y_half = rk4_one_step(x, y, h_current / 2)
            y_two_half = rk4_one_step(x + h_current / 2, y_half, h_current / 2)
            err = abs((16 / 15) * (y_two_half - y_one))

            if err > eps and h_current / 2 >= h_min:
                h_current /= 2
            else:
                break

        x += h_current
        y = y_two_half
        x_values.append(x)
        y_values.append(y)
        h_values.append(h_current)
        err_values.append(err)

        if err < eps / 32 and h_current * 2 <= h_max:
            h_current *= 2

    return np.array(x_values), np.array(y_values), np.array(h_values), np.array(err_values)

# Виведення таблиць
def print_table(title, x, y, y_exact=None, err=None, max_rows=20):
    print("\n" + title)
    print("-" * len(title))
    if y_exact is None:
        y_exact = exact_array(x)
    if err is None:
        err = y - y_exact

    print(f"{'i':>4} {'x':>12} {'y_num':>18} {'y_exact':>18} {'error':>18}")
    rows = min(len(x), max_rows)
    for i in range(rows):
        print(f"{i:4d} {x[i]:12.6f} {y[i]:18.10f} {y_exact[i]:18.10f} {err[i]:18.3e}")
    if len(x) > max_rows:
        print(f"... показано перші {max_rows} рядків із {len(x)}")
    print(f"Максимальна абсолютна похибка = {np.max(np.abs(err)):.6e}")

# Побудова графіків
def plot_xy(x, y, title, xlabel, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o', markersize=3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()


def main():
    # Аналітичний розв'язок
    x_exact = np.linspace(a, b, 400)
    y_exact_plot = exact_array(x_exact)
    plot_xy(x_exact, y_exact_plot,
            "Аналітичний розв'язок y(x)", "x", "y(x)",
            "01_exact_solution.png")

    # Ч.1. Адамс 2-го порядку
    x_ad, y_ad, y_pred, pc_diff, ad_err_est = adams2_predict_correct_fixed(a, b, y0, h)
    ad_exact = exact_array(x_ad)
    ad_err_exact = y_ad - ad_exact

    print_table("Ч.1. Метод прогнозу та корекції Адамса 2-го порядку", x_ad, y_ad, ad_exact, ad_err_exact)

    plot_xy(x_ad, y_ad,
            "Ч.1. Чисельний розв'язок методом Адамса 2-го порядку",
            "x", "y", "02_adams_solution.png")

    plot_xy(x_ad, ad_err_exact,
            "Ч.1. Локальна похибка Адамса через точний розв'язок: y_n - y(x_n)",
            "x", "похибка", "03_adams_error_exact.png")

    plot_xy(x_ad, ad_err_est,
            "Ч.1. Оцінка локальної похибки Адамса через прогноз-корекцію",
            "x", "оцінка похибки", "04_adams_error_estimate.png")

    # Автоматичний вибір кроку для Адамса
    x_ad_auto, y_ad_auto, h_ad_auto, err_ad_auto = adams2_adaptive(a, b, y0, h, eps)
    print_table("Ч.1. Адамс 2-го порядку з автоматичним вибором кроку", x_ad_auto, y_ad_auto)

    plot_xy(x_ad_auto[1:], h_ad_auto,
            "Ч.1. Залежність кроку h(x) для методу Адамса",
            "x", "h", "05_adams_adaptive_step.png")

    # Ч.2. Рунге-Кутта 4-го порядку
    x_rk, y_rk = rk4_fixed(a, b, y0, h_rk)
    rk_exact = exact_array(x_rk)
    rk_err_exact = rk4_local_error_exact(x_rk, y_rk)

    print_table("Ч.2. Метод Рунге-Кутта 4-го порядку", x_rk, y_rk, rk_exact, rk_err_exact)

    plot_xy(x_rk, y_rk,
            "Ч.2. Чисельний розв'язок методом Рунге-Кутта 4-го порядку",
            "x", "y", "06_rk4_solution.png")

    plot_xy(x_rk, rk_err_exact,
            "Ч.2. Локальна похибка РК4 через точний розв'язок: y_n - y(x_n)",
            "x", "похибка", "07_rk4_error_exact.png")

    # Дослідження залежності похибки від кроку для РК4
    h_list = [0.2, 0.1, 0.05, 0.025, 0.0125]
    max_errors = []
    for hh in h_list:
        xx, yy = rk4_fixed(a, b, y0, hh)
        max_errors.append(np.max(np.abs(yy - exact_array(xx))))

    print("\nДослідження залежності похибки РК4 від величини кроку")
    print("-------------------------------------------------------")
    print(f"{'h':>12} {'max error':>18}")
    for hh, ee in zip(h_list, max_errors):
        print(f"{hh:12.6f} {ee:18.6e}")

    plot_xy(np.array(h_list), np.array(max_errors),
            "Ч.2. Залежність максимальної похибки РК4 від кроку h",
            "h", "max |похибка|", "08_rk4_error_vs_h.png")

    # Локальна похибка за методом Рунге
    x_run, rk_runge_err = rk4_runge_error_on_grid(a, b, y0, h_rk)
    plot_xy(x_run, rk_runge_err,
            "Ч.2. Локальна похибка РК4 за методом Рунге",
            "x", "похибка за Рунге", "09_rk4_runge_error.png")

    # Оцінка необхідного кроку для заданої точності
    # Беремо останній перевірений крок і похибку: error ~ C*h^4
    last_h = h_list[-1]
    last_err = max_errors[-1]
    if last_err > 0:
        h_needed = last_h * (eps / last_err) ** 0.25
        print(f"\nОцінка необхідного кроку РК4 для eps={eps}: h ≈ {h_needed:.6e}")

    # Автоматичний вибір кроку для РК4
    x_rk_auto, y_rk_auto, h_rk_auto, err_rk_auto = rk4_adaptive(a, b, y0, h_start=0.1, eps=eps)
    print_table("Ч.2. РК4 з автоматичним вибором кроку", x_rk_auto, y_rk_auto)

    plot_xy(x_rk_auto[1:], h_rk_auto,
            "Ч.2. Залежність кроку h(x) для РК4",
            "x", "h", "10_rk4_adaptive_step.png")




if __name__ == "__main__":
    main()
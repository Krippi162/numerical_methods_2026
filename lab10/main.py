import numpy as np
import matplotlib.pyplot as plt

# --- Задані параметри та функції ---
def f(x, y):
    """Диференціальне рівняння y' = f(x, y)"""
    return x - y

def exact_solution(x):
    """Точний (аналітичний) розв'язок для порівняння"""
    return x - 1 + 2 * np.exp(-x)

a, b = 0, 2          # Відрізок [a, b]
y0 = 1.0             # Початкова умова y(x0) = y0
h_initial = 0.1      # Початковий крок (для RK4 за умовою h=10^-2, але для наочності візьмемо 0.1)
epsilon = 1e-4       # Задана точність для автоматичного вибору кроку

# ==========================================
# ЧАСТИНА 1: Метод Рунге-Кутта 4-го порядку
# ==========================================

def rk4_step(x, y, h):
    """Один крок методу Рунге-Кутта 4-го порядку"""
    k1 = f(x, y)
    k2 = f(x + h/2, y + h * k1/2)
    k3 = f(x + h/2, y + h * k2/2)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

def rk4_fixed_step(a, b, y0, h):
    """Розв'язок методом Рунге-Кутта з фіксованим кроком"""
    x_vals = np.arange(a, b + h, h)
    y_vals = np.zeros(len(x_vals))
    y_vals[0] = y0
    
    for i in range(len(x_vals) - 1):
        y_vals[i+1] = rk4_step(x_vals[i], y_vals[i], h)
    return x_vals, y_vals

def rk4_auto_step(a, b, y0, h, eps):
    """Метод Рунге-Кутта з автоматичним вибором кроку за правилом Рунге"""
    x_vals, y_vals, h_vals = [a], [y0], [h]
    x, y = a, y0
    
    while x < b:
        # Обчислення з кроком h та h/2
        y_h = rk4_step(x, y, h)
        y_half_1 = rk4_step(x, y, h/2)
        y_half_2 = rk4_step(x + h/2, y_half_1, h/2)
        
        # Локальна похибка за методом Рунге
        error = (16 / 15) * abs(y_h - y_half_2)
        
        if error > eps:
            h /= 2  # Зменшуємо крок
        else:
            x += h
            y = y_h
            x_vals.append(x)
            y_vals.append(y)
            h_vals.append(h)
            
            # Якщо похибка дуже мала, збільшуємо крок
            if error <= eps / 32:
                h *= 2
    return np.array(x_vals), np.array(y_vals), np.array(h_vals)

# ЧАСТИНА 2: Метод Адамса (Прогноз і Корекція) 2-го порядку

def adams_pc2_fixed_step(a, b, y0, h, eps_iter=1e-5):
    """Метод прогнозу та корекції Адамса 2-го порядку"""
    x_vals = np.arange(a, b + h, h)
    n = len(x_vals)
    y_vals = np.zeros(n)
    y_vals[0] = y0
    
    # Для Адамса 2-го порядку потрібен ще один початковий вузол (y1)
    # Знайдемо його методом Рунге-Кутта
    y_vals[1] = rk4_step(x_vals[0], y_vals[0], h)
    
    errors_estimate = [0, 0] # Для збереження похибки y_kor - y_pr
    
    for i in range(1, n - 1):
        x_n, y_n = x_vals[i], y_vals[i]
        f_n = f(x_n, y_n)
        f_n_minus_1 = f(x_vals[i-1], y_vals[i-1])
        
        # Етап прогнозу
        y_pr = y_n + (h / 2) * (3 * f_n - f_n_minus_1)
        
        # Етап корекції (ітерації)
        y_kor = y_pr
        while True:
            y_kor_new = y_n + (h / 2) * (f(x_vals[i+1], y_kor) + f_n)
            if abs(y_kor_new - y_kor) < eps_iter:
                y_kor = y_kor_new
                break
            y_kor = y_kor_new
            
        y_vals[i+1] = y_kor
        errors_estimate.append(abs(y_kor - y_pr))
        
    return x_vals, y_vals, np.array(errors_estimate)

# ==========================================
# ВИКОНАННЯ ТА ПОБУДОВА ГРАФІКІВ
# ==========================================

# 1. Розв'язок RK4 з фіксованим кроком та обчислення точної похибки
x_rk, y_rk = rk4_fixed_step(a, b, y0, h_initial)
exact_y_rk = exact_solution(x_rk)
error_rk_exact = np.abs(y_rk - exact_y_rk)

# 2. Розв'язок RK4 з автоматичним вибором кроку
x_rk_auto, y_rk_auto, h_rk_auto = rk4_auto_step(a, b, y0, h_initial, epsilon)

# 3. Розв'язок Адамса з фіксованим кроком
x_ad, y_ad, err_ad_est = adams_pc2_fixed_step(a, b, y0, h_initial)
exact_y_ad = exact_solution(x_ad)
error_ad_exact = np.abs(y_ad - exact_y_ad)

# --- Побудова графіків ---
plt.figure(figsize=(15, 10))

# Графік 1: Розв'язки
plt.subplot(2, 2, 1)
plt.plot(x_rk, exact_y_rk, 'k-', label='Точний розв\'язок')
plt.plot(x_rk, y_rk, 'bo--', markersize=4, label='Рунге-Кутта 4')
plt.plot(x_ad, y_ad, 'rx-.', markersize=4, label='Адамс ПК 2')
plt.title("Розв'язки задачі Коші")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Графік 2: Локальні похибки Адамса
plt.subplot(2, 2, 2)
plt.plot(x_ad, error_ad_exact, 'r-', label='Точна похибка |y_n - y(x_n)|')
plt.plot(x_ad, err_ad_est, 'g--', label='Оцінка похибки |y_kor - y_pr|')
plt.title("Похибки методу Адамса 2-го порядку")
plt.xlabel('x')
plt.ylabel('Похибка')
plt.legend()
plt.grid(True)

# Графік 3: Локальна похибка Рунге-Кутта
plt.subplot(2, 2, 3)
plt.plot(x_rk, error_rk_exact, 'b-', label='Точна похибка |y_n - y(x_n)|')
plt.title("Похибка методу Рунге-Кутта 4-го порядку")
plt.xlabel('x')
plt.ylabel('Похибка')
plt.legend()
plt.grid(True)

# Графік 4: Зміна кроку h(x) для RK4 з авто-кроком
plt.subplot(2, 2, 4)
plt.step(x_rk_auto, h_rk_auto, 'm-', where='post', label='Величина кроку h(x)')
plt.title("Автоматичний вибір кроку (Рунге-Кутта 4)")
plt.xlabel('x')
plt.ylabel('h')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


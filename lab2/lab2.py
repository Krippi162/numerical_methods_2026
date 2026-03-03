import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# ====================
# 1. ПІДГОТОВКА ДАНИХ
# ====================
def prepare_data():
    # Варіант 2: RPS та CPU (%)
    data = [
        ["RPS", "CPU"],
        [50, 20],
        [100, 35],
        [200, 60],
        [400, 110],
        [800, 210]
    ]
    with open('data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

def read_data(filename):
    if not os.path.exists(filename): prepare_data()
    x, y = [], []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row['RPS']))
            y.append(float(row['CPU']))
    return np.array(x), np.array(y)

# =======================
# 2. МАТЕМАТИЧНІ МЕТОДИ
# =======================

# Таблиця розділених різниць (Пункт 2)
def get_divided_diff_table(x, y):
    n = len(y)
    table = np.zeros([n, n])
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (x[i+j] - x[i])
    return table

# Поліном Ньютона
def newton_poly(x_nodes, coeffs, x):
    n = len(x_nodes) - 1
    res = coeffs[n]
    for k in range(1, n + 1):
        res = coeffs[n-k] + (x - x_nodes[n-k]) * res
    return res

# Поліном Лагранжа (Дослідницька частина, п. 4)
def lagrange_poly(x_nodes, y_nodes, x):
    def basis(j):
        p = 1
        for i in range(len(x_nodes)):
            if i != j:
                p *= (x - x_nodes[i]) / (x_nodes[j] - x_nodes[i])
        return p
    return sum(y_nodes[j] * basis(j) for j in range(len(x_nodes)))

# Факторіальний метод (Пункт 3)
def factorial_poly(x_nodes, y_nodes, target_x):
    n = len(y_nodes)
    h = np.mean(np.diff(x_nodes)) # середній крок
    t = (target_x - x_nodes[0]) / h
    diffs = np.zeros((n, n))
    diffs[:, 0] = y_nodes
    for j in range(1, n):
        for i in range(n - j):
            diffs[i, j] = diffs[i+1, j-1] - diffs[i, j-1]
    res = diffs[0, 0]
    t_prod, fact = 1, 1
    for k in range(1, n):
        t_prod *= (t - k + 1)
        fact *= k
        res += (diffs[0, k] / fact) * t_prod
    return res

# ==================================
# 3. ОСНОВНЕ ВИКОНАННЯ ТА ВИСНОВКИ
# ==================================

# Зчитуємо дані
x_data, y_data = read_data('data.csv')
full_table = get_divided_diff_table(x_data, y_data)
coeffs = full_table[0, :]

# Вивід таблиці в консоль
print("\n" + "="*65)
print(f"{'ТАБЛИЦЯ РОЗДІЛЕНИХ РІЗНИЦЬ (ВАРІАНТ 2)':^65}")
print("="*65)
print(f"{'x':>5} | {'f(x)':>5} | {'1st diff':>10} | {'2nd diff':>10} | {'3rd diff':>10}")
print("-" * 65)
for i in range(len(x_data)):
    row = f"{x_data[i]:5.0f} | {y_data[i]:5.0f}"
    for j in range(1, len(x_data) - i):
        row += f" | {full_table[i][j]:10.6f}"
    print(row)

# Прогноз для 600 RPS
target = 600
res_n = newton_poly(x_data, coeffs, target)
res_f = factorial_poly(x_data, y_data, target)

print("\n" + "-"*65)
print(f"Прогноз для {target} RPS (Ньютон): {res_n:.2f}%")
print(f"Прогноз для {target} RPS (Факторіальний): {res_f:.2f}%")
print("-" * 65)

# =======================================
# 4. ДОСЛІДНИЦЬКА ЧАСТИНА (ВІЗУАЛІЗАЦІЯ)
# =======================================

plt.figure(figsize=(15, 10))

# --- ГРАФІК 1: Вплив кількості вузлів та ефект Рунге ---
plt.subplot(2, 2, 1)
x_fine = np.linspace(50, 800, 500)
for n in [5, 10, 20]:
    x_n = np.linspace(50, 800, n)
    # Тестова функція для наочності (тренд + осциляція)
    y_n = 0.25 * x_n + 10 + np.sin(x_n/40) * 7
    c_n = get_divided_diff_table(x_n, y_n)[0, :]
    y_plot = [newton_poly(x_n, c_n, xi) for xi in x_fine]
    plt.plot(x_fine, y_plot, label=f'Вузлів n={n}')
plt.scatter(x_data, y_data, color='red', label='Дані Варіанту 2')
plt.title("1. Вплив n та Ефект Рунге (осциляції при n=20)")
plt.legend(); plt.grid(True)

# --- ГРАФІК 2: Дослідження похибок (Пункт 5 основної частини) ---
plt.subplot(2, 2, 2)
for n in [5, 10, 20]:
    x_n = np.linspace(50, 800, n)
    y_n = 0.25 * x_n + 10 + np.sin(x_n/40) * 7
    c_n = get_divided_diff_table(x_n, y_n)[0, :]
    y_true = 0.25 * x_fine + 10 + np.sin(x_fine/40) * 7
    y_interp = np.array([newton_poly(x_n, c_n, xi) for xi in x_fine])
    plt.plot(x_fine, np.abs(y_true - y_interp), label=f'Похибка n={n}')
plt.yscale('log')
plt.title("2. Графік похибок (логарифмічна шкала)")
plt.legend(); plt.grid(True)

# --- ГРАФІК 3: Порівняння Ньютона та Лагранжа (Пункт 4 досліджень) ---
plt.subplot(2, 2, 3)
y_newt = [newton_poly(x_data, coeffs, xi) for xi in x_fine]
y_lagr = [lagrange_poly(x_data, y_data, xi) for xi in x_fine]
plt.plot(x_fine, y_newt, 'b-', lw=4, alpha=0.4, label='Ньютон')
plt.plot(x_fine, y_lagr, 'r--', lw=1, label='Лагранж')
plt.title("3. Порівняння: Ньютон vs Лагранж")
plt.legend(); plt.grid(True)

# --- ГРАФІК 4: Вплив кроку (Пункт 1 досліджень) ---
plt.subplot(2, 2, 4)
x_fixed_range = np.linspace(50, 800, 500)
for step in [150, 75]: # Різні кроки
    x_s = np.arange(50, 801, step)
    y_s = 0.25 * x_s + 10
    c_s = get_divided_diff_table(x_s, y_s)[0, :]
    plt.plot(x_fixed_range, [newton_poly(x_s, c_s, xi) for xi in x_fixed_range], label=f'Крок {step}')
plt.title("4. Вплив величини кроку")
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()
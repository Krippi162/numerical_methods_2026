import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# Створення файлу даних
def ensure_data_file(filename):
    if not os.path.exists(filename):
        data = [
            ['Month', 'Temp'],
            [1, -2], [2, 0], [3, 5], [4, 10], [5, 15], [6, 20], [7, 23], 
            [8, 22], [9, 17], [10, 10], [11, 5], [12, 0], [13, -10], 
            [14, 3], [15, 7], [16, 13], [17, 19], [18, 20], [19, 22], 
            [20, 21], [21, 18], [22, 15], [23, 10], [24, 3]
        ]
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)

# Метод найменших квадратів та Гаусса
def get_lsq_coeffs(x, y, m):
    n = len(x)
    A_mat = np.zeros((m + 1, m + 1))
    b_vec = np.zeros(m + 1)
    for k in range(m + 1):
        for l in range(m + 1):
            A_mat[k, l] = np.sum(x**(k + l))
        b_vec[k] = np.sum(y * (x**k))
    
    Ab = np.column_stack((A_mat, b_vec))
    for k in range(m + 1):
        max_row = k + np.argmax(np.abs(Ab[k:, k]))
        Ab[[k, max_row]] = Ab[[max_row, k]]
        for i in range(k + 1, m + 1):
            factor = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= factor * Ab[k, k:]
    
    coeffs = np.zeros(m + 1)
    for i in range(m, -1, -1):
        coeffs[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:m+1], coeffs[i+1:])) / Ab[i, i]
    return coeffs

def poly_val(x, coeffs):
    res = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coeffs):
        res += c * (x**i)
    return res

# --- Основна логіка ---
filename = 'data.csv'
ensure_data_file(filename)
with open(filename, 'r') as f:
    reader = csv.DictReader(f)
    x_data = np.array([float(r['Month']) for r in reader])
    f.seek(0)
    y_data = np.array([float(r['Temp']) for r in list(csv.DictReader(f))])

# Вибір оптимального m
best_m, min_var = 1, float('inf')
for m in range(1, 11):
    c = get_lsq_coeffs(x_data, y_data, m)
    var = np.sqrt(np.sum((poly_val(x_data, c) - y_data)**2) / len(x_data))
    if var < min_var:
        min_var, best_m = var, m

best_coeffs = get_lsq_coeffs(x_data, y_data, best_m)
y_approx = poly_val(x_data, best_coeffs)
x_future = np.array([25, 26, 27])
y_future = poly_val(x_future, best_coeffs)

# --- Побудова трьох графіків ---

# 1. Графік апроксимації та фактичних даних 
plt.figure(figsize=(10, 5))
plt.plot(x_data, y_data, 'ro', label='Фактичні дані')
x_smooth = np.linspace(1, 27, 200)
plt.plot(x_smooth, poly_val(x_smooth, best_coeffs), label=f'Апроксимація (m={best_m})')
plt.plot(x_future, y_future, 'bx', label='Прогноз')
plt.title('Апроксимація та екстраполяція температури')
plt.legend(); plt.grid()


# 2. Графік похибки апроксимації 
plt.figure(figsize=(10, 5))
plt.bar(x_data, np.abs(y_data - y_approx), color='gray', label='Похибка ε(x) = |f(x) - φ(x)|')
plt.title('Графік похибки апроксимації')
plt.xlabel('Місяць')
plt.legend(); plt.grid()


# 3. Графік залежності дисперсії від степеня
plt.figure(figsize=(10, 5))
degrees = range(1, 11)
vars_list = [np.sqrt(np.sum((poly_val(x_data, get_lsq_coeffs(x_data, y_data, m)) - y_data)**2) / len(x_data)) for m in degrees]
plt.plot(degrees, vars_list, 'o-')
plt.title('Залежність дисперсії від степеня полінома')
plt.xlabel('Степінь m'); plt.ylabel('Дисперсія')
plt.grid()


plt.show()
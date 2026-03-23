import os
import csv
import matplotlib.pyplot as plt

# Встановлюємо стиль для гарних графіків
plt.style.use('seaborn-v0_8-muted')

# лінійна інтерполяція для знаходження y між вузлами
def get_y_true(x_val, x_nodes, y_nodes):
    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= x_val <= x_nodes[i+1]:
            return y_nodes[i] + (y_nodes[i+1] - y_nodes[i]) * (x_val - x_nodes[i]) / (x_nodes[i+1] - x_nodes[i])
    return y_nodes[-1]

# зчитування даних з csv
def read_data(filename):
    x, y = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # пропускаємо заголовок
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return x, y

# формування матриці a
def form_matrix(x, m):
    a = [[0.0] * (m + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(m + 1):
            a[i][j] = sum(xi**(i+j) for xi in x)
    return a

# формування вектора вільних членів b
def form_vector(x, y, m):
    b = [0.0] * (m + 1)
    for i in range(m + 1):
        b[i] = sum(y[k] * (x[k]**i) for k in range(len(x)))
    return b

# метод гауса з вибором головного елемента
def gauss_solve(a, b):
    n = len(a)
    a_copy = [row[:] for row in a]
    b_copy = b[:]
    
    for k in range(n - 1):
        max_row = k
        for i in range(k + 1, n):
            if abs(a_copy[i][k]) > abs(a_copy[max_row][k]):
                max_row = i
        
        a_copy[k], a_copy[max_row] = a_copy[max_row], a_copy[k]
        b_copy[k], b_copy[max_row] = b_copy[max_row], b_copy[k]
        
        for i in range(k + 1, n):
            if a_copy[k][k] == 0: continue
            factor = a_copy[i][k] / a_copy[k][k]
            for j in range(k, n):
                a_copy[i][j] -= factor * a_copy[k][j]
            b_copy[i] -= factor * b_copy[k]

    x_sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(a_copy[i][j] * x_sol[j] for j in range(i + 1, n))
        x_sol[i] = (b_copy[i] - s) / a_copy[i][i]
    return x_sol

# обчислення значень полінома
def polynomial(x_vals, coef):
    return [sum(coef[i] * (xv**i) for i in range(len(coef))) for xv in x_vals]

# обчислення дисперсії
def calculate_variance(y_true, y_approx):
    n = len(y_true)
    return sum((y_true[i] - y_approx[i])**2 for i in range(n)) / n

# головний блок програми
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'data.csv')

x, y = read_data(data_path)

variances = []
max_degree = 10 
n_nodes = len(x)

for m in range(1, max_degree + 1):
    a_mat = form_matrix(x, m)
    b_vec = form_vector(x, y, m)
    coef = gauss_solve(a_mat, b_vec)
    y_approx = polynomial(x, coef)
    var = calculate_variance(y, y_approx)
    variances.append(var)

optimal_m = variances.index(min(variances)) + 1

a_opt = form_matrix(x, optimal_m)
b_opt = form_vector(x, y, optimal_m)
coef_opt = gauss_solve(a_opt, b_opt)

x_smooth = [x[0] + i * (x[-1] - x[0]) / 200 for i in range(201)]
y_smooth = polynomial(x_smooth, coef_opt)

x_future = [25, 26, 27]
y_future = polynomial(x_future, coef_opt)

h1 = (x[-1] - x[0]) / (20 * n_nodes)
x_err = []
curr_x = x[0]
while curr_x <= x[-1]:
    x_err.append(curr_x)
    curr_x += h1

# ПОБУДОВА ГРАФІКІВ

# Графік 1: Дисперсія
plt.figure(1, figsize=(10, 5))
plt.plot(range(1, max_degree + 1), variances, color='#4A90E2', marker='o', markersize=8, linewidth=2)
plt.axvline(x=optimal_m, color='#E94E77', linestyle='--', label=f'Оптимальне m={optimal_m}')
plt.fill_between(range(1, max_degree + 1), variances, color='#4A90E2', alpha=0.1)
plt.title("Залежність дисперсії від ступеня полінома", fontsize=14, pad=15)
plt.xlabel("Ступінь (m)", fontweight='bold')
plt.ylabel("Дисперсія", fontweight='bold')
plt.legend(frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Графік 2: Апроксимація та прогноз
plt.figure(2, figsize=(10, 6))
plt.scatter(x, y, color='#333333', alpha=0.5, label='Фактичні дані', s=40)
plt.plot(x_smooth, y_smooth, color='#4A90E2', linewidth=2.5, label=f'Апроксимація (m={optimal_m})')
plt.plot(x_future, y_future, color='#E94E77', linestyle='--', marker='s', markersize=6, linewidth=2, label='Прогноз')
plt.title("Апроксимація та прогноз температури", fontsize=14, pad=15)
plt.xlabel("Місяць", fontweight='bold')
plt.ylabel("Температура", fontweight='bold')
plt.legend(facecolor='white', framealpha=1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Графік 3: Похибка
plt.figure(3, figsize=(10, 6))
for m in range(1, max_degree + 1):
    a_mat = form_matrix(x, m)
    b_vec = form_vector(x, y, m)
    c_m = gauss_solve(a_mat, b_vec)
    y_approx_err = polynomial(x_err, c_m)
    y_true_err = [get_y_true(xv, x, y) for xv in x_err]
    error_vals = [abs(y_true_err[i] - y_approx_err[i]) for i in range(len(x_err))]
    
    if m == optimal_m:
        plt.plot(x_err, error_vals, color='#D0021B', linewidth=3, label=f'm={m} (Оптимальна)', zorder=10)
    else:
        plt.plot(x_err, error_vals, alpha=0.2, color='#9B9B9B')

plt.title("Розподіл абсолютної похибки", fontsize=14, pad=15)
plt.xlabel("Місяць", fontweight='bold')
plt.ylabel("Абсолютна похибка", fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, linestyle='-', alpha=0.3)
plt.tight_layout()

plt.show()
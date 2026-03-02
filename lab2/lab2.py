import numpy as np
import matplotlib.pyplot as plt

x_exp = np.array([50, 100, 200, 400, 800])
y_exp = np.array([20, 35, 60, 110, 210])

def target_function(x):
    return 0.25 * x + 7.5 + 5 * np.exp(-x/200)

# --- МЕТОД НЬЮТОНА ---
def get_divided_diff(x, y):
    n = len(y)
    coef = np.zeros([n, n]); coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
    return coef[0, :]

def newton_poly(coef, x_data, x):
    n = len(x_data) - 1; p = coef[n]
    for k in range(1, n + 1):
        p = coef[n-k] + (x - x_data[n-k]) * p
    return p

# --- ОБЧИСЛЕННЯ ТА ВІЗУАЛІЗАЦІЯ ---
x_plot = np.linspace(50, 800, 500)
target_x = 600
n_values = [5, 10, 20]
colors = ['#1f77b4', '#2ca02c', '#ff7f0e'] 

fig, axes = plt.subplots(4, 1, figsize=(9, 12))
results = []

for i, n in enumerate(n_values):
    ax = axes[i]
    
    # Створення вузлів для дослідження
    x_n = np.linspace(50, 800, n)
    y_n = target_function(x_n)
    
    coefs = get_divided_diff(x_n, y_n)
    y_interp = np.array([newton_poly(coefs, x_n, xi) for xi in x_plot])
    
    # Розрахунок похибки для фінального графіка
    error = np.abs(target_function(x_plot) - y_interp)
    results.append((n, error))
    
    # Побудова графіка моделі
    ax.plot(x_plot, y_interp, color=colors[i], label=f'Newton n={n}')
    ax.scatter(x_n, y_n, color='red', s=20, edgecolors='black', label=f'Вузли n={n}', zorder=3)
    
    # Прогноз для 600 RPS
    pred = newton_poly(coefs, x_n, target_x)
    ax.plot(target_x, pred, 'ko', markersize=5)
    
    # Анотація: текст зміщено вбік і вгору, щоб нічого не заступало
    ax.annotate(f'{pred:.1f}%', (target_x, pred), xytext=(10, 10), 
                textcoords='offset points', fontsize=9, fontweight='bold')
    
    ax.set_title(f"Інтерполяція при n = {n}", fontsize=10, pad=5)
    ax.set_ylabel("CPU %", fontsize=9)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)

# --- СПІЛЬНИЙ ГРАФІК ПОХИБОК ---
ax_err = axes[3]
for i, (n, error) in enumerate(results):
    ax_err.plot(x_plot, error, color=colors[i], label=f'n={n}')

ax_err.set_yscale('log')
ax_err.set_title("Порівняння похибок моделей (log scale)", fontsize=10)
ax_err.set_xlabel("RPS", fontsize=9)
ax_err.set_ylabel("Error", fontsize=9)
ax_err.legend(loc='lower left', fontsize=8, ncol=3)
ax_err.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.show()

c5 = get_divided_diff(x_exp, y_exp)
print(f"Результат прогнозу для 600 RPS: {newton_poly(c5, x_exp, 600):.2f}%")
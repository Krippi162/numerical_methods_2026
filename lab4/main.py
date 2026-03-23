import numpy as np
import matplotlib.pyplot as plt

# 1. Визначення функції та її аналітичної похідної
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def M_prime_exact(t):
    # Похідна: M'(t) = 50 * (-0.1) * e^(-0.1t) + 5 * cos(t)
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

# Точка обчислення (за прикладом t0 = 1)
t0 = 1.0
exact_val = M_prime_exact(t0)

# 2. Дослідження залежності похибки від кроку h 
def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

h_values = np.logspace(-20, 3, num=100)
errors = []

for h in h_values:
    approx = central_diff(M, t0, h)
    errors.append(abs(approx - exact_val))

# Пошук оптимального h0
min_error_idx = np.argmin(errors)
h0 = h_values[min_error_idx]
R0 = errors[min_error_idx]

print(f"Точне значення M'({t0}): {exact_val:.10f}")
print(f"Оптимальний крок h0: {h0:.2e}")
print(f"Найкраща точність R0: {R0:.2e}\n")

# 3-6. Метод Рунге-Ромберга
h_fixed = 1e-3
y_h = central_diff(M, t0, h_fixed)
y_2h = central_diff(M, t0, 2 * h_fixed)

R1 = abs(y_h - exact_val)

# Формула Рунге-Ромберга 
y_RR = y_h + (y_h - y_2h) / 3
R2 = abs(y_RR - exact_val)


print(f"y'(h)  при h={h_fixed}: {y_h:.10f}, Похибка R1: {R1:.2e}")
print(f"y'(2h) при h={2*h_fixed}: {y_2h:.10f}")
print(f"Уточнене y_RR: {y_RR:.10f}, Похибка R2: {R2:.2e}")
print(f"Характер зміни: Похибка зменшилась у {R1/R2:.2f} разів\n")

# 7. Метод Ейткена
y_4h = central_diff(M, t0, 4 * h_fixed)

# Формула Ейткена
numerator = (y_2h**2) - (y_4h * y_h)
denominator = 2 * y_2h - (y_4h + y_h)
y_E = numerator / denominator

# Порядок точності p
p = (1 / np.log(2)) * np.log(abs((y_4h - y_2h) / (y_2h - y_h)))
R3 = abs(y_E - exact_val)

print(f"y'(4h) при h={4*h_fixed}: {y_4h:.10f}")
print(f"Уточнене y_E: {y_E:.10f}, Похибка R3: {R3:.2e}")
print(f"Оцінений порядок точності p: {p:.2f}")

plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors, label='Залежність R(h)')
plt.scatter(h0, R0, color='red', label=f'Оптимальне h0={h0:.1e}')
plt.xlabel('Крок h')
plt.ylabel('Похибка R')
plt.title("Дослідження похибки чисельного диференціювання")
plt.grid(True, which="both", ls="-")
plt.legend()
plt.show()
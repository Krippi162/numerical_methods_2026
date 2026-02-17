import requests
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Запит до API (Виправлено URL)
# -------------------------------
url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

try:
    response = requests.get(url)
    data = response.json()
    results = data["results"]
except Exception as e:
    print("Помилка при запиті до API:", e)
    results = []

if results:
    # -------------------------------
    # 2. Розрахунок відстаней (Haversine)
    # -------------------------------
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    n = len(results)
    coords = [(p["latitude"], p["longitude"]) for p in results]
    elevations = [p["elevation"] for p in results]

    distances = [0]
    for i in range(1, n):
        d = haversine(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])
        distances.append(distances[-1] + d)

    x = np.array(distances)
    y = np.array(elevations)

    # -------------------------------
    # 3. Розрахунок кубічних сплайнів (Метод прогонки)
    # -------------------------------
    def solve_spline(x, y):
        n = len(x) - 1
        h = np.diff(x)
        
        # Система для коефіцієнтів c
        A = np.zeros(n + 1)
        B = np.zeros(n + 1)
        C = np.zeros(n + 1)
        D = np.zeros(n + 1)
        
        # Природні граничні умови (c0 = cn = 0)
        B[0] = 1
        B[n] = 1
        
        for i in range(1, n):
            A[i] = h[i-1]
            B[i] = 2 * (h[i-1] + h[i])
            C[i] = h[i]
            D[i] = 3 * ((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
        
        # Метод прогонки
        c = np.zeros(n + 1)
        alpha = np.zeros(n)
        beta = np.zeros(n)
        
        for i in range(n):
            m = A[i] * alpha[i-1] + B[i]
            alpha[i] = -C[i] / m
            beta[i] = (D[i] - A[i] * beta[i-1]) / m
        
        c[n] = (D[n] - A[n] * beta[n-1]) / (A[n] * alpha[n-1] + B[n])
        for i in range(n-1, -1, -1):
            c[i] = alpha[i] * c[i+1] + beta[i]
            
        # Обчислення a, b, d
        a = y[:-1]
        d = (c[1:] - c[:-1]) / (3 * h)
        b = (y[1:] - y[:-1]) / h - h * (c[1:] + 2 * c[:-1]) / 3
        return a, b, c[:-1], d

    a_coeff, b_coeff, c_coeff, d_coeff = solve_spline(x, y)

    # -------------------------------
    # 4. Побудова графіка
    # -------------------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='red', label='GPS вузли')
    
    for i in range(n-1):
        x_seg = np.linspace(x[i], x[i+1], 50)
        dx = x_seg - x[i]
        y_seg = a_coeff[i] + b_coeff[i]*dx + c_coeff[i]*dx**2 + d_coeff[i]*dx**3
        plt.plot(x_seg, y_seg, 'b' if i==0 else 'b')

    plt.title("Профіль висот: Заросляк - Говерла (Кубічні сплайни)")
    plt.xlabel("Відстань (м)")
    plt.ylabel("Висота (м)")
    plt.grid(True)
    plt.legend()
    
    # -------------------------------
    # 5. Додаткові характеристики (Пункт 13)
    # -------------------------------
    total_ascent = sum(max(y[i] - y[i-1], 0) for i in range(1, n))
    energy = 80 * 9.81 * total_ascent
    # ==========================================
# ДОДАТКОВІ ЗАВДАННЯ (Пункт 1, 2, 3)
# ==========================================

# 1. Характеристики маршруту
print("\n--- ХАРАКТЕРИСТИКИ МАРШРУТУ ---")
print(f"Загальна довжина маршруту (м): {distances[-1]:.2f}")

total_ascent = sum(max(elevations[i] - elevations[i-1], 0) for i in range(1, n))
print(f"Сумарний набір висоти (м): {total_ascent:.2f}")

total_descent = sum(max(elevations[i-1] - elevations[i], 0) for i in range(1, n))
print(f"Сумарний спуск (м): {total_descent:.2f}")


# 2. Аналіз градієнта (через похідну сплайна)
# Створюємо густу сітку точок для аналізу (1000 точок на весь шлях)
xx = np.linspace(distances[0], distances[-1], 1000)
yy_full = []

# Обчислюємо значення сплайна для кожної точки сітки
for val in xx:
    # Знаходимо, якому інтервалу належить точка
    idx = np.searchsorted(distances, val) - 1
    idx = max(0, min(idx, len(a_coeff)-1))
    dx = val - distances[idx]
    y_val = a_coeff[idx] + b_coeff[idx]*dx + c_coeff[idx]*dx**2 + d_coeff[idx]*dx**3
    yy_full.append(y_val)

yy_full = np.array(yy_full)

# Розрахунок градієнта (нахилу) у відсотках
# np.gradient обчислює різницю висот між точками сітки
dx_step = xx[1] - xx[0]
grad_full = np.gradient(yy_full, dx_step) * 100

print(f"\n--- АНАЛІЗ ГРАДІЄНТА ---")
print(f"Максимальний підйом (%): {np.max(grad_full):.2f}")
print(f"Максимальний спуск (%): {np.min(grad_full):.2f}")
print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}")

# Ділянки з крутизною > 15%
steep_sections = np.where(np.abs(grad_full) > 15)[0]
if len(steep_sections) > 0:
    print(f"Кількість складних ділянок (крутизна > 15%): {len(steep_sections)}")
else:
    print("Ділянок з крутизною > 15% не виявлено.")


# 3. Механічна енергія підйому
mass = 80
g = 9.81
energy = mass * g * total_ascent

print(f"\n--- ЕНЕРГЕТИЧНІ ВИТРАТИ ---")
print(f"Механічна робота (Дж): {energy:.2f}")
print(f"Механічна робота (кДж): {energy/1000:.2f}")
print(f"Орієнтовні витрати калорій (ккал): {energy/4184:.2f}")
plt.show()
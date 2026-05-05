import os
import numpy as np
import matplotlib.pyplot as plt
import cmath

current_dir = os.path.dirname(os.path.abspath(__file__))

# Шляхи до файлів
tab_file = os.path.join(current_dir, "tabulation.txt")
coeffs_file = os.path.join(current_dir, "poly_coeffs.txt")
graph_file = os.path.join(current_dir, "algebraic_graph.png")



def F(x):
    return np.sin(x) - 0.2 * x

def dF(x):
    return np.cos(x) - 0.2

def d2F(x):
    return -np.sin(x)

def tabulate_function(a, b, h=0.1, filename=tab_file):
    x_vals = np.arange(a, b + h, h)
    y_vals = F(x_vals)
    
    with open(filename, 'w') as f:
        f.write("x\t\tF(x)\n")
        for x, y in zip(x_vals, y_vals):
            f.write(f"{x:.4f}\t{y:.4f}\n")
            
    roots_approx = []
    for i in range(len(y_vals) - 1):
        if y_vals[i] * y_vals[i+1] < 0:
            roots_approx.append((x_vals[i] + x_vals[i+1]) / 2)
            
    return roots_approx

eps = 1e-10

def simple_iteration(x0, tau=-0.5):
    x = x0
    iters = 0
    while True:
        iters += 1
        x_new = x + tau * F(x)
        if abs(F(x_new)) < eps and abs(x_new - x) < eps:
            break
        x = x_new
        if iters > 10000: break
    return x_new, iters

def newton_method(x0):
    x = x0
    iters = 0
    while True:
        iters += 1
        x_new = x - F(x) / dF(x)
        if abs(F(x_new)) < eps and abs(x_new - x) < eps:
            break
        x = x_new
    return x_new, iters

def chebyshev_method(x0):
    x = x0
    iters = 0
    while True:
        iters += 1
        fx = F(x)
        dfx = dF(x)
        d2fx = d2F(x)
        x_new = x - fx / dfx - 0.5 * (fx**2 * d2fx) / (dfx**3)
        if abs(F(x_new)) < eps and abs(x_new - x) < eps:
            break
        x = x_new
    return x_new, iters

def secant_method(x0, x1):
    iters = 0
    while True:
        iters += 1
        f_x1 = F(x1)
        f_x0 = F(x0)
        x_new = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        if abs(F(x_new)) < eps and abs(x_new - x1) < eps:
            break
        x0, x1 = x1, x_new
    return x_new, iters

def divided_diff(x0, x1, x2=None):
    if x2 is None:
        return (F(x1) - F(x0)) / (x1 - x0)
    return (divided_diff(x1, x2) - divided_diff(x0, x1)) / (x2 - x0)

def parabola_method(x0, x1, x2):
    iters = 0
    while True:
        iters += 1
        f_x2 = F(x2)
        f_x2_x1 = divided_diff(x1, x2)
        f_x2_x1_x0 = divided_diff(x0, x1, x2)
        B = f_x2_x1 + f_x2_x1_x0 * (x2 - x1)
        C = f_x2
        A = f_x2_x1_x0
        sqrt_D = np.sqrt(B**2 - 4*A*C + 0j) 
        delta1 = (-B + sqrt_D) / (2*A)
        delta2 = (-B - sqrt_D) / (2*A)
        delta = delta1 if abs(delta1) < abs(delta2) else delta2
        x_new = x2 + delta.real 
        if abs(F(x_new)) < eps and abs(x_new - x2) < eps:
            break
        x0, x1, x2 = x1, x2, x_new
    return x_new, iters

def inverse_interpolation_3pts(x0, x1, x2):
    iters = 0
    while True:
        iters += 1
        y0, y1, y2 = F(x0), F(x1), F(x2)
        t1 = (y1 * y2) / ((y0 - y1) * (y0 - y2)) * x0
        t2 = (y0 * y2) / ((y1 - y0) * (y1 - y2)) * x1
        t3 = (y0 * y1) / ((y2 - y0) * (y2 - y1)) * x2
        x_new = t1 + t2 + t3
        if abs(F(x_new)) < eps and abs(x_new - x2) < eps:
            break
        x0, x1, x2 = x1, x2, x_new
    return x_new, iters

approx_roots = tabulate_function(-1, 3.5, 0.1)
print(f"Корені з табуляції: {approx_roots}")

for r_approx in approx_roots:
    print(f"\n--- Корені біля x = {r_approx:.4f} ---")
    res_si, it_si = simple_iteration(r_approx, tau=0.5 if dF(r_approx) < 0 else -0.5)
    print(f"Проста ітерація:\t x = {res_si:.10f}, it = {it_si}")
    res_n, it_n = newton_method(r_approx)
    print(f"Метод Ньютона:\t\t x = {res_n:.10f}, it = {it_n}")
    res_ch, it_ch = chebyshev_method(r_approx)
    print(f"Метод Чебишева:\t\t x = {res_ch:.10f}, it = {it_ch}")
    res_s, it_s = secant_method(r_approx - 0.1, r_approx)
    print(f"Метод хорд:\t\t x = {res_s:.10f}, it = {it_s}")
    res_p, it_p = parabola_method(r_approx - 0.2, r_approx - 0.1, r_approx)
    print(f"Метод парабол:\t\t x = {res_p:.10f}, it = {it_p}")
    res_i, it_i = inverse_interpolation_3pts(r_approx - 0.2, r_approx - 0.1, r_approx)
    print(f"Зворотна інтерп.:\t x = {res_i:.10f}, it = {it_i}")


def plot_algebraic():
    x_v = np.linspace(-1, 3, 400)
    y_v = x_v**3 - 2*x_v**2 + x_v - 2
    plt.figure(figsize=(8, 5))
    plt.plot(x_v, y_v, label='$F(x) = x^3 - 2x^2 + x - 2$')
    plt.axhline(0, color='black', lw=1); plt.axvline(0, color='black', lw=1)
    plt.plot(2, 0, 'ro', label='Корінь x=2')
    plt.grid(True, ls='--'); plt.legend(); plt.savefig(graph_file); plt.close()
    print(f"Графік збережено: {graph_file}")

plot_algebraic()

with open(coeffs_file, 'w') as f: f.write("1 -2 1 -2\n")

def read_coeffs(filename):
    with open(filename, 'r') as f:
        return [float(c) for c in f.readline().strip().split()]

a_c = read_coeffs(coeffs_file)

def newton_horner(coeffs, x0, eps=1e-10):
    x = x0; iters = 0; m = len(coeffs) - 1
    while True:
        iters += 1
        b = [0]*(m+1); b[0] = coeffs[0]
        for i in range(1, m+1): b[i] = coeffs[i] + x*b[i-1]
        c = [0]*m; c[0] = b[0]
        for i in range(1, m): c[i] = b[i] + x*c[i-1]
        x_new = x - b[m]/c[m-1]
        if abs(b[m]) < eps and abs(x_new - x) < eps: break
        x = x_new
    return x_new, iters

real_r, it_nh = newton_horner(a_c, 2.5)
print(f"Дійсний корінь (Горнер): x = {real_r:.10f}, it = {it_nh}")

def lin_method(coeffs, a0, b0, eps=1e-6):
    a_l, b_l = a0, b0; iters = 0; m = len(coeffs) - 1
    a_rev = list(reversed(coeffs))
    while True:
        iters += 1
        p0, q0 = -2*a_l, a_l**2 + b_l**2
        b = [0]*(m+1); b[m] = a_rev[m]; b[m-1] = a_rev[m-1] - p0*b[m]
        for i in range(m-2, 1, -1): b[i] = a_rev[i] - p0*b[i+1] - q0*b[i+2]
        if abs(b[2]) < 1e-12: break
        q1 = a_rev[0]/b[2]; p1 = (a_rev[1]*b[2] - a_rev[0]*b[3])/(b[2]**2)
        a1 = -p1/2; v = q1 - a1**2
        b1 = np.sqrt(abs(v))
        if abs(a1 - a_l) <= eps and abs(b1 - b_l) <= eps: break
        a_l, b_l = a1, b1
        if iters > 1000: break
    return complex(a_l, b_l), complex(a_l, -b_l), iters

c1, c2, it_l = lin_method(a_c, 0.1, 0.9)
print(f"Комплексні корені (Лін):\n z1 = {c1:.6f}, z2 = {c2:.6f}, it = {it_l}")
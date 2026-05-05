import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x) - 0.5 * x

def df(x):
    return np.cos(x) - 0.5

def d2f(x):
    return -np.sin(x)

EPS = 1e-8
MAX_ITER = 1000


def tabulate(a, b, h=0.1):
    xs = np.arange(a, b + h, h)
    ys = f(xs)
    with open("tabulation.txt", "w") as fout:
        fout.write(f"{'x':>10}  {'f(x)':>14}\n")
        for x, y in zip(xs, ys):
            fout.write(f"{x:10.4f}  {y:14.8f}\n")
    roots_approx = []
    for i in range(len(ys) - 1):
        if ys[i] * ys[i+1] < 0:
            roots_approx.append((xs[i] + xs[i+1]) / 2)
    return xs, ys, roots_approx


def simple_iteration(x0, phi, label=""):
    x = x0
    for i in range(1, MAX_ITER + 1):
        xn = phi(x)
        if abs(xn - x) < EPS and abs(f(xn)) < EPS:
            print(f"[SimpleIter {label}] root={xn:.10f}  iters={i}")
            return xn, i
        x = xn
    print(f"[SimpleIter {label}] не збіглось")
    return x, MAX_ITER


def newton(x0, label=""):
    x = x0
    for i in range(1, MAX_ITER + 1):
        xn = x - f(x) / df(x)
        if abs(xn - x) < EPS and abs(f(xn)) < EPS:
            print(f"[Newton    {label}] root={xn:.10f}  iters={i}")
            return xn, i
        x = xn
    print(f"[Newton {label}] не збіглось")
    return x, MAX_ITER


def chebyshev(x0, label=""):
    x = x0
    for i in range(1, MAX_ITER + 1):
        fx, dfx, d2fx = f(x), df(x), d2f(x)
        xn = x - fx/dfx - (fx**2 * d2fx) / (2 * dfx**3)
        if abs(xn - x) < EPS and abs(f(xn)) < EPS:
            print(f"[Chebyshev {label}] root={xn:.10f}  iters={i}")
            return xn, i
        x = xn
    print(f"[Chebyshev {label}] не збіглось")
    return x, MAX_ITER


def chord(x0, x1, label=""):
    for i in range(1, MAX_ITER + 1):
        fx0, fx1 = f(x0), f(x1)
        xn = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        if abs(xn - x1) < EPS and abs(f(xn)) < EPS:
            print(f"[Chord     {label}] root={xn:.10f}  iters={i}")
            return xn, i
        x0, x1 = x1, xn
    print(f"[Chord {label}] не збіглось")
    return x1, MAX_ITER


def parabola(x0, x1, x2, label=""):
    """Метод Мюллера (метод парабол)."""
    for i in range(1, MAX_ITER + 1):
        f0, f1, f2 = f(x0), f(x1), f(x2)
        h1, h2 = x1 - x0, x2 - x1
        d1 = (f1 - f0) / h1
        d2 = (f2 - f1) / h2
        a = (d2 - d1) / (h1 + h2)
        b = a * h2 + d2
        c = f2
        disc = b**2 - 4*a*c
        if disc < 0:
            disc = 0.0
        s = np.sqrt(disc)
        den = b + s if abs(b + s) >= abs(b - s) else b - s
        dx = -2*c / den
        xn = x2 + dx
        if abs(dx) < EPS and abs(f(xn)) < EPS:
            print(f"[Parabola  {label}] root={xn:.10f}  iters={i}")
            return xn, i
        x0, x1, x2 = x1, x2, xn
    print(f"[Parabola {label}] не збіглось")
    return x2, MAX_ITER


def reverse_interpolation(x0, x1, x2, label=""):
    """Метод зворотної інтерполяції — поліном Лагранжа по 3 вузлах."""
    for i in range(1, MAX_ITER + 1):
        f0, f1, f2 = f(x0), f(x1), f(x2)
        xn = (x0*f1*f2 / ((f0-f1)*(f0-f2)) +
              x1*f0*f2 / ((f1-f0)*(f1-f2)) +
              x2*f0*f1 / ((f2-f0)*(f2-f1)))
        if abs(xn - x2) < EPS and abs(f(xn)) < EPS:
            print(f"[RevInterp {label}] root={xn:.10f}  iters={i}")
            return xn, i
        x0, x1, x2 = x1, x2, xn
    print(f"[RevInterp {label}] не збіглось")
    return x2, MAX_ITER


def horner(coeffs, x):
    """Повертає (P(x), P'(x))."""
    n = len(coeffs) - 1
    b = coeffs[0]
    c = coeffs[0]
    for i in range(1, n):
        b = b * x + coeffs[i]
        c = c * x + b
    b = b * x + coeffs[n]
    return b, c


def newton_horner(coeffs, x0, label=""):
    x = float(x0)
    for i in range(1, MAX_ITER + 1):
        px, dpx = horner(coeffs, x)
        xn = x - px / dpx
        if abs(xn - x) < EPS and abs(horner(coeffs, xn)[0]) < EPS:
            print(f"[NewtonHorner {label}] root={xn:.10f}  iters={i}")
            return xn, i
        x = xn
    print(f"[NewtonHorner {label}] не збіглось")
    return x, MAX_ITER

def lin_bairstow(coeffs, p0, q0):
    """
    Знаходить квадратний множник x^2 + px + q.
    Корені = -p/2 ± sqrt(p^2/4 - q).
    """
    a = list(map(float, coeffs))
    n = len(a) - 1
    p, q = float(p0), float(q0)

    for it in range(MAX_ITER):
        b = [0.0] * (n + 1)
        b[0] = a[0]
        b[1] = a[1] - p * b[0]
        for k in range(2, n + 1):
            b[k] = a[k] - p*b[k-1] - q*b[k-2]

        c = [0.0] * (n + 1)
        c[0] = b[0]
        c[1] = b[1] - p * c[0]
        for k in range(2, n + 1):
            c[k] = b[k] - p*c[k-1] - q*c[k-2]

        J = np.array([[c[n-2], c[n-3]],
                      [c[n-1], c[n-2]]])
        rhs = np.array([b[n-1], b[n]])

        try:
            dp, dq = np.linalg.solve(J, rhs)
        except np.linalg.LinAlgError:
            break

        p += dp
        q += dq

        if abs(dp) < EPS and abs(dq) < EPS:
            disc = p**2 - 4*q
            if disc < 0:
                re = -p / 2
                im = np.sqrt(-disc) / 2
                print(f"[Lin/Bairstow] iters={it+1}  roots = {re:.8f} ± {im:.8f}i")
            else:
                r1 = (-p + np.sqrt(disc)) / 2
                r2 = (-p - np.sqrt(disc)) / 2
                print(f"[Lin/Bairstow] iters={it+1}  roots = {r1:.8f}, {r2:.8f}")
            return p, q, it + 1

    print("[Lin/Bairstow] не збіглось")
    return p, q, MAX_ITER

if __name__ == "__main__":

    xs, ys, roots_approx = tabulate(-10, 10, 0.1)
    print("Наближені корені:", [f"{r:.2f}" for r in roots_approx])
    print()

    phi = lambda x: 2 * np.sin(x)   # |phi'|<1 біля ±1.895

    print("─── Корінь ≈ +1.895 (зростання) ───")
    simple_iteration(1.5, phi, "+")
    newton(1.5, "+")
    chebyshev(1.5, "+")
    chord(1.3, 2.0, "+")
    parabola(1.3, 1.6, 2.0, "+")
    reverse_interpolation(1.3, 1.6, 2.0, "+")

    print()
    print("─── Корінь ≈ -1.895 (спадання) ───")
    simple_iteration(-1.5, phi, "-")
    newton(-1.5, "-")
    chebyshev(-1.5, "-")
    chord(-2.0, -1.3, "-")
    parabola(-2.0, -1.6, -1.3, "-")
    reverse_interpolation(-2.0, -1.6, -1.3, "-")

    print()
    print("─── x³ - x² + x - 0.5 = 0 ───")
    poly = [1.0, -1.0, 1.0, -0.5]
    with open("polynomial.txt", "w") as fout:
        fout.write(" ".join(map(str, poly)) + "\n")

    newton_horner(poly, x0=0.5, label="real")
    lin_bairstow(poly, p0=-0.35, q0=0.77)

    # Графік 1 — трансцендентна функція
    x_plot = np.linspace(-10, 10, 1000)
    plt.figure(figsize=(10, 4))
    plt.plot(x_plot, f(x_plot), color="royalblue", label="f(x) = sin(x) − 0.5x")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.scatter(roots_approx, [0]*len(roots_approx), color="red", zorder=5, label="наближені корені")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.title("Трансцендентна функція та її нулі")
    plt.tight_layout()
    plt.savefig("plot.png", dpi=130)
    print("\nГрафік збережено: plot.png")

    # Графік 2 — алгебраїчне рівняння
    x_poly = np.linspace(-1, 2, 500)
    y_poly = np.polyval(poly, x_poly)
    plt.figure(figsize=(8, 5))
    plt.plot(x_poly, y_poly, color="darkorange", label=r"$x^3 - x^2 + x - 0.5$")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.scatter([0.6478], [0], color="red", zorder=5, label="дійсний корінь ≈ 0.648")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.title("Алгебраїчне рівняння: $x^3 - x^2 + x - 0.5 = 0$")
    plt.tight_layout()
    plt.savefig("plot_poly.png", dpi=130)
    print("Графік збережено: plot_poly.png")
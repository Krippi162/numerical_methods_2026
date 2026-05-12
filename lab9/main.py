import numpy as np
import matplotlib.pyplot as plt


def norm(v):
    return float(np.linalg.norm(v))


def exploratory_search(func, base, step, eps1, q, reduce_steps=True):
    x = base.copy()
    n = len(x)

    for i in range(n):
        while True:
            current_value = func(x)

            x_plus = x.copy()
            x_plus[i] += step[i]

            if func(x_plus) < current_value:
                x = x_plus
                break

            x_minus = x.copy()
            x_minus[i] -= step[i]

            if func(x_minus) < current_value:
                x = x_minus
                break

            if reduce_steps:
                step[i] /= q

                if step[i] < eps1:
                    break
            else:
                break

    return x, step


def hooke_jeeves(func, x0, step0, q=2.0, p=2.0, eps1=1e-6, eps2=1e-10, max_iter=10000):
    x0 = np.array(x0, dtype=float)

    if np.iterable(step0):
        step = np.array(step0, dtype=float)
    else:
        step = np.array([step0] * len(x0), dtype=float)

    trajectory = [x0.copy()]
    iterations = 0

    while iterations < max_iter:
        iterations += 1

        f_old = func(x0)

        # Досліджуючий пошук
        x1, step = exploratory_search(
            func=func,
            base=x0,
            step=step.copy(),
            eps1=eps1,
            q=q,
            reduce_steps=True
        )

        f_new = func(x1)

        # Якщо нову точку не знайдено, завершуємо пошук
        if np.allclose(x1, x0, atol=0.0, rtol=0.0):
            break

        trajectory.append(x1.copy())

        # Критерії закінчення пошуку з методички
        if norm(x1 - x0) < eps1 and abs(f_new - f_old) < eps2:
            break

        # Пошук по зразку
        while iterations < max_iter:
            iterations += 1

            xp = x1 + p * (x1 - x0)

            # Досліджуючий пошук з точки xp,
            # але без зменшення величини кроку
            x2, _ = exploratory_search(
                func=func,
                base=xp,
                step=step.copy(),
                eps1=eps1,
                q=q,
                reduce_steps=False
            )

            if func(x2) < func(x1):
                x0 = x1.copy()
                x1 = x2.copy()
                trajectory.append(x1.copy())
            else:
                x0 = x1.copy()
                break

    return x0, func(x0), iterations, np.array(trajectory)


# ---------------------------------------------------------
# Тестова цільова функція з методички: функція Розенброка
# ---------------------------------------------------------

def rosenbrock(X):
    x1, x2 = X
    return 100 * (x1 ** 2 - x2) ** 2 + (x1 - 1) ** 2


# ---------------------------------------------------------
# Задана система нелінійних рівнянь, m = 2
# ---------------------------------------------------------

# f1(x, y) = x^2 + y^2 - 4 = 0
# f2(x, y) = x - y = 0

def f1_system(x, y):
    return x ** 2 + y ** 2 - 4


def f2_system(x, y):
    return x - y


# Цільова функція:
# Phi(x, y) = f1(x, y)^2 + f2(x, y)^2

def phi_system(X):
    x, y = X
    return f1_system(x, y) ** 2 + f2_system(x, y) ** 2


def save_trajectory(filename, trajectory, func):
    with open(filename, "w", encoding="utf-8") as file:
        file.write("k\tx\ty\tPhi(x, y)\n")

        for k, point in enumerate(trajectory):
            file.write(
                f"{k}\t{point[0]:.10f}\t{point[1]:.10f}\t{func(point):.10e}\n"
            )


def plot_system_graph(solution):
    x = np.linspace(-3, 3, 500)

    y_circle_top = np.sqrt(np.maximum(0, 4 - x ** 2))
    y_circle_bottom = -y_circle_top

    plt.figure(figsize=(7, 6))

    plt.plot(x, y_circle_top, label="x^2 + y^2 - 4 = 0")
    plt.plot(x, y_circle_bottom)
    plt.plot(x, x, label="x - y = 0")

    plt.scatter(
        solution[0],
        solution[1],
        s=60,
        label="знайдений розв'язок"
    )

    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Графіки рівнянь системи")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")

    plt.savefig("system_equations.png", dpi=200)
    plt.show()


def plot_objective_and_trajectory(trajectory):
    x = np.linspace(-3, 3, 300)
    y = np.linspace(-3, 3, 300)

    X, Y = np.meshgrid(x, y)

    Z = f1_system(X, Y) ** 2 + f2_system(X, Y) ** 2

    plt.figure(figsize=(7, 6))

    plt.contour(X, Y, Z, levels=30)
    plt.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        marker="o",
        label="траєкторія спуску"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Цільова функція Phi(x, y) та траєкторія спуску")
    plt.grid(True)
    plt.legend()

    plt.savefig("objective_trajectory.png", dpi=200)
    plt.show()


def main():
    print("Лабораторна робота №9")
    print("Метод Хука-Дживса багатовимірної оптимізації\n")

    q = 2.0
    p = 2.0
    eps1 = 1e-6
    eps2 = 1e-10

    print("1) Тестування програми на функції Розенброка")

    x_min_test, f_min_test, it_test, tr_test = hooke_jeeves(
        func=rosenbrock,
        x0=[-1.2, 0.0],
        step0=[0.5, 0.5],
        q=q,
        p=p,
        eps1=eps1,
        eps2=eps2
    )

    print("Початкове наближення: X0 = (-1.2, 0.0)")
    print(f"Знайдений мінімум: X* = ({x_min_test[0]:.8f}, {x_min_test[1]:.8f})")
    print(f"f(X*) = {f_min_test:.10e}")
    print(f"Кількість ітерацій: {it_test}")
    print(f"Кількість точок траєкторії: {len(tr_test)}\n")

    print("2) Розв'язання системи нелінійних рівнянь")
    print("f1(x, y) = x^2 + y^2 - 4 = 0")
    print("f2(x, y) = x - y = 0")
    print("Phi(x, y) = f1(x, y)^2 + f2(x, y)^2\n")

    x_min, phi_min, it, trajectory = hooke_jeeves(
        func=phi_system,
        x0=[1.0, 1.0],
        step0=[0.5, 0.5],
        q=q,
        p=p,
        eps1=eps1,
        eps2=eps2
    )

    print(f"q = {q}")
    print(f"p = {p}")
    print(f"eps1 = {eps1}")
    print(f"eps2 = {eps2}")
    print("Початкове наближення: X0 = (1.0, 1.0)")
    print("Початковий крок: dX = (0.5, 0.5)\n")

    print("Знайдений розв'язок системи:")
    print(f"x = {x_min[0]:.10f}")
    print(f"y = {x_min[1]:.10f}")
    print(f"Phi(x, y) = {phi_min:.10e}")
    print(f"f1(x, y) = {f1_system(x_min[0], x_min[1]):.10e}")
    print(f"f2(x, y) = {f2_system(x_min[0], x_min[1]):.10e}")
    print(f"Кількість кроків на траєкторії спуску: {len(trajectory) - 1}")
    print(f"Кількість ітерацій алгоритму: {it}\n")

    save_trajectory("trajectory.txt", trajectory, phi_system)

    print("Координати точок траєкторії записано у файл trajectory.txt")
    print("Графіки збережено у файли system_equations.png та objective_trajectory.png")

    plot_system_graph(x_min)
    plot_objective_and_trajectory(trajectory)


if __name__ == "__main__":
    main()
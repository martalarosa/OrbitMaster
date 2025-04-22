''' данное ПМО реализует моделирование движения НИСЗ в центральном поле Земли с возможностью
подключения гравитационных возмущений от Луны, Солнца, Юпитера и солнечного ветра.
реализовано с использованием объектно-ориентированной архитектуры и паттерна "Декоратор".
построен графический интерфейс на Tkinter с возможностью ввода начальных условий.
строятся графики эволюции координат, скоростей и 3D-траектория спутника.
мспользуется численное интегрирование системы ОДУ методом solve_ivp из библиотеки SciPy.
при включенной галочке "Физически корректная модель" учитывается реакция Земли на гравитацию внешних тел
(используется модель дифференциальных ускорений: влияние на спутник минус влияние на Землю).
при отключённой галочке — применяется демонстрационный режим: возмущения рассчитываются как абсолютные,
что позволяет получить более наглядные эффекты моделирования. '''

import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from math import sqrt, sin, cos, pi
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# гравитационные параметры (в км^3/с^2)
MU_EARTH = 398600.4418
MU_MOON = 4902.800066
MU_SUN = 132712440018.0
MU_JUPITER = 126686534

# радиусы орбит (в км)
R_MOON = 384400.0
R_SUN = 149600000.0
R_JUPITER = 778500000.0

# угловые скорости (рад/с)
W_MOON = sqrt(MU_EARTH / (R_MOON**3))
W_SUN = 2 * pi / (365.25 * 86400)
W_JUPITER = 2 * pi / (11.86 * 365.25 * 86400)

SOLAR_WIND_ACCEL = 1e-7  # км/с^2

# гравитационные модели
class GravityModel:
    def acceleration(self, t, state):
        raise NotImplementedError

class EarthGravity(GravityModel):
    def acceleration(self, t, state):
        x, y, z = state[0], state[1], state[2]
        r = sqrt(x*x + y*y + z*z)
        if r != 0:
            a = -MU_EARTH / r**3
            return a * x, a * y, a * z
        return 0, 0, 0

class GravityDecorator(GravityModel):
    def __init__(self, base):
        self.base = base

    def acceleration(self, t, state):
        return self.base.acceleration(t, state)

class MoonGravity(GravityDecorator):
    def __init__(self, base, correct=True):
        super().__init__(base)
        self.correct = correct

    def acceleration(self, t, state):
        ax, ay, az = self.base.acceleration(t, state)
        x, y, z = state[0], state[1], state[2]
        theta = W_MOON * t
        xm, ym, zm = R_MOON * cos(theta), R_MOON * sin(theta), 0
        dx, dy, dz = xm - x, ym - y, zm - z
        r1 = sqrt(dx*dx + dy*dy + dz*dz)
        r2 = sqrt(xm*xm + ym*ym + zm*zm)
        if r1 != 0:
            am = MU_MOON / r1**3
            ax += am * dx
            ay += am * dy
            az += am * dz
        if self.correct and r2 != 0:
            am_e = MU_MOON / r2**3
            ax -= am_e * xm
            ay -= am_e * ym
            az -= am_e * zm
        return ax, ay, az

class SunGravity(GravityDecorator):
    def __init__(self, base, correct=True):
        super().__init__(base)
        self.correct = correct

    def acceleration(self, t, state):
        ax, ay, az = self.base.acceleration(t, state)
        x, y, z = state[0], state[1], state[2]
        theta = W_SUN * t
        xs, ys, zs = R_SUN * cos(theta), R_SUN * sin(theta), 0
        dx, dy, dz = xs - x, ys - y, zs - z
        r1 = sqrt(dx*dx + dy*dy + dz*dz)
        r2 = sqrt(xs*xs + ys*ys + zs*zs)
        if r1 != 0:
            asun = MU_SUN / r1**3
            ax += asun * dx
            ay += asun * dy
            az += asun * dz
        if self.correct and r2 != 0:
            asun_e = MU_SUN / r2**3
            ax -= asun_e * xs
            ay -= asun_e * ys
            az -= asun_e * zs
        return ax, ay, az

class JupiterGravity(GravityDecorator):
    def __init__(self, base, correct=True):
        super().__init__(base)
        self.correct = correct

    def acceleration(self, t, state):
        ax, ay, az = self.base.acceleration(t, state)
        x, y, z = state[0], state[1], state[2]
        theta = W_JUPITER * t
        xj, yj, zj = R_JUPITER * cos(theta), R_JUPITER * sin(theta), 0
        dx, dy, dz = xj - x, yj - y, zj - z
        r1 = sqrt(dx*dx + dy*dy + dz*dz)
        r2 = sqrt(xj*xj + yj*yj + zj*zj)
        if r1 != 0:
            aj = MU_JUPITER / r1**3
            ax += aj * dx
            ay += aj * dy
            az += aj * dz
        if self.correct and r2 != 0:
            aj_e = MU_JUPITER / r2**3
            ax -= aj_e * xj
            ay -= aj_e * yj
            az -= aj_e * zj
        return ax, ay, az

class SolarWind(GravityDecorator):
    def acceleration(self, t, state):
        ax, ay, az = self.base.acceleration(t, state)
        ax += SOLAR_WIND_ACCEL
        return ax, ay, az

if __name__ == "__main__":
    # интерфейс
    root = tk.Tk()
    root.title("НИСЗ: моделирование с возмущениями")

    frame = tk.Frame(root)
    frame.pack()

    tk.Label(frame, text="x0, y0, z0 (км)").grid(row=0, column=0)
    entry_x = tk.Entry(frame); entry_x.insert(0, "7000")
    entry_y = tk.Entry(frame); entry_y.insert(0, "0")
    entry_z = tk.Entry(frame); entry_z.insert(0, "0")
    entry_x.grid(row=1, column=0); entry_y.grid(row=1, column=1); entry_z.grid(row=1, column=2)

    tk.Label(frame, text="vx0, vy0, vz0 (км/с)").grid(row=2, column=0)
    entry_vx = tk.Entry(frame); entry_vx.insert(0, "0")
    entry_vy = tk.Entry(frame); entry_vy.insert(0, "7.12")
    entry_vz = tk.Entry(frame); entry_vz.insert(0, "0.5")
    entry_vx.grid(row=3, column=0); entry_vy.grid(row=3, column=1); entry_vz.grid(row=3, column=2)

    tk.Label(frame, text="Шаг (с)").grid(row=4, column=0)
    entry_step = tk.Entry(frame)
    entry_step.insert(0, "60")
    entry_step.grid(row=4, column=1)

    tk.Label(frame, text="Время моделирования (с)").grid(row=4, column=2)
    entry_time = tk.Entry(frame)
    entry_time.insert(0, "86400")
    entry_time.grid(row=5, column=2)

    combo_var = tk.StringVar()
    combo = ttk.Combobox(frame, textvariable=combo_var, width=30, state="readonly")
    combo['values'] = ["Только Земля", "+ Луна", "+ Солнце", "+ Луна и Солнце", "+ Юпитер", "+ Солнечный ветер"]
    combo.current(0)
    combo.grid(row=5, column=0, columnspan=2)

    correct_model_var = tk.BooleanVar(value=True)
    check_correct = tk.Checkbutton(frame, text="Физически корректная модель (учитывать реакцию Земли)", variable=correct_model_var)
    check_correct.grid(row=6, column=0, columnspan=3)

    plot_frame = tk.Frame(root)
    plot_frame.pack()

    def run():
        x0 = float(entry_x.get()); y0 = float(entry_y.get()); z0 = float(entry_z.get())
        vx0 = float(entry_vx.get()); vy0 = float(entry_vy.get()); vz0 = float(entry_vz.get())
        step = float(entry_step.get())
        tf = float(entry_time.get())
        t0 = 0
        t_eval = np.arange(t0, tf, step)
        correct = correct_model_var.get()

        model = EarthGravity()
        choice = combo_var.get()
        if choice == "Только Земля":
            model = EarthGravity()
        elif choice == "+ Луна":
            model = MoonGravity(model, correct)
        elif choice == "+ Солнце":
            model = SunGravity(model, correct)
        elif choice == "+ Луна и Солнце":
            model = MoonGravity(SunGravity(model, correct), correct)
        elif choice == "+ Юпитер":
            model = JupiterGravity(model, correct)
        elif choice == "+ Солнечный ветер":
            model = SolarWind(model)

        def ode(t, s):
            vx, vy, vz = s[3:6]
            ax, ay, az = model.acceleration(t, s)
            return [vx, vy, vz, ax, ay, az]

        sol = solve_ivp(ode, [t0, tf], [x0, y0, z0, vx0, vy0, vz0], t_eval=t_eval, rtol=1e-9, atol=1e-9)

        fig = Figure(figsize=(10, 9))
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3, projection='3d')

        x, y, z = sol.y[0], sol.y[1], sol.y[2]
        vx, vy, vz = sol.y[3], sol.y[4], sol.y[5]

        ax1.plot(sol.t, x, label='x')
        ax1.plot(sol.t, y, label='y')
        ax1.plot(sol.t, z, label='z')
        ax1.set_title("Координаты в зависимости от времени")
        ax1.set_ylabel("км")
        ax1.grid()
        ax1.legend()

        ax2.plot(sol.t, vx, label='vx')
        ax2.plot(sol.t, vy, label='vy')
        ax2.plot(sol.t, vz, label='vz')
        ax2.set_title("Скорости в зависимости от времени")
        ax2.set_xlabel("время, с")
        ax2.set_ylabel("км/с")
        ax2.grid()
        ax2.legend()

        ax3.plot(x, y, z, color='purple')
        ax3.set_title("3D-траектория орбиты НИСЗ", fontsize=10)
        ax3.set_xlabel("X, км")
        ax3.set_ylabel("Y, км")
        ax3.set_zlabel("Z, км")
        ax3.grid(True)

        global canvas
        for child in plot_frame.winfo_children(): child.destroy()
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    btn = tk.Button(frame, text="Запустить моделирование", command=run)
    btn.grid(row=7, column=0, columnspan=3, pady=10)

    root.mainloop()

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")

from OrbitMaster import EarthGravity, MoonGravity, SunGravity, JupiterGravity, SolarWind

def test_earth_gravity_vector():
    model = EarthGravity()
    ax, ay, az = model.acceleration(0, [7000, 0, 0, 0, 0, 0])
    assert ax < 0 and ay == 0 and az == 0

def test_moon_gravity_no_reaction():
    model = MoonGravity(EarthGravity(), correct=False)
    ax, ay, az = model.acceleration(100000, [7000, 0, 0, 0, 0, 0])
    assert isinstance(ax, float)

def test_moon_gravity_with_reaction_vector_difference():
    base = EarthGravity()
    state = [7000, 0, 0, 0, 0, 0]
    t = 100000
    acc_no_react = np.array(MoonGravity(base, correct=False).acceleration(t, state))
    acc_with_react = np.array(MoonGravity(base, correct=True).acceleration(t, state))
    diff = np.linalg.norm(acc_with_react - acc_no_react)
    assert diff > 0  # влияние реакции на Землю изменяет вектор ускорения

def test_solar_wind_effect():
    base = EarthGravity()
    no_wind = base.acceleration(0, [7000, 0, 0, 0, 0, 0])[0]
    with_wind = SolarWind(base).acceleration(0, [7000, 0, 0, 0, 0, 0])[0]
    assert with_wind > no_wind  # солнечный ветер ослабляет притяжение Земли

def test_jupiter_gravity_demo_mode():
    model = JupiterGravity(EarthGravity(), correct=False)
    ax, ay, az = model.acceleration(100000, [7000, 0, 0, 0, 0, 0])
    assert isinstance(ax, float)
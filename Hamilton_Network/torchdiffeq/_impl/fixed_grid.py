from .solvers import FixedGridODESolver
from . import rk_common
import torch

class Euler(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        #print(tuple(dt * f_ for f_ in func(t, y)))
        return tuple(dt * f_ for f_ in func(t, y))

    @property
    def order(self):
        return 1


class Midpoint(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        y_mid = tuple(y_ + f_ * dt / 2 for y_, f_ in zip(y, func(t, y)))
        return tuple(dt * f_ for f_ in func(t + dt / 2, y_mid))

    @property
    def order(self):
        return 2


class RK4(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        return rk_common.rk4_alt_step_func(func, t, dt, y)

    @property
    def order(self):
        return 4


class Leapfrog(FixedGridODESolver):
    def step_func(self, func, t, dt, y_z):
        dy_z=func(t, y_z)
        a=tuple(dy_z_[:, int(dy_z_.shape[1]/2):dy_z_.shape[1]] for dy_z_ in dy_z)
        z=tuple(y_z_[:, int(y_z_.shape[1]/2):y_z_.shape[1]] for y_z_ in y_z)
        dz= tuple(dt * a_ for a_ in a)
        z=tuple(z_+dz_ for z_,dz_ in zip(z, dz))
        out = tuple(torch.cat((z_, z_), 1) for z_ in z)
        dy_z1 = func(t+dt, out)
        b = tuple(dy_z_[:, 0:int(dy_z_.shape[1] / 2)] for dy_z_ in dy_z1)
        dy = tuple(dt * b_ for b_ in b)
        out=tuple(torch.cat((dy_, dz_),1) for dy_, dz_ in zip(dy, dz))
        return out

    @property
    def order(self):
        return 1



class Verlet(FixedGridODESolver):
    def step_func(self, func, t, dt, y_z):
        dy_z = func(t, y_z)
        a = tuple(dy_z_[:, int(dy_z_.shape[1] / 2):dy_z_.shape[1]] for dy_z_ in dy_z)
        b = tuple(dy_z_[:, 0:int(dy_z_.shape[1] / 2)] for dy_z_ in dy_z)
        z = tuple(y_z_[:, int(y_z_.shape[1] / 2):y_z_.shape[1]] for y_z_ in y_z)
        y = tuple(y_z_[:, 0:int(y_z_.shape[1] / 2)] for y_z_ in y_z)
        dy = tuple(dt * b_ + pow(dt, 2) / 2 * a_ for b_, a_ in zip(b, a))
        y1 = tuple(y_ + dy_ for y_, dy_ in zip(y, dy))
        out = tuple(torch.cat((y_, z_), 1) for y_, z_ in zip(y1, z))
        dy_z1 = func(t + dt, out)
        a1 = tuple(dy_z1_[:, int(dy_z1_.shape[1] / 2):dy_z1_.shape[1]] for dy_z1_ in dy_z1)
        dz = tuple(dt * (a_ + a1_) / 2 for a_, a1_ in zip(a, a1))
        out = tuple(torch.cat((dy_, dz_), 1) for dy_, dz_ in zip(dy, dz))
        return out

    @property
    def order(self):
        return 2
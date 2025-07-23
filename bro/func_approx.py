"""
Helpers to create function approximations.
"""

import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def num_coeffs(dims, degree):
    """
    Returns the number of coefficients of a dims-dimensional
    degree-degree polynomial.
    """
    return math.comb(degree + dims, degree)

def yup_all_ones(degree, *coords):
    """
    Evaluates:
        1 + x + y + x^2 + x*y + y^2 + ...
    For the given x/y/..., and up to the given degree exponent.
    """
    dims = len(coords)
    terms = []
    for sumdeg in range(degree + 1):
        for exps in itertools.product(range(sumdeg + 1), repeat=dims):
            exps = exps[::-1]
            if sum(exps) != sumdeg:
                continue
            term = np.prod([C**e for C, e in zip(coords, exps)], axis=0)
            terms.append(term)
    return np.stack(terms).T


class RationalPolynomial:
    def __init__(self, dims, n, m):
        self.pcount = num_coeffs(dims, n)
        self.qcount = num_coeffs(dims, m)
        self.dims = dims
        self.n = n
        self.m = m
        self.pones = None
        self.qones = None

    def initial_coeffs(self):
        # Start with 1 / 1.
        initial = np.zeros(self.pcount + self.qcount)
        initial[0] = 1.0
        initial[self.pcount] = 1.0
        return initial

    @property
    def coeffs(self):
        return self._coeffs
    @coeffs.setter
    def coeffs(self, new_coeffs):
        if len(new_coeffs) != self.pcount + self.qcount:
            raise ValueError(f"expected {self.pcount + self.qcount} "
                    f"vals, got {len(new_coeffs)}")
        self._coeffs = new_coeffs

    def cook(self, pones, qones, *, coeffs=None):
        if coeffs is None:
            coeffs = self.coeffs
        if pones.ndim != 2 or pones.shape[-1] != self.pcount:
            raise ValueError(f"expected yup_all_ones shape, got {pones.shape}")
        if qones.ndim != 2 or qones.shape[-1] != self.qcount:
            raise ValueError(f"expected yup_all_ones shape, got {qones.shape}")
        pc = coeffs[:self.pcount]
        qc = coeffs[self.pcount:]
        P = np.sum(pc * pones, axis=-1)
        Q = np.sum(qc * qones, axis=-1)
        return P/Q

    def __repr__(self):
        pc = self.coeffs[:self.pcount]
        qc = self.coeffs[self.pcount:]
        P = ", ".join(f"{x:.5g}" for x in pc)
        Q = ", ".join(f"{x:.5g}" for x in qc)
        return f"({P}) / ({Q})"


def approximate(func, *coords, n=3, m=3, vectorised=False):
    flatcoords = [x.ravel() for x in coords]
    if not vectorised:
        func = np.vectorise(func)
    dims = len(coords)

    ratpoly = RationalPolynomial(dims=dims, n=n, m=m)
    pones = yup_all_ones(n, *flatcoords)
    qones = yup_all_ones(m, *flatcoords)
    real = func(*flatcoords)
    def loss(coeffs):
        return ratpoly.cook(pones, qones, coeffs=coeffs) - real

    # Find the best set of coefficients.
    res = least_squares(loss, ratpoly.initial_coeffs())
    ratpoly.coeffs = res.x

    return ratpoly

def havealook(ratpoly, func, *coords, vectorised=False, threed=False):
    flatcoords = [x.ravel() for x in coords]
    if not vectorised:
        func = np.vectorise(func)
    dims = len(coords)
    if dims != 1 and dims != 2:
        raise ValueError(f"huh {dims}")
    pones = yup_all_ones(ratpoly.n, *flatcoords)
    qones = yup_all_ones(ratpoly.m, *flatcoords)
    approx = ratpoly.cook(pones, qones).reshape(coords[0].shape)
    real = func(*coords)
    error = 100 * np.abs((approx - real) / real)

    if dims == 2 and threed:
        plotme = [
            ([
                (real, "real", ("b", "Blues")),
                (approx, "approx", ("b", "Oranges")),
            ], "Approximation [real=blue, approx=orange]"),
            (error, r"%error"),
        ]
    else:
        plotme = [
            (real, "Real"),
            (approx, "Approximation"),
            (error, r"%error"),
        ]
    for data, title in plotme:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d" if threed else None)
        conts = []
        if dims == 2:
            def plotter(z, colour="viridis"):
                if threed:
                    ax.plot_surface(*coords, z, cmap=colour,
                            edgecolor="none", alpha=0.9)
                else:
                    cont = ax.contourf(*coords, z, levels=100, cmap=colour)
                    conts.append(cont)
        else:
            def plotter(y, colour=None):
                ax.plot(*coords, y, color=colour)
        if isinstance(data, list):
            for subdata, label, (twod_col, threed_col) in data:
                plotter(subdata, threed_col if dims == 2 else twod_col)
        else:
            plotter(data)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if dims == 2:
            if threed:
                ax.set_zlabel("Z")
                ax.view_init(elev=30, azim=135)
            else:
                for cont in conts:
                    fig.colorbar(cont, ax=ax)
    plt.tight_layout()
    plt.show()


def real(x, y):
    return 1 + 100*np.sin(3*x) * np.exp(-y) + y**2

N = 30
X = np.linspace(2, 4, N)
Y = np.linspace(4, 6, N)
X, Y = np.meshgrid(X, Y)

ratpoly = approximate(real, X, Y, n=2, m=2, vectorised=True)
havealook(ratpoly, real, X, Y, vectorised=True)
print(ratpoly)

"""
Helpers to create function approximations.
"""

import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy


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

def poly_ordering(dims, degree):
    """
    returns a list of strings of each term.
    """
    if dims > 3:
        raise ValueError("havent thought that far ahead")
    coords = "xyz"[:dims]
    terms = []
    def tostr(c, e):
        if e == 0:
            return ""
        if e == 1:
            return c
        return f"{c}^{e}"
    for sumdeg in range(degree + 1):
        for exps in itertools.product(range(sumdeg + 1), repeat=dims):
            exps = exps[::-1]
            if sum(exps) != sumdeg:
                continue
            term = " ".join(tostr(c, e) for c, e in zip(coords, exps))
            terms.append(term)
    terms = [t or "1" for t in terms]
    return terms


class RationalPolynomial:
    def __init__(self, dims, n, m):
        self.dims = dims
        self.n = n
        self.m = m
        self.pcount = num_coeffs(dims, n)
        self.qcount = num_coeffs(dims, m)

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

    def ones(self, *flatcoords):
        pones = yup_all_ones(self.n, *flatcoords)
        qones = yup_all_ones(self.m, *flatcoords)
        return pones, qones

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

    @classmethod
    def of(cls, func, *coords, n=3, m=3, vectorised=False):
        flatcoords = [x.ravel() for x in coords]
        if not vectorised:
            func = np.vectorize(func)
        dims = len(coords)

        ratpoly = cls(dims=dims, n=n, m=m)
        pones, qones = ratpoly.ones(*flatcoords)
        real = func(*flatcoords)
        def diff(coeffs):
            dif = ratpoly.cook(pones, qones, coeffs=coeffs) - real
            return dif
        def maxprop(coeffs):
            dif = ratpoly.cook(pones, qones, coeffs=coeffs) - real
            return np.max(np.abs(dif / real))

        # Find the best set of coefficients.
        coeffs = ratpoly.initial_coeffs()
        # Initially least squares (since its much more stable).
        res = scipy.optimize.least_squares(diff, coeffs)
        coeffs = res.x
        # Then minimise max error.
        res = scipy.optimize.minimize(maxprop, coeffs)
        coeffs = res.x

        ratpoly.coeffs = coeffs
        return ratpoly


    def havealook(self, func, *coords, vectorised=False, threed=False):
        flatcoords = [x.ravel() for x in coords]
        if not vectorised:
            func = np.vectorize(func)
        if self.dims != len(coords):
            raise ValueError(f"expected {self.dims} dims, got {len(coords)}")
        if self.dims != 1 and self.dims != 2:
            raise ValueError(f"huh {self.dims}")
        pones, qones = self.ones(*flatcoords)
        approx = self.cook(pones, qones).reshape(coords[0].shape)
        real = func(*coords)
        error = 100 * np.abs((approx - real) / real)

        if self.dims == 2 and threed:
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
            if self.dims == 2:
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
                    plotter(subdata, threed_col if self.dims == 2 else twod_col)
            else:
                plotter(data)
            ax.set_title(title)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            if self.dims == 2:
                if threed:
                    ax.set_zlabel("Z")
                    ax.view_init(elev=30, azim=135)
                else:
                    for cont in conts:
                        fig.colorbar(cont, ax=ax)
        plt.tight_layout()
        plt.show()

from CoolProp.CoolProp import PropsSI

def real(t):
    return PropsSI("D", "T", t, "Q", 0, "N2O")

N = 30
X = np.linspace(-5 + 273.15, 35 + 273.15, N)

ratpoly = RationalPolynomial.of(real, X, n=3, m=2)
ratpoly.havealook(real, X)
print(ratpoly)
print(poly_ordering(ratpoly.dims, 2))
print(poly_ordering(ratpoly.dims, 3))

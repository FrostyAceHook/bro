"""
Helpers to create function approximations.
"""

import functools
import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy


@functools.cache
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
            return f"{c}"
        return f"{c}^{e} "
    for sumdeg in range(degree + 1):
        for exps in itertools.product(range(sumdeg + 1), repeat=dims):
            exps = exps[::-1]
            if sum(exps) != sumdeg:
                continue
            term = "".join(tostr(c, e) for c, e in zip(coords, exps))
            terms.append(term.strip())
    return terms

def all_idx_tuples(max_cost=None):
    """
    Yields all index tuples and their cost in the order of least-cost to
    most-cost. Note there are infinite index tuples, dont let this thing
    cook forever. Also note i cant be bothered to account for dims altering
    the cost.
    """
    # we can vary degree and num of coeffs.
    # define cost as:
    #   sum((i + 1) * (ci != 0))
    # since each extra degree requires +1 multiplication and addition.
    # so: (0, 3)
    #  has cost (1 + 4) = 5
    # while: (0, 1, 2)
    #  has cost (1, 2, 3) = 6
    def costof(idxs):
        return sum(idxs) + len(idxs)
    # so, lets consider some costs and all idx tuples:
    #  cost = 1
    #    (0, )
    #  cost = 2
    #    (1, )
    #  cost = 3
    #    (2, )
    #    (0, 1)
    #  cost = 4
    #    (3, )
    #    (0, 2)
    #  cost = 5
    #    (4, )
    #    (0, 3)
    #    (1, 2)
    #  cost = 6
    #    (5, )
    #    (0, 4)
    #    (1, 3)
    #    (0, 1, 2)
    #  cost = 7
    #    (6, )
    #    (0, 5)
    #    (1, 4)
    #    (2, 3)
    #    (0, 1, 3)
    #  cost = 8
    #    (7, )
    #    (0, 6)
    #    (1, 5)
    #    (2, 4)
    #    (0, 1, 4)
    #    (0, 2, 3)
    #  cost = 13  [only looking at len 3]
    #    (0, 1, 9)
    #    (0, 2, 8)
    #    (0, 3, 7)
    #    (0, 4, 6)
    #    (1, 3, 6)
    #    (1, 4, 5)
    #  cost = 14  [only looking at initial longest]
    #    (0, 1, 2, 7)
    # its damn counting init.
    # reminder:
    #   sum[i in N < n] = n * (n-1) / 2
    # alright we got the visualisation lets get on with it.
    for cost in itertools.count(1):
        if max_cost is not None and cost > max_cost:
            return

        # want to go longest to shortest actually.
        lengths = []
        for length in itertools.count(1):
            minsum = (length * (length - 1)) // 2
            if length + minsum > cost:
                break
            lengths.append(length)
        for length in lengths[::-1]:
            minsum = (length * (length - 1)) // 2
            idxs = list(range(length))
            idxs[-1] = cost - minsum - 1
            while True:
                yield tuple(idxs), cost
                index = length - 1
                while index > 0:
                    if idxs[index] - 1 > idxs[index - 1] + 1:
                        idxs[index] -= 1
                        idxs[index - 1] += 1
                        break
                    index -= 1
                if index == 0:
                    break

class Polynomial:
    def __init__(self, dims, idxs):
        """
        dims .... integer number of inputs.
        idxs .... tuple of integers specifying non-zero coeffs into
                  the infinite poly-ordering.
        """
        assert idxs == tuple(sorted(idxs))
        assert all(idx >= 0 for idx in idxs)
        # find degree as the smallest which fits all indices.
        degree = 0
        maxidx = max(idxs)
        while num_coeffs(dims, degree) < maxidx + 1:
            degree += 1
        self.degree = degree
        self.dims = dims
        self.count = len(idxs)
        self.countall = num_coeffs(self.dims, self.degree)
        assert all(idx < self.countall for idx in idxs)
        self.idxs = list(idxs) # list to make numpy slicing work

    def initial_coeffs(self):
        initial = np.zeros(self.count)
        # If we have a constant, start with 1.
        if 0 in self.idxs:
            initial[0] = 1.0
        return initial

    @property
    def coeffs(self):
        return self._coeffs
    @coeffs.setter
    def coeffs(self, new_coeffs):
        if len(new_coeffs) != self.count:
            raise ValueError(f"expected {self.count} coeffs, "
                    f"got {len(new_coeffs)}")
        self._coeffs = new_coeffs

    def ones(self, *flatcoords):
        return yup_all_ones(self.degree, *flatcoords)

    def cook(self, ones, *, coeffs=None):
        if coeffs is None:
            coeffs = self.coeffs
        if len(coeffs) != self.count:
            raise ValueError(f"expected {self.count} coeffs, "
                    f"got {len(new_coeffs)}")
        if ones.ndim != 2 or ones.shape[-1] != self.countall:
            raise ValueError("expected yup_all_ones shape, "
                    f"got {ones.shape}")
        allcoeffs = np.zeros(self.countall)
        allcoeffs[self.idxs] = coeffs
        return np.sum(allcoeffs * ones, axis=-1)

    def __repr__(self):
        allcoeffs = np.zeros(self.countall)
        allcoeffs[self.idxs] = self.coeffs
        variables = poly_ordering(self.dims, self.countall)
        def mul(c, x):
            if c == 0:
                return ""
            if c == 1:
                return x
            return f"{c:.5g} {x}".strip()
        terms = [mul(c, x) for c, x in zip(allcoeffs, variables)]
        while terms and not terms[0].strip():
            terms = terms[1:]
        def add(t):
            if not t.strip():
                return ""
            if t[0] == "-":
                return f" - {t[1:]}"
            return f" + {t}"
        terms = terms[0] + "".join(add(t) for t in terms[1:])
        return terms




class RationalPolynomial:
    def __init__(self, dims, pidxs, qidxs):
        self.dims = dims
        self.p = Polynomial(dims, pidxs)
        self.q = Polynomial(dims, qidxs)
    def __repr__(self):
        return f"({self.p}) / ({self.q})"

    def initial_coeffs(self):
        xs = self.p.initial_coeffs(), self.q.initial_coeffs()
        return np.concat(xs)

    def set_coeffs(self, new_coeffs):
        try:
            self.p.coeffs = new_coeffs[:self.p.count]
            self.q.coeffs = new_coeffs[self.p.count:]
        except:
            if hasattr(self.p, "coeffs"):
                del self.p.coeffs
            if hasattr(self.q, "coeffs"):
                del self.q.coeffs
            raise

    def ones(self, *flatcoords):
        return self.p.ones(*flatcoords), self.q.ones(*flatcoords)

    def cook(self, pones, qones, *, coeffs=None):
        if coeffs is not None:
            pcoeffs = coeffs[:self.p.count]
            qcoeffs = coeffs[self.p.count:]
        else:
            pcoeffs = self.p.coeffs
            qcoeffs = self.q.coeffs
        P = self.p.cook(pones, coeffs=pcoeffs)
        Q = self.q.cook(qones, coeffs=qcoeffs)
        if (Q == 0).any():
            raise ZeroDivisionError("Q poly returned 0")
        return P / Q

    def approximate(self, func, *coords, vectorised=False):
        if len(coords) != self.dims:
            raise TypeError(f"expected {self.dims} dims, got {len(coords)}")
        flatcoords = [x.ravel() for x in coords]
        if not vectorised:
            func = np.vectorize(func)

        pones, qones = self.ones(*flatcoords)
        real = func(*flatcoords)
        def diff(coeffs):
            approx = self.cook(pones, qones, coeffs=coeffs)
            return approx - real
        def maxprop(coeffs):
            approx = self.cook(pones, qones, coeffs=coeffs)
            return np.max(np.abs(approx / real - 1))

        # Find the best set of coefficients.
        coeffs = self.initial_coeffs()
        # Initially least squares (since its much more stable).
        res = scipy.optimize.least_squares(diff, coeffs)
        coeffs = res.x
        # Then minimise max error.
        res = scipy.optimize.minimize(maxprop, coeffs)
        coeffs = res.x
        # Then normalise (if it hasnt eliminated that var anyway?).
        if coeffs[self.p.count - 1] != 0:
            coeffs /= coeffs[self.p.count - 1]

        self.set_coeffs(coeffs)
        return float(maxprop(coeffs))


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
        error = 100 * np.abs(approx / real - 1)

        plotme = [
            ([
                (real, "real", ("b", "Blues")),
                (approx, "approx", ("orange", "Oranges")),
            ], "Approximation [real=blue, approx=orange]"),
            (error, "|%error|"),
        ]
        fig = plt.figure(figsize=(6 * len(plotme), 6))
        for i, (data, title) in enumerate(plotme):
            ax = fig.add_subplot(1, len(plotme), 1 + i, projection="3d" if threed else None)
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


    @classmethod
    def of(cls, func, *coords, vectorised=False, max_error=0.01):
        flatcoords = [x.ravel() for x in coords]
        if not vectorised:
            func = np.vectorize(func)
        dims = len(coords)

        # self = cls(dims, (0, 1, 2), (0, 1, 2))
        # maxerr = self.approximate(func, *coords, vectorised=True)
        # print(maxerr)
        # return self

        # go through all numerator idx tuples, over all denoms with
        # upto a slightly higher cost.
        for pidxs, max_cost in all_idx_tuples():
            for qidxs, _ in all_idx_tuples(max_cost=2*max_cost):
                if pidxs == (0, ) and qidxs == (0, ):
                    continue
                self = cls(dims, pidxs, qidxs)
                try:
                    maxerr = self.approximate(func, *coords, vectorised=True)
                except ZeroDivisionError as e:
                    continue
                start = f"{pidxs} {qidxs} .."
                start += "." * (30 - len(start))
                print(f"{start} {maxerr:.4g}" + " !"*(maxerr<0.05) + "!"*(maxerr<0.01))
                if maxerr <= max_error:
                    return self


        # # Rough cost for each term is 1+exp.
        # maxsum = 1
        # while True:
        #     pairs = []
        #     n = 1
        #     while n <= maxsum:
        #         m = maxsum - n
        #         pairs.append((n, m))
        #         n += 1
        #     def sorter(pair):
        #         n, m = pair
        #         return abs(n - m), n
        #     for n, m in sorted(pairs, key=sorter):
        #         ratpoly, maxerr = cls.approximate(func, *coords,
        #                 n=n, m=m, vectorised=True)
        #         if maxerr <= max_error:
        #             return ratpoly
        #     maxsum += 1

    @classmethod
    def of_speced(cls, pidxs, qidxs, func, *coords, vectorised=False):
        flatcoords = [x.ravel() for x in coords]
        if not vectorised:
            func = np.vectorize(func)
        dims = len(coords)

        self = cls(dims, pidxs, qidxs)
        maxerr = self.approximate(func, *coords, vectorised=True)
        return self, maxerr





from CoolProp.CoolProp import PropsSI

def real(t):
    # return t ** 2 + 4
    return PropsSI("D", "T", t, "Q", 1, "N2O")

N = 500
X = np.linspace(-5 + 273.15, 35 + 273.15, N)

ratpoly = RationalPolynomial.of(real, X, max_error=0.02, vectorised=True)
ratpoly.havealook(real, X)
print(ratpoly)

if True:
    ratpoly, _ = RationalPolynomial.of_speced((2,), (0, 1), real, X, vectorised=True)
    ratpoly.havealook(real, X)
    print(ratpoly)


plt.show()

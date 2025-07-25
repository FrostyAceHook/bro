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

def iter_idxs(dims, at_cost=None, max_cost=None):
    """
    Yields all index tuples and their cost in the order of least-cost to
    most-cost. Note there are infinite index tuples, dont let this thing
    cook forever.
    """
    def degreeof(i):
        if dims == 1:
            return i
        for k in itertools.count(0):
            if num_coeffs(dims, k) > i:
                return k
    # we can vary degree and num of coeffs. define a cost heuristic as:
    #   dims * degreeof(max(idxs)) + 2 * len(idxs) - 1
    # since the poly constructs all powers on the way to the highest,
    # and then each term requires 1 mul and 1 add/sub. (note im not
    # accounting for repeated squaring i cannot be fucked). the -1 is
    # just to ensure that the lowest cost tuple of (0,) has a cost of 1.
    # so, cost is entirely determined by length and greatest value. its
    # lowk just counting init.
    costi = 1
    costf = 1000000 # real.
    if at_cost is not None:
        costi = at_cost
        costf = at_cost
    if max_cost is not None:
        costf = max_cost
    for cost in range(costi, costf + 1):
        # want to go longest to shortest actually.
        longest = 0
        while dims * degreeof(longest) + 2*(longest + 1) <= cost:
            longest += 1
        for length in range(longest, 0, -1):
            # want to go smallest to greatest.
            last = length - 2 # lower bound - 1
            while dims * degreeof(last + 1) + 2*length < cost:
                last += 1
            while dims * degreeof(last + 1) + 2*length == cost:
                last += 1
                for combo in itertools.combinations(range(last), length - 1):
                    yield combo + (last, ), cost



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
                    f"got {len(coeffs)}")
        if ones.ndim != 2 or ones.shape[-1] != self.countall:
            raise ValueError("expected yup_all_ones shape, "
                    f"got {ones.shape}")
        return np.sum(coeffs * ones[:, self.idxs], axis=-1)

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
        if not terms:
            return "1"
        terms = terms[0] + "".join(add(t) for t in terms[1:])
        return terms




class RationalPolynomial:
    """
    Approximates a function as a rational polynomial of the form:
           p0 + p1 x + p2 x^2 + ... + x^n
        -----------------------------------
         q0 + q1 x + q2 x^2 + ... + qm x^m
    """

    def __init__(self, dims, pidxs, qidxs):
        self.dims = dims
        self.p = Polynomial(dims, pidxs)
        self.q = Polynomial(dims, qidxs)
    def __repr__(self):
        return f"({self.p}) / ({self.q})"

    def initial_coeffs(self):
        initial = np.zeros(self.p.count - 1 + self.q.count)
        initial[self.p.count - 1] = 1.0 # avoid /0
        return initial

    def set_const(self, const):
        if self.p.idxs != [0] or self.q.idxs != [0]:
            raise ValueError("cannot make non-constant a constant")
        self.p.coeffs = np.array([const])
        self.q.coeffs = np.array([1.0])
    def set_coeffs(self, new_coeffs):
        try:
            pcoeffs = new_coeffs[:self.p.count - 1]
            pcoeffs = np.append(pcoeffs, 1.0)
            self.p.coeffs = pcoeffs
            self.q.coeffs = new_coeffs[self.p.count - 1:]
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
            pcoeffs = coeffs[:self.p.count - 1]
            pcoeffs = np.append(pcoeffs, 1.0)
            qcoeffs = coeffs[self.p.count - 1:]
        else:
            pcoeffs = self.p.coeffs
            qcoeffs = self.q.coeffs
        P = self.p.cook(pones, coeffs=pcoeffs)
        Q = self.q.cook(qones, coeffs=qcoeffs)
        with np.errstate(divide="ignore"):
            return P / Q

    def approximate(self, real, pones, qones):
        # Optimise via least squares.
        def diff(coeffs):
            approx = self.cook(pones, qones, coeffs=coeffs)
            return approx - real
        coeffs = self.initial_coeffs()
        res = scipy.optimize.least_squares(diff, coeffs)
        coeffs = res.x
        self.set_coeffs(coeffs)
        approx = self.cook(pones, qones)
        return float(np.max(np.abs(approx / real - 1)))


    def havealook(self, real, *coords, mask=None, threed=False, figtitle=None):
        flatcoords = [x.ravel() for x in coords]
        if self.dims != len(coords):
            raise ValueError(f"expected {self.dims} dims, got {len(coords)}")
        if self.dims != 1 and self.dims != 2:
            raise ValueError(f"huh {self.dims}")
        pones, qones = self.ones(*flatcoords)
        approx = self.cook(pones, qones).reshape(coords[0].shape)
        error = 100 * np.abs(approx / real - 1)

        if mask is not None:
            real[~mask] = np.nan
            approx[~mask] = np.nan
            error[~mask] = np.nan

        if len(coords) == 1 or threed:
            plotme = [
                ([
                    (real, "real", ("b", "Blues")),
                    (approx, "approx", ("orange", "Oranges")),
                ], "Approximation [real=blue, approx=orange]"),
                (error, "|%error|"),
            ]
        else:
            plotme = [
                (real, "Real"),
                (approx, "Approximation"),
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
        if figtitle is not None:
            fig.suptitle(figtitle)
        plt.tight_layout()


    @classmethod
    def _all_idx_tuples(cls, dims):
        # Iterate through the rat polys in cost order, where the cost of
        # the rat poly is just the sum of each polys cost.
        for cost in itertools.count(1):
            for pidxs, pcost in iter_idxs(dims, max_cost=cost - 1):
                if pidxs[-1] > 2*len(pidxs):
                    continue
                for qidxs, _ in iter_idxs(dims, at_cost=cost-pcost):
                    if qidxs[-1] > 2*len(qidxs):
                        continue
                    # (x + x^2) / x == 1 + x
                    if len(qidxs) == 1 and pidxs[0] != 0:
                        continue
                    # note that while f(x)/c is the same as scaling all
                    # coefficients, it isnt the same for us since we fix
                    # the highest index numerator coefficient to 1.
                    # if qidxs == (0,):
                    #     continue
                    yield pidxs, qidxs


    @classmethod
    def of(cls, real, *coords, max_error=0.01, spec=None):
        if max_error > 0.05:
            raise ValueError("too inaccurate")
        dims = len(coords)
        flatcoords = [x.ravel() for x in coords]
        real = real.ravel()

        if (real == 0.0).any():
            raise Exception("zero?")

        # Try constant first.
        if spec is None:
            specisconst = True
        else:
            if len(spec) != 2:
                raise ValueError("spec should only have pidxs and qidxs, "
                        f"got {len(spec)}")
            specisconst = (list(spec[0]), list(spec[1])) == ([0], [0])
        mid = (np.max(real) + np.min(real)) / 2
        maxerr = float(np.max(np.abs(mid / real - 1)))
        if maxerr <= max_error and specisconst:
            start = "const. .."
            start += "." * (30 - len(start))
            start += f" {100 * maxerr:.4g}%"
            start += " !"*(maxerr < 2*max_error) + "!"*(maxerr < 1.5*max_error)
            print(start)
            self = cls(dims, (0, ), (0, ))
            self.set_const(mid)
            return self

        padto = 35
        N = 6
        ponesN = yup_all_ones(N, *flatcoords)
        qonesN = yup_all_ones(N, *flatcoords)
        best = float("inf")

        if spec is None:
            idxs = cls._all_idx_tuples(dims)
        else:
            idxs = [spec]

        # Iterate through the rat polys in cost order, where the cost of
        # the rat poly is just the sum of each polys cost.
        for pidxs, qidxs in idxs:
            self = cls(dims, pidxs, qidxs)
            if self.p.degree <= N:
                pones = ponesN[:, :self.p.countall]
            else:
                pones = self.p.ones(*flatcoords)
            if self.q.degree <= N:
                qones = qonesN[:, :self.q.countall]
            else:
                qones = self.q.ones(*flatcoords)

            s = f"{pidxs} / {qidxs}"
            s += " " * (padto - len(s))
            print(s, end="\r")
            maxerr = self.approximate(real, pones, qones)
            if maxerr > best:
                continue
            # if maxerr > 5*max_error:
            #     continue
            start = f"{pidxs} / {qidxs} .."
            start += "." * (padto - len(start))
            start += f" {100 * maxerr:.4g}%"
            start += " !"*(maxerr < 2*max_error) + "!"*(maxerr < 1.5*max_error)
            print(start)
            if maxerr <= max_error or spec is not None:
                return self
            best = maxerr







def _main():
    from CoolProp.CoolProp import PropsSI

    def concspace(lo, conc, hi, N, strength=1.0):
        if not (hi > lo):
            raise ValueError("hi <= lo")
        if not (lo <= conc and conc <= hi):
            raise ValueError("conc oob")
        # Do between 0 and 1.
        x = np.linspace(0, 1, N)
        b = (conc - lo) / (hi - lo)
        b = max(b, 1e-5)
        b = min(b, 1 - 1e-5)
        # https://www.desmos.com/calculator/pr6knfebgh
        def afromb(b):
            b2 = b*b
            b3 = b2*b
            b4 = b3*b
            term = 2*b3 - 3*b2 + np.sqrt(b4 - 2*b3 + b2) + b
            a = np.cbrt(term / 2)
            a -= (b - b2) * np.cbrt(2 / term)
            a += b
            return a
        a = afromb(b)
        def biased(x):
            denom = 3*a*a - 3*a + 1
            c1 = 3*a*a/denom
            c2 = -3*a/denom
            c3 = 1/denom
            return c1*x + c2*x*x + c3*x*x*x
        xbias = biased(x)
        # Weighted sum of bias and linear.
        y = (x + xbias * strength) / (1 + strength)
        # Map to lo..hi.
        return lo + (hi - lo) * y


    def do(func, Xs, Ys=None, maskf=None, max_error=0.01, spec=None):
        print(func.__name__)
        Xs = (Xs.ravel(),) if Ys is None else (Xs.ravel(), Ys.ravel())
        bounds = [(X.min(), X.max()) for X in Xs]
        fines = [np.linspace(lo, hi, 100) for lo, hi in bounds]
        real = func(*Xs)
        if maskf is not None:
            mask = maskf(*Xs)
            Xs = [X[mask] for X in Xs]
            real = real[mask]
        ratpoly = RationalPolynomial.of(real, *Xs, max_error=max_error, spec=spec)
        fines = np.meshgrid(*fines)
        realfine = func(*[x.ravel() for x in fines]).reshape(fines[0].shape)
        maskfine = maskf(*fines) if maskf is not None else None
        ratpoly.havealook(realfine, *fines, mask=maskfine, figtitle=func.__name__)
        print(func.__name__, "->", ratpoly)


    def spaceT(conc, strength=1.0, N=80):
        # temp of tank reasonably within -10..35 dC.
        return 273.15 + concspace(-10, conc, 35, N=N, strength=strength)

    def nox_rho_satliq(T):
        return PropsSI("D", "T", T, "Q", 0, "N2O")
    # do(nox_rho_satliq, spaceT(30))

    def nox_rho_satvap(T):
        return PropsSI("D", "T", T, "Q", 1, "N2O")
    # do(nox_rho_satvap, spaceT(30), max_error=0.02) # so ill behaved :(

    def nox_P_satliq(T):
        return PropsSI("P", "T", T, "Q", 0, "N2O")
    # do(nox_P_satliq, spaceT(30))

    def nox_P(T, rho): # only for vapour
        return PropsSI("P", "T", T, "D", rho, "N2O")
    def space_Trho():
        # density of [un]saturated tank vapour reasonably within 1..325.
        X = 273.15 + np.linspace(-10, 35, 40)
        Y = np.linspace(1, 250, 40)
        X, Y = np.meshgrid(X, Y)
        X = X.ravel()
        Y = Y.ravel()
        # but its never gonna be denser than saturated density for a temp.
        def mask(X, Y):
            rhosat = PropsSI("D", "T", X.ravel(), "Q", 1, "N2O")
            rhosat = rhosat.reshape(X.shape)
            return (Y <= rhosat)
        return X, Y, mask
    # do(nox_P, *space_Trho(), max_error=0.02)


    def spaceP(conc=3, strength=0.0, N=80):
        # pressure of saturated tank reasonably within 1..7 MPa.
        return concspace(1, conc, 7, N=N, strength=strength)

    def nox_s_satliq(P):
        return PropsSI("S", "P", P * 1e6, "Q", 0, "N2O")
    # do(nox_s_satliq, spaceP(3.5))
    def nox_s_satvap(P):
        return PropsSI("S", "P", P * 1e6, "Q", 1, "N2O")
    # do(nox_s_satvap, spaceP(3.5))

    def nox_cp(T, P): # only for vapour
        return PropsSI("C", "T", T, "P", P * 1e6, "N2O")
    def space_TP():
        X = 273.15 + concspace(-10, 35, 35, strength=2.0, N=50)
        Y = concspace(1, 7, 7, strength=2.0, N=50)
        X, Y = np.meshgrid(X, Y)
        X = X.ravel()
        Y = Y.ravel()
        # but its never gonna be hotter than Tsat for a pressure.
        def mask(X, Y):
            Tsat = PropsSI("T", "P", Y.ravel() * 1e6, "Q", 1, "N2O")
            Tsat = Tsat.reshape(X.shape)
            # return (X > 0)
            return (X > Tsat)
        return X, Y, mask
    # X, Y, mask = space_TP()
    # Z = nox_cp(X.ravel(), Y.ravel()).reshape(X.shape)
    # Z[~mask(X,Y)] = np.nan
    # contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    # plt.colorbar(contour)
    # plt.show()
    do(nox_cp, *space_TP())

    # do(nox_cp, *space_TP(), spec=[(1, 2, 3), (0, 1, 2, 5)])

    def nox_cv_satliq(T):
        pass
    def nox_cv_satvap(T):
        pass
    def nox_cv(T, P):
        pass

    def nox_h_satliq(T):
        pass
    def nox_h(T, rho):
        pass

    def nox_u_satliq(T):
        pass
    def nox_u_satvap(T):
        pass
    def nox_u(T, rho):
        pass

    def nox_Z(T, rho):
        pass

    plt.show()

if __name__ == "__main__":
    _main()

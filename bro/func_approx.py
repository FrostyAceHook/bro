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
@functools.cache
def degreeof(dims, i):
    """
    Returns the degree (sum of exponenets) of the given index.
    """
    if dims == 1:
        return i
    for k in itertools.count(0):
        if num_coeffs(dims, k) > i:
            return k
@functools.cache
def invdegreeof(dims, k):
    """
    Returns the smallest i s.t. degreeof(i) == k. Note that there may
    be many such solutions, this returns the smallest.
    """
    if dims == 1:
        return k
    for i in itertools.count(0):
        if degreeof(dims, i) == k:
            return i

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
    most-cost (then ordered by descending length and then ascending last
    index). Note there are infinite index tuples, dont let this thing go
    forever.
    """
    # we can vary degree and num of coeffs. define a cost heuristic as:
    #   dims * degreeof(max(idxs)) + 2 * len(idxs) - 1
    # since the poly constructs all powers on the way to the highest,
    # and then each term requires 1 mul and 1 add/sub. (note im not
    # accounting for repeated squaring i cannot be fucked). the -1 is
    # just to ensure that the lowest cost tuple of (0,) has a cost of 1.
    # so, cost is entirely determined by length and greatest value.
    costi = 1
    costf = 1000000 # real.
    if at_cost is not None:
        costi = at_cost
        costf = at_cost
    if max_cost is not None:
        costf = max_cost
    for cost in range(costi, costf + 1):
        # want to go longest to shortest.
        longest = 0
        while dims * degreeof(dims, longest) + 2*(longest + 1) - 1 <= cost:
            longest += 1
        for length in range(longest, 0, -1):
            # want to go smallest to greatest.
            #  dims * degreeof(last) + 2*length - 1 = cost
            #  dims * degreeof(last) = cost - 2*length + 1
            # may not be solutions of this cost for this length.
            if (cost - 2*length + 1) % dims:
                continue
            #  degreeof(last) = (cost - 2*length + 1) // dims
            #  last = invdegreeof((cost - 2*length + 1) // dims)
            last = invdegreeof(dims, (cost - 2*length + 1) // dims) - 1
            while dims * degreeof(dims, last + 1) + 2*length - 1 == cost:
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
            return f"{repr(float(c))} {x}".strip()
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

    def approximate(self, real, pones, qones, method):
        # Optimise via least squares.
        def diff(coeffs):
            approx = self.cook(pones, qones, coeffs=coeffs)
            return approx - real
        coeffs = self.initial_coeffs()
        res = scipy.optimize.least_squares(diff, coeffs, method=method)
        coeffs = res.x
        self.set_coeffs(coeffs)
        approx = self.cook(pones, qones)
        # Return max %error (as proportion tho).
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
    def _all_idx_tuples(cls, dims, stay_low=False):
        # Iterate through the rat polys in cost order, where the cost of
        # the rat poly is just the sum of each polys cost.
        for cost in itertools.count(1):
            for pidxs, pcost in iter_idxs(dims, max_cost=cost - 1):
                if stay_low and pidxs[-1] > 1.5*len(pidxs):
                    continue
                for qidxs, _ in iter_idxs(dims, at_cost=cost-pcost):
                    if stay_low and qidxs[-1] > 1.5*len(qidxs):
                        continue
                    # if the numerator and denominator both dont have
                    # constants, x can be factored from both.
                    # (x + x^2) / x == 1 + x
                    # for higher dimensions, we dont bother checking but
                    # the same factorisation can be applied to find
                    # redundant options.
                    if dims == 1 and pidxs[0] != 0 and qidxs[0] != 0:
                        continue
                    # note that while f(x)/c is the same as scaling all
                    # coefficients, it isnt the same for us since we fix
                    # the highest index numerator coefficient to 1.
                    # if qidxs == (0,):
                    #     continue
                    yield pidxs, qidxs


    @classmethod
    def of(cls, real, *coords, max_error=0.01, spec=None, stay_low=False,
            printall=False, method="lm"):
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

        padto = 45
        N = 6
        ponesN = yup_all_ones(N, *flatcoords)
        qonesN = yup_all_ones(N, *flatcoords)
        best = float("inf")

        if spec is None:
            idxs = cls._all_idx_tuples(dims, stay_low=stay_low)
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
            maxerr = self.approximate(real, pones, qones, method)
            if maxerr > best:
                continue
            if not printall and spec is None and maxerr > 5*max_error:
                continue
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
        if strength == 0.0:
            return np.linspace(lo, hi, N)
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


    def do(func, X, Y=None, maskf=None, max_error=0.01, spec=None,
                stay_low=False, printall=False):
        print(func.__name__)
        Xs = (X.ravel(),) if Y is None else (X.ravel(), Y.ravel())
        bounds = [(X.min(), X.max()) for X in Xs]
        fines = [np.linspace(lo, hi, 100) for lo, hi in bounds]
        real = func(*Xs)
        if maskf is not None:
            mask = maskf(*Xs)
            Xs = [X[mask] for X in Xs]
            real = real[mask]
        ratpoly = RationalPolynomial.of(real, *Xs, max_error=max_error,
                spec=spec, stay_low=stay_low, printall=printall)
        fines = np.meshgrid(*fines)
        realfine = func(*[x.ravel() for x in fines]).reshape(fines[0].shape)
        maskfine = maskf(*fines) if maskf is not None else None
        ratpoly.havealook(realfine, *fines, mask=maskfine, figtitle=func.__name__)
        print(func.__name__, "->", ratpoly)
        print()
    def peek(func, X, Y, maskf):
        Z = func(X.ravel(), Y.ravel()).reshape(X.shape)
        Z[~maskf(X, Y)] = np.nan
        fig = plt.figure(figsize=(8, 5))
        fig.suptitle(func.__name__)
        ax = fig.add_subplot(1, 1, 1)
        contour = ax.contourf(X, Y, Z, levels=100, cmap="viridis")
        fig.colorbar(contour, ax=ax)


    def spaceT(conc=None, strength=0.0, N=120):
        # temp of tank reasonably within -10..35 dC.
        return 273.15 + concspace(-10, conc, 35, N=N, strength=strength)
    # NOTE: to aid the function/avoid overflow, all pressure calcs are done in MPa.
    def spaceP(conc=None, strength=0.0, N=120):
        # pressure of tank reasonably within 1..7 MPa.
        return concspace(1, conc, 7, N=N, strength=strength)
    def spacerho(conc=None, strength=0.0, N=120):
        # density of [un]saturated tank vapour reasonably within 1..325.
        return concspace(1, conc, 250, N=N, strength=strength)
    def space_TP(N=80, Tconc=None, Tstrength=0.0, Pconc=None, Pstrength=0.0):
        X = spaceT(conc=Tconc, strength=Tstrength, N=N)
        Y = spaceP(conc=Pconc, strength=Pstrength, N=N)
        X, Y = np.meshgrid(X, Y)
        # but its never gonna be hotter than Tsat for a pressure.
        def mask(X, Y):
            Tsat = PropsSI("T", "P", Y.ravel() * 1e6, "Q", 1, "N2O")
            Tsat = Tsat.reshape(X.shape)
            return (X > Tsat)
        return X, Y, mask
    def space_Trho(N=80, Tconc=None, Tstrength=0.0, rhoconc=None, rhostrength=0.0):
        X = spaceT(conc=Tconc, strength=Tstrength, N=N)
        Y = spacerho(conc=rhoconc, strength=rhostrength, N=N)
        X, Y = np.meshgrid(X, Y)
        # but its never gonna be denser than saturated density for a temp.
        def mask(X, Y):
            rhosat = PropsSI("D", "T", X.ravel(), "Q", 1, "N2O")
            rhosat = rhosat.reshape(X.shape)
            return (Y <= rhosat)
        return X, Y, mask


    def nox_rho_satliq(T):
        return PropsSI("D", "T", T, "Q", 0, "N2O")
    do(nox_rho_satliq, spaceT(conc=30, strength=1.5))

    def nox_rho_satvap(T):
        return PropsSI("D", "T", T, "Q", 1, "N2O")
    # do(nox_rho_satvap, spaceT(conc=30, strength=2.0), max_error=0.0001, printall=True) # so ill behaved :(
    # do(nox_rho_satvap, spaceT(conc=30, strength=2.0), spec=[(0, 1, 4), (0, 1, 4)])
    do(nox_rho_satvap, spaceT(conc=30, strength=2.0), spec=[(0, 1), (0, 1, 2)])

    def nox_P_satliq(T):
        return PropsSI("P", "T", T, "Q", 0, "N2O")
    do(nox_P_satliq, spaceT(), max_error=0.0115) # can get away w quadratic :)

    def nox_P(T, rho): # only for vapour
        return PropsSI("P", "T", T, "D", rho, "N2O")
    # peek(nox_P, *space_Trho())
    # do(nox_P, *space_Trho(Tconc=-5, Tstrength=2.0, rhoconc=50, rhostrength=1.0), max_error=0.0001, printall=True)
    do(nox_P, *space_Trho(Tconc=-5, Tstrength=1.0), spec=[(4, 5, 8, 9), (0,)])

    def nox_s_satliq(P):
        return PropsSI("S", "P", P * 1e6, "Q", 0, "N2O")
    do(nox_s_satliq, spaceP(conc=7, strength=0.5))
    def nox_s_satvap(P):
        return PropsSI("S", "P", P * 1e6, "Q", 1, "N2O")
    do(nox_s_satvap, spaceP(conc=7, strength=1.5))

    def nox_cp(T, P): # only for vapour
        return PropsSI("C", "T", T, "P", P * 1e6, "N2O")
    # peek(nox_cp, *space_TP())
    # do(nox_cp, *space_TP(), stay_low=True, max_error=0.0001, printall=True)
    do(nox_cp, *space_TP(), spec=[(3, 4, 5, 6), (2, 3, 4, 5, 7)])


    def nox_cv_satliq(T):
        return PropsSI("O", "T", T, "Q", 0, "N2O")
    do(nox_cv_satliq, spaceT())
    def nox_cv_satvap(T):
        return PropsSI("O", "T", T, "Q", 1, "N2O")
    do(nox_cv_satvap, spaceT(), max_error=0.017) # nice one at 1.7%

    def nox_cv(T, P):
        return PropsSI("O", "T", T, "P", P * 1e6, "N2O")
    # peek(nox_cv, *space_TP())
    # do(nox_cv, *space_TP(), stay_low=True, max_error=0.0001, printall=True)
    do(nox_cv, *space_TP(), spec=[(2, 4, 5, 6), (0, 1, 2)])

    def nox_h_satliq(T):
        return PropsSI("H", "T", T, "Q", 0, "N2O")
    do(nox_h_satliq, spaceT(), max_error=0.02) # super simple soln at 2%

    def nox_h(T, rho):
        return PropsSI("H", "T", T, "D", rho, "N2O")
    # peek(nox_h, *space_Trho())
    do(nox_h, *space_Trho(), max_error=0.0105) # mate why couldnt all the 2d funcs have been this simple.


    def nox_u_satliq(T):
        return PropsSI("U", "T", T, "Q", 0, "N2O")
    do(nox_u_satliq, spaceT(conc=25, strength=1.0), max_error=0.014)
    def nox_u_satvap(T):
        return PropsSI("U", "T", T, "Q", 1, "N2O")
    do(nox_u_satvap, spaceT(conc=-5, strength=1.0))

    def nox_u(T, rho):
        return PropsSI("U", "T", T, "D", rho, "N2O")
    # peek(nox_u, *space_Trho())
    do(nox_u, *space_Trho()) # ANOTHER BANGER

    def nox_Z(T, rho):
        return PropsSI("Z", "T", T, "D", rho, "N2O")
    # peek(nox_Z, *space_Trho())
    do(nox_Z, *space_Trho(), max_error=0.011)


    plt.show()


    """
Canon output:

nox_rho_satliq
(0, 1, 2) / (0, 1) .......................... 0.7908% !!
nox_rho_satliq -> (157151.12823174248 - 813.8826644193667 x + x^2) / (74.57612932055565 - 0.2348599590791976 x)

nox_rho_satvap
(0, 1) / (0, 1, 2) .......................... 3.357%
nox_rho_satvap -> (-252.882039808751 + x) / (-14.832030176269813 + 0.10545924440865234 x - 0.00018415699738461084 x^2)

nox_P_satliq
(2, 3) / (0,) ............................... 4.654%
(0, 1, 2) / (0,) ............................ 0.8391% !!
nox_P_satliq -> (54347.01215675704 - 459.63144619907933 x + x^2) / (0.0010950795892039642)

nox_P
(4, 5, 8, 9) / (0,) ......................... 0.5213% !!
nox_P -> (1245.3085211000268 xy - 2476.9434303190055 y^2 + 4.5848819175612 xy^2 + y^3) / (6.609810524334591)

nox_s_satliq
(1,) / (0, 1, 2) ............................ 1.259% !!
(0, 1) / (0, 1, 2) .......................... 1.198% !!
(0, 1) / (0, 1, 3) .......................... 0.9068% !!
nox_s_satliq -> (0.20684447939143843 + x) / (0.00196337089508787 + 0.0009025188040885002 x - 4.1418606497075e-06 x^3)

nox_s_satvap
(0,) / (0, 1) ............................... 3.45%
(0,) / (0, 1, 2) ............................ 2.277%
(1,) / (0, 1, 3) ............................ 2.077%
(0,) / (0, 1, 2, 3) ......................... 0.9881% !!
nox_s_satvap -> (1) / (0.0005237695555884831 + 6.2442065066513e-05 x - 1.1609056509777005e-05 x^2 + 1.0964425747711539e-06 x^3)

nox_cp
(3, 4, 5, 6) / (2, 3, 4, 5, 7) .............. 2.105%
nox_cp -> (618.7744602768846 x^2 - 53619.71494364253 xy + 648249.0052956727 y^2 + x^3) / (-104364.80843903389 y + 1.0419898422695852 x^2 + 561.1789501111347 xy + 1598.4985271937874 y^2 - 0.9883915366256657 x^2 y)

nox_cv_satliq
(0, 1, 2) / (0,) ............................ 3.647%
(1,) / (0, 1, 2) ............................ 3.501%
(1, 2, 3) / (0,) ............................ 3.448%
(0, 3, 4) / (0,) ............................ 3.397%
(1, 2, 4) / (0,) ............................ 3.38%
(1, 3, 4) / (0,) ............................ 3.312%
(2, 3, 4) / (0,) ............................ 3.226%
(0, 1, 2) / (0, 1) .......................... 0.2389% !!
nox_cv_satliq -> (1144775.8711670972 - 3936.9793029653038 x + x^2) / (1192.5631799152063 - 3.787505750839443 x)

nox_cv_satvap
(2,) / (0,) ................................. 4.128%
(1,) / (0, 1) ............................... 4.011%
(1,) / (0, 2) ............................... 3.826%
(0, 1) / (0, 1) ............................. 3.302% !
(0, 1, 2) / (0,) ............................ 2.027% !!
(1,) / (0, 1, 2) ............................ 1.585% !!
nox_cv_satvap -> (x) / (-1.480927678522855 + 0.013307429042598332 x - 2.4688024330268616e-05 x^2)

nox_cv
(2, 4, 5, 6) / (0, 1, 2) .................... 0.8357% !!
nox_cv -> (4763761.21385638 y - 24513.987100000944 xy - 46704.46867095853 y^2 + x^3) / (-47385.87739194443 + 289.0454919271015 x - 5000.196506775096 y)

nox_h_satliq
(0, 1) / (0,) ............................... 5.644%
(0, 2) / (0,) ............................... 5.036%
(1, 2) / (0,) ............................... 4.733%
(4,) / (0,) ................................. 3.14% !
(2,) / (0, 1) ............................... 2.879% !!
(2,) / (0, 2) ............................... 2.576% !!
(0, 1) / (0, 1) ............................. 2.112% !!
(1, 2) / (0, 1) ............................. 2.087% !!
(1, 3) / (0, 1) ............................. 2.066% !!
(1, 4) / (0, 1) ............................. 2.037% !!
(0, 1, 2, 3) / (0,) ......................... 1.496% !!
nox_h_satliq -> (-21407367.636026487 + 230083.54206007754 x - 825.6343192575749 x^2 + x^3) / (1.315764476256798)

nox_h
(1,) / (0, 2) ............................... 3.833%
(0, 1) / (0, 2) ............................. 0.8932% !!
nox_h -> (155.7853196205554 + x) / (0.0009619058954610157 + 1.252184201380225e-06 y)

nox_u_satliq
(0, 1) / (0,) ............................... 5.263%
(1,) / (0, 1) ............................... 1.942% !!
(2,) / (0, 3) ............................... 1.671% !!
(0, 1, 2, 3) / (0,) ......................... 1.427% !!
(0, 1, 2, 4) / (0,) ......................... 1.409% !!
(0, 1, 3, 4) / (0,) ......................... 1.392% !!
nox_u_satliq -> (-6176985187.102616 + 43975766.57385516 x - 552.6737487069622 x^3 + x^4) / (849.9127250854099)

nox_u_satvap
(0,) / (1,) ................................. 4.602%
(0, 2) / (3,) ............................... 4.124%
(2,) / (0, 1, 2) ............................ 3.751%
(0, 1) / (0, 1, 2) .......................... 3.311%
(0, 1, 2) / (0, 1) .......................... 0.2543% !!
nox_u_satvap -> (30813392721.76704 - 97659111.3253914 x + x^2) / (83162.17882870714 - 262.5451257336752 x)

nox_u
(0, 1) / (0, 2) ............................. 0.6615% !!
nox_u -> (225.67703828046328 + x) / (0.0012625313845092653 + 1.2779649751838328e-06 y)

nox_Z
(0,) / (0, 1, 2, 5) ......................... 3.871%
(0,) / (0, 2, 3, 5) ......................... 3.846%
(0,) / (0, 2, 4, 5) ......................... 1.347% !!
(0, 2, 4, 5) / (0,) ......................... 1.083% !!
nox_Z -> (380984.4850035124 - 3758.6254131330284 y + 8.779189092243614 xy + y^2) / (381511.9341439733)


    """


if __name__ == "__main__":
    _main()

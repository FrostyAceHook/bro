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
        for cost in itertools.count(2):
            for pidxs, pcost in iter_idxs(dims, max_cost=cost - 1):
                if stay_low and pidxs[-1] > 1.5*len(pidxs):
                    continue
                for qidxs, _ in iter_idxs(dims, at_cost=cost-pcost):
                    if stay_low:
                        if qidxs[-1] > 1.5*len(qidxs):
                            continue
                        minlen = min(len(qidxs), len(pidxs))
                        maxlen = max(len(qidxs), len(pidxs))
                        if maxlen > 1.5*minlen:
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

    def __repr__(self):
        return f"({self.p}) / ({self.q})"

    def cython(self):
        assert self.dims <= 3
        p = self.p
        q = self.q
        allidxs = set(p.idxs) | set(q.idxs)
        dims = self.dims
        degree = max(p.degree, q.degree)
        ftoa = lambda x: repr(float(x))
        s = ""

        def poly(negs, ones, cs, xs):
            parts = []
            for neg, one, c, x in zip(negs, ones, cs, xs):
                pre = "-" if neg else "+"
                if not parts:
                    if pre == "+":
                        pre = ""
                else:
                    pre += " "
                if not x:
                    parts.append(f"{pre}{c} ")
                else:
                    if one:
                        parts.append(f"{pre}{x} ")
                    else:
                        parts.append(f"{pre}{c}*{x} ")
            return "".join(parts).strip().lstrip()


        # Make all powers.
        xnames = {}
        i = -1
        for sumdeg in range(degree + 1):
            for exps in itertools.product(range(sumdeg + 1), repeat=dims):
                exps = exps[::-1]
                if sum(exps) != sumdeg:
                    continue
                i += 1
                if i not in allidxs:
                    continue
                exps += (0, ) * (3 - len(exps))
                parts = []
                parts += [f"x{exps[0]}"] * (exps[0] > 0)
                parts += [f"y{exps[1]}"] * (exps[1] > 0)
                parts += [f"z{exps[2]}"] * (exps[2] > 0)
                name = "".join(parts)
                xnames[i] = name
                if name:
                    s += f"    cdef double {name} = \n"

        if q.idxs == [0]:
            coeffs = p.coeffs / q.coeffs[0]
            # just normal poly.
            ones = []
            for i, coeff in zip(p.idxs, coeffs):
                ones.append(abs(coeff) == 1.0 and xnames[i])
                if ones[-1]:
                    continue
                s += f"    cdef double c{i} = <double>{ftoa(abs(coeff))}\n"
            negs = [c < 0 for c in coeffs]
            cs = [f"c{i}" for i in p.idxs]
            xs = [xnames[i] for i in p.idxs]
            s += f"    return " + poly(negs, ones, cs, xs)
            return s

        pones = []
        for i, coeff in zip(p.idxs, p.coeffs):
            pones.append(abs(coeff) == 1.0 and xnames[i])
            if pones[-1]:
                continue
            s += f"    cdef double p{i} = <double>{ftoa(abs(coeff))}\n"
        qones = []
        for i, coeff in zip(q.idxs, q.coeffs):
            qones.append(abs(coeff) == 1.0 and xnames[i])
            if qones[-1]:
                continue
            s += f"    cdef double q{i} = <double>{ftoa(abs(coeff))}\n"
        pnegs = [c < 0 for c in p.coeffs]
        qnegs = [c < 0 for c in q.coeffs]
        pcs = [f"p{i}" for i in p.idxs]
        qcs = [f"q{i}" for i in q.idxs]
        pxs = [xnames[i] for i in p.idxs]
        qxs = [xnames[i] for i in q.idxs]
        s += f"    cdef double P = {poly(pnegs, pones, pcs, pxs)}\n"
        s += f"    cdef double Q = {poly(qnegs, qones, qcs, qxs)}\n"
        s += f"    return P / Q"
        return s





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


    def do(func, X, Y=None, maskf=None, **of_kwargs):
        print(func.__name__)
        Xs = (X.ravel(),) if Y is None else (X.ravel(), Y.ravel())
        bounds = [(X.min(), X.max()) for X in Xs]
        fines = [np.linspace(lo, hi, 100) for lo, hi in bounds]
        real = func(*Xs)
        if maskf is not None:
            mask = maskf(*Xs, training=True)
            Xs = [X[mask] for X in Xs]
            real = real[mask]
        ratpoly = RationalPolynomial.of(real, *Xs, **of_kwargs)
        fines = np.meshgrid(*fines)
        realfine = func(*[x.ravel() for x in fines]).reshape(fines[0].shape)
        maskfine = maskf(*fines) if maskf is not None else None
        ratpoly.havealook(realfine, *fines, mask=maskfine, figtitle=func.__name__)
        print(func.__name__, "->", ratpoly)
        print(ratpoly.cython())
        print()
    def peek(func, X, Y, maskf=None):
        Z = func(X.ravel(), Y.ravel()).reshape(X.shape)
        if maskf is not None:
            Z[~maskf(X, Y)] = np.nan
        fig = plt.figure(figsize=(8, 5))
        fig.suptitle(func.__name__)
        ax = fig.add_subplot(1, 1, 1)
        contour = ax.contourf(X, Y, Z, levels=100, cmap="viridis")
        fig.colorbar(contour, ax=ax)


    # NOTE: to avoid overflow, all pressure calcs are done in MPa.

    def nox_in_T(conc=None, strength=0.0, N=120):
        # temp of tank reasonably within -10..35 dC.
        return 273.15 + concspace(-10, conc, 35, N=N, strength=strength)
    def nox_in_Psat(conc=None, strength=0.0, N=120):
        # pressure of saturated tank for our temp range within 2 .. 7.2 MPa.
        return concspace(2, conc, 7.2, N=N, strength=strength)
    def nox_in_P(conc=None, strength=0.0, N=120):
        # all nox pressures reasonably within 0.08 .. 7.2 MPa.
        return concspace(0.08, conc, 7.2, N=N, strength=strength)
    def nox_in_rho(conc=None, strength=0.0, N=120):
        # density of vapour reasonably within 1..325 kg/m^3.
        return concspace(1, conc, 250, N=N, strength=strength)
    def nox_in_TP(N=60, Tconc=None, Tstrength=0.0, Pconc=None, Pstrength=0.0):
        X = nox_in_T(conc=Tconc, strength=Tstrength, N=N)
        Y = nox_in_P(conc=Pconc, strength=Pstrength, N=N)
        X, Y = np.meshgrid(X, Y)
        # but its never gonna be higher pressure than Psat for a temp.
        def mask(X, Y, training=False):
            Psat = PropsSI("P", "T", X.ravel(), "Q", 1, "N2O")
            Psat = Psat.reshape(Y.shape)
            Psat /= 1e6
            mask = (Y < Psat)
            if training:
                mask &= (Y > Psat - 4)
            return mask
        return X, Y, mask
    def nox_in_Trho(N=50, Tconc=None, Tstrength=0.0, rhoconc=None, rhostrength=0.0):
        X = nox_in_T(conc=Tconc, strength=Tstrength, N=N)
        Y = nox_in_rho(conc=rhoconc, strength=rhostrength, N=N)
        X, Y = np.meshgrid(X, Y)
        # but its never gonna be denser than saturated density for a temp.
        def mask(X, Y, training=False):
            rhosat = PropsSI("D", "T", X.ravel(), "Q", 1, "N2O")
            rhosat = rhosat.reshape(X.shape)
            return (Y <= rhosat)
        return X, Y, mask


    def nox_rho_satliq(T):
        return PropsSI("D", "T", T, "Q", 0, "N2O")
    do(nox_rho_satliq, nox_in_T(conc=35, strength=1.5))

    def nox_rho_satvap(T):
        return PropsSI("D", "T", T, "Q", 1, "N2O")
    # do(nox_rho_satvap, nox_in_T(conc=30, strength=2.0), max_error=0.0001, printall=True) # so ill behaved :(
    do(nox_rho_satvap, nox_in_T(conc=30, strength=2.0), spec=[(0, 1, 4), (0, 1, 4)])

    def nox_P_satliq(T):
        return PropsSI("P", "T", T, "Q", 0, "N2O")
    do(nox_P_satliq, nox_in_T(), max_error=0.0115) # can get away w quadratic :)

    def nox_P(T, rho): # only for vapour
        return PropsSI("P", "T", T, "D", rho, "N2O")
    # peek(nox_P, *nox_in_Trho())
    # do(nox_P, *nox_in_Trho(Tconc=-10, Tstrength=2.0, rhoconc=50, rhostrength=1.0), max_error=0.0001, printall=True)
    do(nox_P, *nox_in_Trho(Tconc=-10, Tstrength=1.0), spec=[(5, 7, 9), (1,)])

    def nox_s_satliq(P):
        return PropsSI("S", "P", P * 1e6, "Q", 0, "N2O")
    do(nox_s_satliq, nox_in_Psat(conc=7, strength=0.5))
    def nox_s_satvap(P):
        return PropsSI("S", "P", P * 1e6, "Q", 1, "N2O")
    do(nox_s_satvap, nox_in_Psat(conc=7, strength=0.5))

    def nox_cp(T, P): # only for vapour
        return PropsSI("C", "T", T, "P", P * 1e6, "N2O")
    # peek(nox_cp, *nox_in_TP())
    # do(nox_cp, *nox_in_TP(Tconc=35, Tstrength=0.5, Pconc=7.2, Pstrength=0.5), stay_low=True, max_error=0.0001, printall=True)
    do(nox_cp, *nox_in_TP(Tconc=35, Tstrength=0.5, Pconc=7.2, Pstrength=0.5), spec=[(1, 2, 3), (0, 1, 2, 5)])

    def nox_cv_satliq(T):
        return PropsSI("O", "T", T, "Q", 0, "N2O")
    # do(nox_cv_satliq, nox_in_T())
    def nox_cv_satvap(T):
        return PropsSI("O", "T", T, "Q", 1, "N2O")
    do(nox_cv_satvap, nox_in_T(conc=35, strength=1.0), max_error=0.017) # nice one at 1.7%

    def nox_cv(T, P): # only for vapour
        return PropsSI("O", "T", T, "P", P * 1e6, "N2O")
    # peek(nox_cv, *nox_in_TP())
    # do(nox_cv, *nox_in_TP(), stay_low=True, max_error=0.0001, printall=True)
    do(nox_cv, *nox_in_TP(), spec=[(2, 4, 5, 6), (0, 1, 2)])

    def nox_h_satliq(T):
        return PropsSI("H", "T", T, "Q", 0, "N2O")
    do(nox_h_satliq, nox_in_T(conc=35, strength=1.0), max_error=0.02) # super simple soln at 2%

    def nox_h(T, rho): # only for vapour
        return PropsSI("H", "T", T, "D", rho, "N2O")
    # peek(nox_h, *nox_in_Trho())
    do(nox_h, *nox_in_Trho(), max_error=0.0105) # mate why couldnt all the 2d funcs have been this simple.


    def nox_u_satliq(T):
        return PropsSI("U", "T", T, "Q", 0, "N2O")
    do(nox_u_satliq, nox_in_T(conc=25, strength=1.0), max_error=0.014)
    def nox_u_satvap(T):
        return PropsSI("U", "T", T, "Q", 1, "N2O")
    do(nox_u_satvap, nox_in_T(conc=-10, strength=2.0))

    def nox_u(T, rho): # only for vapour
        return PropsSI("U", "T", T, "D", rho, "N2O")
    # peek(nox_u, *nox_in_Trho())
    do(nox_u, *nox_in_Trho()) # ANOTHER BANGER

    def nox_Z(T, rho): # only for vapour
        return PropsSI("Z", "T", T, "D", rho, "N2O")
    # peek(nox_Z, *nox_in_Trho())
    do(nox_Z, *nox_in_Trho(), max_error=0.011)


    plt.show()


    """
Canon output:

nox_rho_satliq
(0, 1, 2) / (0, 1) .......................... 0.6808% !!
nox_rho_satliq -> (1.5609e+05 - 810.65 x + x^2) / (73.7 - 0.23238 x)
    cdef double x1 =
    cdef double x2 =
    cdef double p0 = <double>156087.80354721082
    cdef double p1 = <double>810.653176062025
    cdef double q0 = <double>73.69985723964697
    cdef double q1 = <double>0.23238113977053437
    cdef double P = p0 - p1*x1 + x2
    cdef double Q = q0 - q1*x1
    return P / Q

nox_rho_satvap
(0, 1, 4) / (0, 1, 4) ....................... 0.09131% !!
nox_rho_satvap -> (1.3017e+10 - 7.2262e+07 x + x^4) / (-2.9225e+08 + 1.1908e+06 x - 0.0083643 x^4)
    cdef double x1 =
    cdef double x4 =
    cdef double p0 = <double>13017372521.157866
    cdef double p1 = <double>72262011.77066447
    cdef double q0 = <double>292245993.87807804
    cdef double q1 = <double>1190768.9555756454
    cdef double q4 = <double>0.00836425317722044
    cdef double P = p0 - p1*x1 + x4
    cdef double Q = -q0 + q1*x1 - q4*x4
    return P / Q

nox_P_satliq
(2, 3) / (0,) ............................... 4.654%
(0, 1, 2) / (0,) ............................ 0.8391% !!
nox_P_satliq -> (54347 - 459.63 x + x^2) / (0.0010951)
    cdef double x1 =
    cdef double x2 =
    cdef double c0 = <double>49628367.373975985
    cdef double c1 = <double>419724.2380649199
    cdef double c2 = <double>913.1756356877407
    return c0 - c1*x1 + c2*x2

nox_P
(5, 7, 9) / (1,) ............................ 0.792% !!
nox_P -> (-957.88 y^2 + 3.5012 x^2 y + y^3) / (0.018523 x)
    cdef double x1 =
    cdef double y2 =
    cdef double x2y1 =
    cdef double y3 =
    cdef double p5 = <double>957.8831858912396
    cdef double p7 = <double>3.5011895479129755
    cdef double q1 = <double>0.018522546839394596
    cdef double P = -p5*y2 + p7*x2y1 + y3
    cdef double Q = q1*x1
    return P / Q

nox_s_satliq
(0, 1) / (0,) ............................... 3.844%
(0, 1) / (0, 1) ............................. 3.81%
(1,) / (0, 1, 2) ............................ 3.268%
(0, 1) / (1, 2) ............................. 3.169%
(1,) / (0, 1, 3) ............................ 2.794%
(0, 1) / (1, 3) ............................. 2.466%
(0,) / (0, 1, 2, 3) ......................... 1.541% !
(0, 2, 3, 5) / (0,) ......................... 1.451% !!
(0, 1, 2, 3) / (0, 1) ....................... 0.8644% !!
nox_s_satliq -> (359.38 + 103.87 x - 26.497 x^2 + x^3) / (1.0548 - 0.13307 x)
    cdef double x1 =
    cdef double x2 =
    cdef double x3 =
    cdef double p0 = <double>359.3783151517807
    cdef double p1 = <double>103.86623173005904
    cdef double p2 = <double>26.496588420618643
    cdef double q0 = <double>1.0548444415506883
    cdef double q1 = <double>0.13306626678729863
    cdef double P = p0 + p1*x1 - p2*x2 + x3
    cdef double Q = q0 - q1*x1
    return P / Q

nox_s_satvap
(0,) / (0, 3) ............................... 3.286%
(0,) / (0, 1, 2, 3) ......................... 3.028%
(0, 1, 2) / (0, 1) .......................... 0.4748% !!
nox_s_satvap -> (246.11 - 40.262 x + x^2) / (0.14145 - 0.018736 x)
    cdef double x1 =
    cdef double x2 =
    cdef double p0 = <double>246.10930235262038
    cdef double p1 = <double>40.262395529528995
    cdef double q0 = <double>0.14145003304901121
    cdef double q1 = <double>0.018736153535782087
    cdef double P = p0 - p1*x1 + x2
    cdef double Q = q0 - q1*x1
    return P / Q

nox_cp
(1, 2, 3) / (0, 1, 2, 5) .................... 3.573%
nox_cp -> (-116.52 x - 7300.8 y + x^2) / (-91.651 + 0.52417 x - 14.922 y + 0.71462 y^2)
    cdef double x1 =
    cdef double y1 =
    cdef double x2 =
    cdef double y2 =
    cdef double p1 = <double>116.51610731426311
    cdef double p2 = <double>7300.76925443315
    cdef double q0 = <double>91.65124829819894
    cdef double q1 = <double>0.5241714826256261
    cdef double q2 = <double>14.921840347891258
    cdef double q5 = <double>0.7146233838229532
    cdef double P = -p1*x1 - p2*y1 + x2
    cdef double Q = -q0 + q1*x1 - q2*y1 + q5*y2
    return P / Q

nox_cv_satvap
(1,) / (0,) ................................. 8.148%
(2,) / (0,) ................................. 4.308%
(0, 1) / (0,) ............................... 3.936%
(1,) / (0, 1) ............................... 3.308% !
(1,) / (0, 2) ............................... 3.158% !
(0, 1) / (0, 1) ............................. 2.768% !
(0, 1, 2) / (0,) ............................ 1.721% !!
(1,) / (0, 1, 2) ............................ 1.341% !!
nox_cv_satvap -> (x) / (-1.7249 + 0.015009 x - 2.7648e-05 x^2)
    cdef double x1 =
    cdef double x2 =
    cdef double q0 = <double>1.7249360631879567
    cdef double q1 = <double>0.015009299578479001
    cdef double q2 = <double>2.7648375677840006e-05
    cdef double P = x1
    cdef double Q = -q0 + q1*x1 - q2*x2
    return P / Q

nox_cv
(2, 4, 5, 6) / (0, 1, 2) .................... 1.19% !!
nox_cv -> (5.7893e+06 y - 28106 xy + 4133.8 y^2 + x^3) / (-44609 + 277.53 x - 4639.1 y)
    cdef double x1 =
    cdef double y1 =
    cdef double x1y1 =
    cdef double y2 =
    cdef double x3 =
    cdef double p2 = <double>5789250.582826521
    cdef double p4 = <double>28105.916013494785
    cdef double p5 = <double>4133.783907249727
    cdef double q0 = <double>44609.46603039637
    cdef double q1 = <double>277.5261817149765
    cdef double q2 = <double>4639.100711485678
    cdef double P = p2*y1 - p4*x1y1 + p5*y2 + x3
    cdef double Q = -q0 + q1*x1 - q2*y1
    return P / Q

nox_h_satliq
(0, 1) / (0,) ............................... 5.704%
(0, 2) / (0,) ............................... 4.377%
(1, 2) / (0,) ............................... 4.109%
(4,) / (0,) ................................. 3.237% !
(2,) / (0, 1) ............................... 2.641% !!
(2,) / (0, 2) ............................... 2.414% !!
(0, 1) / (0, 1) ............................. 1.875% !!
nox_h_satliq -> (66.033 + x) / (0.007195 - 1.8826e-05 x)
    cdef double x1 =
    cdef double p0 = <double>66.0327027959993
    cdef double q0 = <double>0.007194968322024715
    cdef double q1 = <double>1.882583428861607e-05
    cdef double P = p0 + x1
    cdef double Q = q0 - q1*x1
    return P / Q

nox_h
(0, 1) / (0, 2) ............................. 0.889% !!
nox_h -> (156.61 + x) / (0.00096376 + 1.2531e-06 y)
    cdef double x1 =
    cdef double y1 =
    cdef double p0 = <double>156.6103714457326
    cdef double q0 = <double>0.0009637606915044852
    cdef double q2 = <double>1.2531446548205385e-06
    cdef double P = p0 + x1
    cdef double Q = q0 + q2*y1
    return P / Q

nox_u_satliq
(0, 1) / (0,) ............................... 5.263%
(1,) / (0, 1) ............................... 1.942% !!
(2,) / (0, 3) ............................... 1.671% !!
(0, 1, 2, 3) / (0,) ......................... 1.427% !!
(0, 1, 2, 4) / (0,) ......................... 1.409% !!
(0, 1, 3, 4) / (0,) ......................... 1.392% !!
nox_u_satliq -> (-6.177e+09 + 4.3976e+07 x - 552.67 x^3 + x^4) / (849.91)
    cdef double x1 =
    cdef double x3 =
    cdef double x4 =
    cdef double c0 = <double>7267787.626643519
    cdef double c1 = <double>51741.50859952818
    cdef double c3 = <double>0.6502711777275986
    cdef double c4 = <double>0.0011765913963689713
    return -c0 + c1*x1 - c3*x3 + c4*x4

nox_u_satvap
(0,) / (1,) ................................. 4.215%
(2,) / (0, 1, 2) ............................ 3.748%
(0, 1) / (0, 1, 2) .......................... 3.311%
(0, 1, 2) / (0, 1) .......................... 0.1887% !!
nox_u_satvap -> (3.9314e+10 - 1.2468e+08 x + x^2) / (1.0624e+05 - 335.65 x)
    cdef double x1 =
    cdef double x2 =
    cdef double p0 = <double>39314394868.610306
    cdef double p1 = <double>124682202.67355517
    cdef double q0 = <double>106235.11214047845
    cdef double q1 = <double>335.64679750282
    cdef double P = p0 - p1*x1 + x2
    cdef double Q = q0 - q1*x1
    return P / Q

nox_u
(1,) / (0, 2) ............................... 4.4%
(0, 1) / (0, 2) ............................. 0.6589% !!
nox_u -> (226.5 + x) / (0.0012646 + 1.2787e-06 y)
    cdef double x1 =
    cdef double y1 =
    cdef double p0 = <double>226.49887221481575
    cdef double q0 = <double>0.001264621721201892
    cdef double q2 = <double>1.2786597468309012e-06
    cdef double P = p0 + x1
    cdef double Q = q0 + q2*y1
    return P / Q

nox_Z
(0,) / (0, 1, 2, 5) ......................... 3.864%
(0,) / (0, 2, 3, 5) ......................... 3.845%
(0,) / (0, 2, 4, 5) ......................... 1.332% !!
(0, 2, 4, 5) / (0,) ......................... 1.044% !!
nox_Z -> (3.8099e+05 - 3756.7 y + 8.7719 xy + y^2) / (3.8151e+05)
    cdef double y1 =
    cdef double x1y1 =
    cdef double y2 =
    cdef double c0 = <double>0.998648554728038
    cdef double c2 = <double>0.009846953884435787
    cdef double c4 = <double>2.2992669091214318e-05
    cdef double c5 = <double>2.621170870338741e-06
    return c0 - c2*y1 + c4*x1y1 + c5*y2



    """


if __name__ == "__main__":
    _main()

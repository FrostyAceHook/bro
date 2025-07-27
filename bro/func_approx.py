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


_allidxs_cache = {}
def allidxs_at_cost(dims, cost, blitz=0.0, _cache=True):
    """
    Returns a list of all index tuples with the given cost, ordered by
    descending length and then ascending last index.
    """

    # Cache small costs.
    if _cache and cost < 20:
        key = (dims, cost, blitz)
        if key not in _allidxs_cache:
            it = allidxs_at_cost(dims, cost, blitz, _cache=False)
            _allidxs_cache[key] = list(it)
        yield from _allidxs_cache[key]
        return

    # We can vary degree and num of coeffs. define a cost heuristic as:
    #   dims * degreeof(max(idxs)) + 2 * len(idxs) - 1
    # since the poly constructs all powers on the way to the highest,
    # and then each term requires 1 mul and 1 add/sub. (note im not
    # accounting for repeated squaring i cannot be fucked). the -1 is
    # just to ensure that the lowest cost tuple of (0,) has a cost of 1.
    # so, cost is entirely determined by length and greatest value.
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
            # If blitzing, dont let the powers get too extreme without a
            # lot of other powers around. We only do this when blitzing
            # because it means we may miss cheaper solutions, however
            # those solutions are generally unlikely and by eliminating
            # them we churn through possiblities much faster.
            if blitz > 0:
                if last > (1 + 1.0/blitz) * length:
                    break
            for combo in itertools.combinations(range(last), length - 1):
                yield combo + (last,), cost

def allidxs(dims, min_cost, max_cost, blitz=0.0):
    """
    Returns a list of all index tuples with a cost less than the given
    maximum.
    """
    if min_cost > max_cost:
        return
    if min_cost < max_cost:
        yield from allidxs(dims, min_cost, max_cost - 1, blitz=blitz)
    yield from allidxs_at_cost(dims, max_cost, blitz=blitz)




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
        initial[self.p.count - 1] = 1.0e-3 # avoid /0
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
    def _all_idx_tuples(cls, dims, blitz=0.0, starting_cost=2):
        # Iterate through the rat polys in cost order, where the cost of
        # the rat poly is just the sum of each polys cost.
        for cost in itertools.count(starting_cost):
            min_cost = 1
            max_cost = cost - 1
            # If blitzing, don't let the numer/denom cost difference get too extreme
            # (since its more likely to be an accurate approx if not).
            if blitz > 0:
                min_cost = int(max_cost / (2 + 1.0/blitz))
                max_cost -= min_cost
            for pidxs, pcost in allidxs(dims, min_cost, max_cost, blitz=blitz):
                for qidxs, _ in allidxs_at_cost(dims, cost - pcost, blitz=blitz):
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
    def of(cls, real, *coords, max_error=0.01, spec=None, blitz=False,
            starting_cost=2, printall=False, method="lm"):
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

        padto = 55
        N = 6
        ponesN = yup_all_ones(N, *flatcoords)
        qonesN = yup_all_ones(N, *flatcoords)
        best = float("inf")

        if spec is None:
            idxs = cls._all_idx_tuples(dims, blitz=blitz, starting_cost=starting_cost)
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
        ftoa = lambda x: "+"*bool(x>=0.0) + repr(float(x))
        s = ""

        def poly(ones, cs, xs):
            parts = []
            for one, c, x in zip(ones, cs, xs):
                pre = "+ " * (not not parts)
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
                    s += f"    double {name} = ;\n"

        if q.idxs == [0]:
            coeffs = p.coeffs / q.coeffs[0]
            # just normal poly.
            ones = []
            for i, coeff in zip(p.idxs, coeffs):
                ones.append(abs(coeff) == 1.0 and xnames[i])
                if ones[-1]:
                    continue
                s += f"    double c{i} = {ftoa(coeff)};\n"
            cs = [f"c{i}" for i in p.idxs]
            xs = [xnames[i] for i in p.idxs]
            s += f"    return {poly(ones, cs, xs)};"
            return s
        if p.idxs == [0]:
            # inverse poly.
            numer = float(p.coeffs[0] / q.coeffs[-1])
            coeffs = q.coeffs / q.coeffs[-1]
            ones = []
            s += f"    double n0 = {ftoa(numer)};\n"
            for i, coeff in zip(q.idxs, coeffs):
                ones.append(abs(coeff) == 1.0 and xnames[i])
                if ones[-1]:
                    continue
                s += f"    double d{i} = {ftoa(coeff)};\n"
            cs = [f"d{i}" for i in q.idxs]
            xs = [xnames[i] for i in q.idxs]
            s += f"    double Num = {'-'*(numer < 0)}n0;\n"
            s += f"    double Den = {poly(ones, cs, xs)};\n"
            s += "    return Num / Den;"
            return s

        pones = []
        for i, coeff in zip(p.idxs, p.coeffs):
            pones.append(abs(coeff) == 1.0 and xnames[i])
            if pones[-1]:
                continue
            s += f"    double n{i} = {ftoa(coeff)};\n"
        qones = []
        for i, coeff in zip(q.idxs, q.coeffs):
            qones.append(abs(coeff) == 1.0 and xnames[i])
            if qones[-1]:
                continue
            s += f"    double d{i} = {ftoa(coeff)};\n"
        pcs = [f"n{i}" for i in p.idxs]
        qcs = [f"d{i}" for i in q.idxs]
        pxs = [xnames[i] for i in p.idxs]
        qxs = [xnames[i] for i in q.idxs]
        s += f"    double Num = {poly(pones, pcs, pxs)};\n"
        s += f"    double Den = {poly(qones, qcs, qxs)};\n"
        s += f"    return Num / Den;"
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


    DO_PLOTTING = True
    def do(func, X, Y=None, maskf=None, **of_kwargs):
        print(func.__name__)
        Xs = (X.ravel(),) if Y is None else (X.ravel(), Y.ravel())
        bounds = [(X.min(), X.max()) for X in Xs]
        real = func(*Xs)
        if maskf is not None:
            mask = maskf(*Xs, training=True)
            Xs = [X[mask] for X in Xs]
            real = real[mask]
        ratpoly = RationalPolynomial.of(real, *Xs, **of_kwargs)
        if DO_PLOTTING:
            fines = [np.linspace(lo, hi, 100) for lo, hi in bounds]
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
        cont = ax.contourf(X, Y, Z, levels=100, cmap="viridis")
        fig.colorbar(cont, ax=ax)
        plt.tight_layout()
    def compare(X, Y, realf, approxf):
        real = realf(X, Y)
        approx = approxf(X, Y)
        err = 100 * np.abs(real / approx - 1.0)
        fig = plt.figure(figsize=(8, 5))
        for i, data in enumerate([real, approx, err]):
            ax = fig.add_subplot(1, 3, 1 + i)
            cont = ax.contourf(X, Y, data, levels=100, cmap="viridis")
            fig.colorbar(cont, ax=ax)
        plt.tight_layout()




    # NOTE: to avoid overflow, all pressure calcs are done in MPa.

    def nox_in_T(conc=None, strength=0.0, N=120):
        # temp of tank reasonably within -10..35 dC.
        return 273.15 + concspace(-10, conc, 35, strength=strength, N=N)
    def nox_in_Psat(conc=None, strength=0.0, N=120):
        # pressure of saturated tank for our temp range within 2 .. 7.2 MPa.
        return concspace(2, conc, 7.2, strength=strength, N=N)
    def nox_in_P(conc=None, strength=0.0, N=120):
        # all nox pressures reasonably within 0.08 .. 7.2 MPa.
        return concspace(0.08, conc, 7.2, strength=strength, N=N)
    def nox_in_rho(conc=None, strength=0.0, N=120):
        # density of vapour reasonably within 1..325 kg/m^3.
        return concspace(1, conc, 325, strength=strength, N=N)
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


    def nox_T_sat(P):
        return PropsSI("T", "P", P * 1e6, "Q", 0, "N2O")
    # do(nox_T_sat, nox_in_Psat(), max_error=0.00001, printall=True)
    do(nox_T_sat, nox_in_Psat(), spec=[(0, 1), (0, 1)])

    def nox_rho_satliq(T):
        return PropsSI("D", "T", T, "Q", 0, "N2O")
    # do(nox_rho_satliq, nox_in_T(conc=35, strength=1.5))
    do(nox_rho_satliq, nox_in_T(conc=35, strength=1.5), spec=[(0, 1, 2), (0, 1)])

    def nox_rho_satvap(T):
        return PropsSI("D", "T", T, "Q", 1, "N2O")
    # so ill behaved :(
    # do(nox_rho_satvap, nox_in_T(), max_error=0.0001, printall=True)
    do(nox_rho_satvap, nox_in_T(), spec=[(0, 1, 3), (0, 1, 3)])

    def nox_P_satliq(T):
        return PropsSI("P", "T", T, "Q", 0, "N2O")
    # do(nox_P_satliq, nox_in_T())
    do(nox_P_satliq, nox_in_T(), spec=[(0, 1, 2), (0,)]) # can get away w quadratic :)

    def nox_P(T, rho): # only for vapour
        return PropsSI("P", "T", T, "D", rho, "N2O")
    # peek(nox_P, *nox_in_Trho())
    # do(nox_P, *nox_in_Trho(Tconc=-10, Tstrength=2.0, rhoconc=50, rhostrength=1.0), max_error=0.0001, printall=True)
    do(nox_P, *nox_in_Trho(Tconc=-10, Tstrength=1.0), spec=[(5, 7, 9), (1,)])

    def nox_s_satliq(P):
        return PropsSI("S", "P", P * 1e6, "Q", 0, "N2O")
    # do(nox_s_satliq, nox_in_Psat(conc=7, strength=0.5))
    do(nox_s_satliq, nox_in_Psat(conc=7, strength=0.5), spec=[(0,), (0, 1, 2, 6)])
    def nox_s_satvap(P):
        return PropsSI("S", "P", P * 1e6, "Q", 1, "N2O")
    # do(nox_s_satvap, nox_in_Psat(conc=7, strength=0.5))
    do(nox_s_satvap, nox_in_Psat(conc=7, strength=0.5), spec=[(0, 1, 2), (0, 1)])

    def nox_cp(T, P): # only for vapour
        return PropsSI("C", "T", T, "P", P * 1e6, "N2O")
    # peek(nox_cp, *nox_in_TP())
    # do(nox_cp, *nox_in_TP(Tconc=35, Tstrength=0.5, Pconc=7.2, Pstrength=0.5), max_error=0.0001, printall=True)
    do(nox_cp, *nox_in_TP(Tconc=35, Tstrength=0.5, Pconc=7.2, Pstrength=0.5), spec=[(1, 2, 3), (0, 1, 2, 5)])

    def nox_cv_satliq(T):
        return PropsSI("O", "T", T, "Q", 0, "N2O")
    # do(nox_cv_satliq, nox_in_T())
    do(nox_cv_satliq, nox_in_T(), spec=[(0, 1, 2), (0, 1)])
    def nox_cv_satvap(T):
        return PropsSI("O", "T", T, "Q", 1, "N2O")
    # do(nox_cv_satvap, nox_in_T(conc=35, strength=1.0))
    do(nox_cv_satvap, nox_in_T(conc=35, strength=1.0), spec=[(1,), (0, 1, 2)])

    def nox_cv(T, P): # only for vapour
        return PropsSI("O", "T", T, "P", P * 1e6, "N2O")
    # peek(nox_cv, *nox_in_TP())
    # do(nox_cv, *nox_in_TP(), max_error=0.0001, printall=True)
    do(nox_cv, *nox_in_TP(), spec=[(2, 4, 5, 6), (0, 1, 2)])

    def nox_h_satliq(T):
        return PropsSI("H", "T", T, "Q", 0, "N2O")
    # do(nox_h_satliq, nox_in_T(conc=35, strength=1.0))
    do(nox_h_satliq, nox_in_T(conc=35, strength=1.0), spec=[(1,), (0, 1)])

    def nox_h_satvap(T):
        return PropsSI("H", "T", T, "Q", 1, "N2O")
    # do(nox_h_satvap, nox_in_T(), max_error=0.00001, printall=True)
    do(nox_h_satvap, nox_in_T(), spec=[(0, 1, 2), (0, 1)])

    def nox_h(T, rho): # only for vapour
        return PropsSI("H", "T", T, "D", rho, "N2O")
    # mate why couldnt all the 2d funcs have been this simple.
    # peek(nox_h, *nox_in_Trho())
    # do(nox_h, *nox_in_Trho())
    do(nox_h, *nox_in_Trho(), spec=[(1,), (0, 1, 2)])


    def nox_u_satliq(T):
        return PropsSI("U", "T", T, "Q", 0, "N2O")
    # do(nox_u_satliq, nox_in_T(conc=25, strength=1.0))
    do(nox_u_satliq, nox_in_T(conc=25, strength=1.0), spec=[(0, 1, 3, 4), (0,)])
    def nox_u_satvap(T):
        return PropsSI("U", "T", T, "Q", 1, "N2O")
    # do(nox_u_satvap, nox_in_T(conc=-10, strength=2.0))
    do(nox_u_satvap, nox_in_T(conc=-10, strength=2.0), spec=[(0, 1, 2), (0, 1)])

    def nox_u(T, rho): # only for vapour
        return PropsSI("U", "T", T, "D", rho, "N2O")
    # ANOTHER BANGER
    # peek(nox_u, *nox_in_Trho())
    # do(nox_u, *nox_in_Trho())
    do(nox_u, *nox_in_Trho(), spec=[(1,), (0, 1, 2)])

    def nox_Z(T, rho): # only for vapour
        return PropsSI("Z", "T", T, "D", rho, "N2O")
    # peek(nox_Z, *nox_in_Trho())
    # do(nox_Z, *nox_in_Trho())
    do(nox_Z, *nox_in_Trho(), spec=[(0, 2, 4, 5), (0,)])


    from . import optimiser
    fuel = optimiser.PARAFFIN
    ox = optimiser.NOX
    cea = optimiser.CEA_Obj(propName="", oxName=ox.name, fuelName=fuel.name,
                  isp_units="sec",
                  cstar_units="m/s",
                  pressure_units="Pa",
                  temperature_units="K",
                  sonic_velocity_units="m/s",
                  enthalpy_units="J/kg",
                  density_units="kg/m^3",
                  specific_heat_units="J/kg-K")

    # NOTE: same pressure calcs done in MPa, and ofr must be >0.5 (for anything lower,
    #       assume no combustion occurs)

    def cea_in_P(conc=None, strength=0.0, N=120):
        # chamber pressure reasonably within 0.08 .. 7 MPa.
        return concspace(0.08, conc, 7, strength=strength, N=N)
    def cea_in_ofr(conc=None, strength=0.0, N=120):
        # ox-fuel ratio reasonably below 13.
        return concspace(0.5, conc, 13, strength=strength, N=N)
    # def cea_in_eps(conc=None, strength=0.0, N=120):
    #     # nozzle expansion ratio reasonably within 5..40.
    #     return concspace(5, conc, 40, strength=strength, N=N)
    def cea_in_Pofr(Pconc=None, Pstrength=0.0, ofrconc=None, ofrstrength=0.0, N=40):
        X = cea_in_P(Pconc, Pstrength, N=N)
        Y = cea_in_ofr(ofrconc, ofrstrength, N=N)
        return np.meshgrid(X, Y)


    # Tcomb is a huge pain to approx, so we split into a high-ofr approx and
    # a low-ofr approx which combine to cover the whole input space. This
    # should have negligible performance impacts because the motor generally
    # stays in high ofr at the beginning then is in low ofr for the rest (so
    # very predictable branch).
    def cea_Tcomb_in_high(N=70):
        X = cea_in_P(conc=None, strength=0.0, N=N)
        Y = concspace(4, 4, 13, strength=1.0, N=N)
        return np.meshgrid(X, Y)
    def cea_Tcomb_in_low(N=70):
        X = cea_in_P(conc=0.1, strength=1.0, N=N)
        Y = concspace(0.5, 0.5, 4, strength=1.0, N=N)
        return np.meshgrid(X, Y)
    cea_get_Tcomb = np.vectorize(cea.get_Tcomb)
    def cea_Tcomb_high(P, ofr):
        return cea_get_Tcomb(P * 1e6, ofr)
    def cea_Tcomb_low(P, ofr): # just get a different __name__
        return cea_get_Tcomb(P * 1e6, ofr)
    # peek(cea_Tcomb_high, *np.meshgrid(cea_in_P(), cea_in_ofr()))

    # do(cea_Tcomb_high, *cea_Tcomb_in_high(), blitz=1.0, max_error=0.00001, printall=True)
    do(cea_Tcomb_high, *cea_Tcomb_in_high(), spec=[(0, 1, 2, 5), (1, 4, 5, 6, 7)])

    # do(cea_Tcomb_low, *cea_Tcomb_in_low(), blitz=1.0, max_error=0.00001, printall=True)
    do(cea_Tcomb_low, *cea_Tcomb_in_low(), spec=[(0, 2, 4, 5), (1, 2, 4, 5, 7, 8)])


    # after having a peek at cp, id say "looks fucking grim mate". lets split
    # over ofr again. HOLY its a doozy. split low again into left and right.
    def cea_Cp_in_high(N=50):
        X = cea_in_P(conc=None, strength=0.0, N=N)
        Y = concspace(4, 4, 13, strength=2.0, N=N)
        return np.meshgrid(X, Y)
    def cea_Cp_in_low_left(N=40):
        X = concspace(0.08, 1, 1, strength=2.0, N=N)
        Y = concspace(0.5, 4, 4, strength=2.0, N=N)
        X, Y = np.meshgrid(X, Y)
        def mask(X, Y, training=False):
            # return np.ones(X.shape, dtype=bool)
            if training:
                return np.ones(X.shape, dtype=bool)
            return (X > 0.15)
        return X, Y, mask
    def cea_Cp_in_low_right(N=40):
        X = concspace(1, 7, 7, strength=1.0, N=N)
        Y = concspace(0.5, 4, 4, strength=2.0, N=N)
        X, Y = np.meshgrid(X, Y)
        def mask(X, Y, training=False):
            return np.ones(X.shape, dtype=bool)
        return X, Y, mask
    # "should be independant of eps"
    cea_get_Cp = np.vectorize(lambda a, b: cea.get_Chamber_Cp(a, b, 20))
    def cea_Cp_high(P, ofr):
        return cea_get_Cp(P * 1e6, ofr)
    def cea_Cp_low_left(P, ofr):
        return cea_get_Cp(P * 1e6, ofr)
    def cea_Cp_low_right(P, ofr):
        return cea_get_Cp(P * 1e6, ofr)
    # peek(cea_Cp_high, *np.meshgrid(cea_in_P(), cea_in_ofr()))

    # do(cea_Cp_high, *cea_Cp_in_high(), blitz=1.0, max_error=0.00001, printall=True)
    do(cea_Cp_high, *cea_Cp_in_high(N=100), spec=[(1, 4, 6, 7, 8, 9), (0, 1, 2, 4, 8, 9)])

    # do(cea_Cp_low_left, *cea_Cp_in_low_left(), blitz=4.0, max_error=0.00001, printall=True)
    do(cea_Cp_low_left, *cea_Cp_in_low_left(N=100), spec=[(0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 7, 8, 9)])

    # do(cea_Cp_low_right, *cea_Cp_in_low_right(), blitz=3.0, starting_cost=24, max_error=0.00001, printall=True)
    do(cea_Cp_low_right, *cea_Cp_in_low_right(N=100), spec=[(0, 1, 2, 5), (0, 1, 2, 4, 5, 8)])

    def cea_Cp_combined(P, ofr):
        if ofr >= 4:
            x1 = P
            y1 = ofr
            x1y1 = P*ofr
            x3 = P*P*P
            x2y1 = x1y1 * P
            x1y2 =x1y1 * ofr
            y3 = ofr*ofr*ofr
            p1 = 3802.3766890029965
            p4 = 959.1089167566258
            p6 = 0.9740826587244922
            p7 = 3.5480311720693734
            p8 = 75.4934686487755
            q0 = 0.06375088655715791
            q1 = 2.0937979929620574
            q2 = 0.012683649993537984
            q4 = 0.5040092283785854
            q8 = 0.03403190253379037
            q9 = 0.00023048707705274537
            P = p1*x1 - p4*x1y1 + p6*x3 - p7*x2y1 + p8*x1y2 + y3
            Q = q0 + q1*x1 - q2*y1 - q4*x1y1 + q8*x1y2 + q9*y3
            return P / Q
        if P >= 1:
            x1 = P
            y1 = ofr
            x1y1 = P*ofr
            y2 = ofr*ofr
            x1y2 = x1y1*ofr
            p0 = 4.138832624431316
            p1 = 0.29817405524316326
            p2 = 3.2575841375358316
            q0 = 0.0005884213088340639
            q1 = 7.309846342943449e-05
            q2 = 0.0007528849169182324
            q4 = 4.3580339523010414e-05
            q5 = 0.0003868623137207404
            q8 = 1.9187426863527176e-05
            P = p0 + p1*x1 - p2*y1 + y2
            Q = q0 + q1*x1 - q2*y1 - q4*x1y1 + q5*y2 + q8*x1y2
            return P / Q
        x1 = P
        y1 = ofr
        x2 = P*P
        x1y1 = P*ofr
        y2 = ofr*ofr
        x2y1 = x1y1 * P
        x1y2 = x1y1 * ofr
        y3 = ofr*ofr*ofr
        p0 = 3.0551837534331043
        p1 = 1.1168085957129472
        p2 = 3.057032048465364
        p3 = 0.08801722733929103
        p4 = 0.2480662508580779
        q0 = 0.00035410768902353467
        q1 = 0.00019003775520538854
        q2 = 0.0004712179241529132
        q4 = 0.00012521491909272612
        q5 = 0.00021512281760482563
        q7 = 4.846442773758666e-07
        q8 = 3.615723753207365e-05
        q9 = 2.406864519884499e-05
        P = p0 + p1*x1 - p2*y1 - p3*x2 - p4*x1y1 + y2
        Q = q0 + q1*x1 - q2*y1 - q4*x1y1 + q5*y2 + q7*x2y1 + q8*x1y2 + q9*y3
        return P / Q
    # compare(*np.meshgrid(cea_in_P(N=100), cea_in_ofr(N=100)), cea_Cp, np.vectorize(cea_Cp_combined))


    # should also be indep of eps? (also stupid non si return value)
    cea_get_Mw = np.vectorize(lambda a, b: 1e-3 * cea.get_Chamber_MolWt_gamma(a, b, 20)[0])
    def cea_Mw(P, ofr):
        return cea_get_Mw(P * 1e6, ofr)
    # peek(cea_Mw, *cea_in_Pofr())
    # do(cea_Mw, *cea_in_Pofr(N=40), blitz=1.0, max_error=0.00001, printall=True)
    do(cea_Mw, *cea_in_Pofr(N=100), spec=[(0, 1, 3, 4, 5), (1, 2, 3, 5)])


    plt.show()


    """
Canon output:


nox_T_sat
(0, 1) / (0, 1) ....................................... 0.1251% !!
nox_T_sat -> (3.6114 + x) / (0.016766 + 0.0025292 x)
    double x1 = ;
    double n0 = +3.6113787984517662;
    double d0 = +0.01676607833192893;
    double d1 = +0.0025292160857822753;
    double Num = n0 + x1;
    double Den = d0 + d1*x1;
    return Num / Den;

nox_rho_satliq
(0, 1, 2) / (0, 1) .................................... 0.6808% !!
nox_rho_satliq -> (1.5609e+05 - 810.65 x + x^2) / (73.7 - 0.23238 x)
    double x1 = ;
    double x2 = ;
    double n0 = +156087.8754291886;
    double n1 = -810.6533977445022;
    double d0 = +73.69992119127612;
    double d1 = -0.2323813281191897;
    double Num = n0 + n1*x1 + x2;
    double Den = d0 + d1*x1;
    return Num / Den;

nox_rho_satvap
(0, 1, 3) / (0, 1, 3) ................................. 0.07754% !!
nox_rho_satvap -> (3.3298e+07 - 2.0435e+05 x + x^3) / (-6.8652e+05 + 3146.2 x - 0.0097151 x^3)
    double x1 = ;
    double x3 = ;
    double n0 = +33298391.761922084;
    double n1 = -204351.1261299974;
    double d0 = -686517.2650492929;
    double d1 = +3146.20641220234;
    double d3 = -0.009715066047700149;
    double Num = n0 + n1*x1 + x3;
    double Den = d0 + d1*x1 + d3*x3;
    return Num / Den;

nox_P_satliq
(0, 1, 2) / (0,) ...................................... 0.8391% !!
nox_P_satliq -> (54347 - 459.63 x + x^2) / (0.0010951)
    double x1 = ;
    double x2 = ;
    double c0 = +49628368.03406795;
    double c1 = -419724.2426924475;
    double c2 = +913.1756437857401;
    return c0 + c1*x1 + c2*x2;

nox_P
(5, 7, 9) / (1,) ...................................... 0.8546% !!
nox_P -> (-1004.4 y^2 + 3.7264 x^2 y + y^3) / (0.019778 x)
    double x1 = ;
    double y2 = ;
    double x2y1 = ;
    double y3 = ;
    double n5 = -1004.4435204505293;
    double n7 = +3.7264490783988635;
    double d1 = +0.019777885554525112;
    double Num = n5*y2 + n7*x2y1 + y3;
    double Den = d1*x1;
    return Num / Den;

nox_s_satliq
(0,) / (0, 1, 2, 6) ................................... 0.8252% !!
nox_s_satliq -> (1) / (0.0024715 - 0.00047915 x + 4.4383e-05 x^2 - 3.0236e-09 x^6)
    double x1 = ;
    double x2 = ;
    double x6 = ;
    double n0 = -330735366.98577213;
    double d0 = -817420.0670422539;
    double d1 = +158470.39606104503;
    double d2 = -14678.975863932443;
    double Num = -n0;
    double Den = d0 + d1*x1 + d2*x2 + x6;
    return Num / Den;

nox_s_satvap
(0, 1, 2) / (0, 1) .................................... 0.4748% !!
nox_s_satvap -> (246.11 - 40.262 x + x^2) / (0.14145 - 0.018736 x)
    double x1 = ;
    double x2 = ;
    double n0 = +246.10928756012322;
    double n1 = -40.26239399448462;
    double d0 = +0.14145002404877624;
    double d1 = -0.018736152701492044;
    double Num = n0 + n1*x1 + x2;
    double Den = d0 + d1*x1;
    return Num / Den;

nox_cp
(1, 2, 3) / (0, 1, 2, 5) .............................. 3.573%
nox_cp -> (-116.52 x - 7300.7 y + x^2) / (-91.651 + 0.52417 x - 14.922 y + 0.71462 y^2)
    double x1 = ;
    double y1 = ;
    double x2 = ;
    double y2 = ;
    double n1 = -116.51699084548176;
    double n2 = -7300.734145907978;
    double d0 = -91.65103697146496;
    double d1 = +0.5241696296156855;
    double d2 = -14.921756479203275;
    double d5 = +0.7146187152094382;
    double Num = n1*x1 + n2*y1 + x2;
    double Den = d0 + d1*x1 + d2*y1 + d5*y2;
    return Num / Den;

nox_cv_satliq
(0, 1, 2) / (0, 1) .................................... 0.2389% !!
nox_cv_satliq -> (1.1448e+06 - 3937 x + x^2) / (1192.6 - 3.7875 x)
    double x1 = ;
    double x2 = ;
    double n0 = +1144787.6651075622;
    double n1 = -3937.01681226655;
    double d0 = +1192.576384943476;
    double d1 = -3.7875478741887108;
    double Num = n0 + n1*x1 + x2;
    double Den = d0 + d1*x1;
    return Num / Den;

nox_cv_satvap
(1,) / (0, 1, 2) ...................................... 1.341% !!
nox_cv_satvap -> (x) / (-1.7249 + 0.015009 x - 2.7648e-05 x^2)
    double x1 = ;
    double x2 = ;
    double d0 = -1.7249439211880886;
    double d1 = +0.015009354557928997;
    double d2 = -2.7648472161050912e-05;
    double Num = x1;
    double Den = d0 + d1*x1 + d2*x2;
    return Num / Den;

nox_cv
(2, 4, 5, 6) / (0, 1, 2) .............................. 1.19% !!
nox_cv -> (5.7893e+06 y - 28106 xy + 4134.1 y^2 + x^3) / (-44609 + 277.53 x - 4639.1 y)
    double x1 = ;
    double y1 = ;
    double x1y1 = ;
    double y2 = ;
    double x3 = ;
    double n2 = +5789256.963839073;
    double n4 = -28105.935930542626;
    double n5 = +4134.052836043407;
    double d0 = -44609.45406542292;
    double d1 = +277.52613563434414;
    double d2 = -4639.098422516753;
    double Num = n2*y1 + n4*x1y1 + n5*y2 + x3;
    double Den = d0 + d1*x1 + d2*y1;
    return Num / Den;

nox_h_satliq
(1,) / (0, 1) ......................................... 1.982% !
nox_h_satliq -> (x) / (0.0056002 - 1.4443e-05 x)
    double x1 = ;
    double d0 = +0.0056002005901963905;
    double d1 = -1.4442735468669312e-05;
    double Num = x1;
    double Den = d0 + d1*x1;
    return Num / Den;

nox_h_satvap
(0, 1, 2) / (0, 1) .................................... 0.34% !!
nox_h_satvap -> (4.4029e+10 - 1.3889e+08 x + x^2) / (1.0663e+05 - 334.35 x)
    double x1 = ;
    double x2 = ;
    double n0 = +44029181155.05621;
    double n1 = -138886400.2908723;
    double d0 = +106633.1549244389;
    double d1 = -334.3462973733325;
    double Num = n0 + n1*x1 + x2;
    double Den = d0 + d1*x1;
    return Num / Den;

nox_h
(1,) / (0, 1, 2) ...................................... 0.7634% !!
nox_h -> (x) / (0.00038667 + 8.2292e-07 x + 8.2072e-07 y)
    double x1 = ;
    double y1 = ;
    double d0 = +0.00038666795882217524;
    double d1 = +8.229213007432222e-07;
    double d2 = +8.20724847683098e-07;
    double Num = x1;
    double Den = d0 + d1*x1 + d2*y1;
    return Num / Den;

nox_u_satliq
(0, 1, 3, 4) / (0,) ................................... 1.392% !!
nox_u_satliq -> (-6.177e+09 + 4.3976e+07 x - 552.67 x^3 + x^4) / (849.91)
    double x1 = ;
    double x3 = ;
    double x4 = ;
    double c0 = -7267788.558102235;
    double c1 = +51741.51506234207;
    double c3 = -0.6502712552305997;
    double c4 = +0.0011765915303324403;
    return c0 + c1*x1 + c3*x3 + c4*x4;

nox_u_satvap
(0, 1, 2) / (0, 1) .................................... 0.1887% !!
nox_u_satvap -> (3.46e+10 - 1.0973e+08 x + x^2) / (93496 - 295.4 x)
    double x1 = ;
    double x2 = ;
    double n0 = +34599947743.215485;
    double n1 = -109730783.13591044;
    double d0 = +93495.74130581485;
    double d1 = -295.39713418629657;
    double Num = n0 + n1*x1 + x2;
    double Den = d0 + d1*x1;
    return Num / Den;

nox_u
(1,) / (0, 1, 2) ...................................... 0.5425% !!
nox_u -> (x) / (0.00037742 + 1.1447e-06 x + 7.2723e-07 y)
    double x1 = ;
    double y1 = ;
    double d0 = +0.00037742028086887655;
    double d1 = +1.1447255525542155e-06;
    double d2 = +7.272303965256336e-07;
    double Num = x1;
    double Den = d0 + d1*x1 + d2*y1;
    return Num / Den;

nox_Z
(0, 2, 4, 5) / (0,) ................................... 1.002% !!
nox_Z -> (3.9114e+05 - 3864.8 y + 9.0463 xy + y^2) / (3.9176e+05)
    double y1 = ;
    double x1y1 = ;
    double y2 = ;
    double c0 = +0.9984208759508151;
    double c2 = -0.009865255977849555;
    double c4 = +2.309181429551549e-05;
    double c5 = +2.5526116512486693e-06;
    return c0 + c2*y1 + c4*x1y1 + c5*y2;

cea_Tcomb_high
(0, 1, 2, 5) / (1, 4, 5, 6, 7) ........................ 4.291%
cea_Tcomb_high -> (-23.4 + 2.1208 x + 5.9214 y + y^2) / (0.001823 x - 0.00029379 xy + 0.00045936 y^2 - 9.8517e-06 x^3 + 2.2945e-05 x^2 y)
    double x1 = ;
    double y1 = ;
    double x1y1 = ;
    double y2 = ;
    double x3 = ;
    double x2y1 = ;
    double n0 = -23.39955473159082;
    double n1 = +2.1207746242104513;
    double n2 = +5.921369062404011;
    double d1 = +0.0018229851740627402;
    double d4 = -0.000293794779021038;
    double d5 = +0.0004593559262770208;
    double d6 = -9.851749432640628e-06;
    double d7 = +2.294535003217659e-05;
    double Num = n0 + n1*x1 + n2*y1 + y2;
    double Den = d1*x1 + d4*x1y1 + d5*y2 + d6*x3 + d7*x2y1;
    return Num / Den;

cea_Tcomb_low
(0, 2, 4, 5) / (1, 2, 4, 5, 7, 8) ..................... 3.876%
cea_Tcomb_low -> (-0.33089 + 7.5862 y + 8.8088 xy + y^2) / (0.00022362 x + 0.0094414 y + 0.0078956 xy - 0.0011657 y^2 - 1.3745e-05 x^2 y - 0.0010261 xy^2)
    double x1 = ;
    double y1 = ;
    double x1y1 = ;
    double y2 = ;
    double x2y1 = ;
    double x1y2 = ;
    double n0 = -0.3308903646269448;
    double n2 = +7.586197253157754;
    double n4 = +8.808819483900445;
    double d1 = +0.00022361880493252257;
    double d2 = +0.00944138717224415;
    double d4 = +0.007895570851432925;
    double d5 = -0.0011657313243630958;
    double d7 = -1.3745327377942885e-05;
    double d8 = -0.001026130067031718;
    double Num = n0 + n2*y1 + n4*x1y1 + y2;
    double Den = d1*x1 + d2*y1 + d4*x1y1 + d5*y2 + d7*x2y1 + d8*x1y2;
    return Num / Den;

cea_Cp_high
(1, 4, 6, 7, 8, 9) / (0, 1, 2, 4, 8, 9) ............... 4.766%
cea_Cp_high -> (3802.4 x - 959.11 xy + 0.97408 x^3 - 3.548 x^2 y + 75.493 xy^2 + y^3) / (0.063751 + 2.0938 x - 0.012684 y - 0.50401 xy + 0.034032 xy^2 + 0.00023049 y^3)
    double x1 = ;
    double y1 = ;
    double x1y1 = ;
    double x3 = ;
    double x2y1 = ;
    double x1y2 = ;
    double y3 = ;
    double n1 = +3802.3766890029965;
    double n4 = -959.1089167566258;
    double n6 = +0.9740826587244922;
    double n7 = -3.5480311720693734;
    double n8 = +75.4934686487755;
    double d0 = +0.06375088655715791;
    double d1 = +2.0937979929620574;
    double d2 = -0.012683649993537984;
    double d4 = -0.5040092283785854;
    double d8 = +0.03403190253379037;
    double d9 = +0.00023048707705274537;
    double Num = n1*x1 + n4*x1y1 + n6*x3 + n7*x2y1 + n8*x1y2 + y3;
    double Den = d0 + d1*x1 + d2*y1 + d4*x1y1 + d8*x1y2 + d9*y3;
    return Num / Den;

cea_Cp_low_left
(0, 1, 2, 3, 4, 5) / (0, 1, 2, 4, 5, 7, 8, 9) ......... 11.83%
cea_Cp_low_left -> (3.0552 + 1.1168 x - 3.057 y - 0.088017 x^2 - 0.24807 xy + y^2) / (0.00035411 + 0.00019004 x - 0.00047122 y - 0.00012521 xy + 0.00021512 y^2 + 4.8464e-07 x^2 y + 3.6157e-05 xy^2 + 2.4069e-05 y^3)
    double x1 = ;
    double y1 = ;
    double x2 = ;
    double x1y1 = ;
    double y2 = ;
    double x2y1 = ;
    double x1y2 = ;
    double y3 = ;
    double n0 = +3.0551837534331043;
    double n1 = +1.1168085957129472;
    double n2 = -3.057032048465364;
    double n3 = -0.08801722733929103;
    double n4 = -0.2480662508580779;
    double d0 = +0.00035410768902353467;
    double d1 = +0.00019003775520538854;
    double d2 = -0.0004712179241529132;
    double d4 = -0.00012521491909272612;
    double d5 = +0.00021512281760482563;
    double d7 = +4.846442773758666e-07;
    double d8 = +3.615723753207365e-05;
    double d9 = +2.406864519884499e-05;
    double Num = n0 + n1*x1 + n2*y1 + n3*x2 + n4*x1y1 + y2;
    double Den = d0 + d1*x1 + d2*y1 + d4*x1y1 + d5*y2 + d7*x2y1 + d8*x1y2 + d9*y3;
    return Num / Den;

cea_Cp_low_right
(0, 1, 2, 5) / (0, 1, 2, 4, 5, 8) ..................... 5.754%
cea_Cp_low_right -> (4.1388 + 0.29817 x - 3.2576 y + y^2) / (0.00058842 + 7.3098e-05 x - 0.00075288 y - 4.358e-05 xy + 0.00038686 y^2 + 1.9187e-05 xy^2)
    double x1 = ;
    double y1 = ;
    double x1y1 = ;
    double y2 = ;
    double x1y2 = ;
    double n0 = +4.138832624431316;
    double n1 = +0.29817405524316326;
    double n2 = -3.2575841375358316;
    double d0 = +0.0005884213088340639;
    double d1 = +7.309846342943449e-05;
    double d2 = -0.0007528849169182324;
    double d4 = -4.3580339523010414e-05;
    double d5 = +0.0003868623137207404;
    double d8 = +1.9187426863527176e-05;
    double Num = n0 + n1*x1 + n2*y1 + y2;
    double Den = d0 + d1*x1 + d2*y1 + d4*x1y1 + d5*y2 + d8*x1y2;
    return Num / Den;

cea_Mw
(0, 1, 3, 4, 5) / (1, 2, 3, 5) ........................ 4.911%
cea_Mw -> (0.18418 + 0.77809 x + 0.0058258 x^2 + 0.14145 xy + y^2) / (64.334 x + 66.804 y + 0.15251 x^2 + 30.326 y^2)
    double x1 = ;
    double y1 = ;
    double x2 = ;
    double x1y1 = ;
    double y2 = ;
    double n0 = +0.1841818441210476;
    double n1 = +0.7780884860427217;
    double n3 = +0.005825775742224297;
    double n4 = +0.14145416999491983;
    double d1 = +64.3339970083186;
    double d2 = +66.80439701942177;
    double d3 = +0.15251463574592666;
    double d5 = +30.325772888462538;
    double Num = n0 + n1*x1 + n3*x2 + n4*x1y1 + y2;
    double Den = d1*x1 + d2*y1 + d3*x2 + d5*y2;
    return Num / Den;

    """


if __name__ == "__main__":
    _main()

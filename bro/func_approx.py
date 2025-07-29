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
                if last / dims > (1 + 1.0/blitz) * length:
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

    def cook(self, ones, *, coeffs=None):
        if coeffs is None:
            coeffs = self.coeffs
        if len(coeffs) != self.count:
            raise ValueError(f"expected {self.count} coeffs, "
                    f"got {len(coeffs)}")
        if ones.ndim != 2 or ones.shape[-1] < self.countall:
            raise ValueError("expected yup_all_ones shape, "
                    f"got {ones.shape}")
        return np.sum(coeffs * ones[:, self.idxs], axis=-1)

    def as_numpy(self):
        if self.dims != 1:
            raise ValueError("numpy cannot do multivariate polynomials")
        poly = np.zeros(self.degree + 1, dtype=float)
        for idx, coeff in zip(self.idxs, self.coeffs):
            poly[len(poly) - 1 - idx] = coeff
        return poly

    def __repr__(self, short):
        allcoeffs = np.zeros(self.countall)
        allcoeffs[self.idxs] = self.coeffs
        variables = poly_ordering(self.dims, self.countall)
        def mul(c, x):
            if c == 0:
                return ""
            if c == 1:
                return x
            c = f"{c:.5g}" if short else repr(float(c))
            return f"{c} {x}".strip()
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

    def _get_coeffs(self, coeffs):
        pcoeffs = coeffs[:self.p.count - 1]
        pcoeffs = np.append(pcoeffs, 1.0)
        qcoeffs = coeffs[self.p.count - 1:]
        return pcoeffs, qcoeffs
    def set_coeffs(self, new_coeffs):
        try:
            self.p.coeffs, self.q.coeffs = self._get_coeffs(new_coeffs)
        except:
            if hasattr(self.p, "coeffs"):
                del self.p.coeffs
            if hasattr(self.q, "coeffs"):
                del self.q.coeffs
            raise

    def _error(self, real, span, approx, maximum=True, noabserr=False):
        relative = np.abs((approx - real) / span)
        absolute = np.abs(approx / real - 1)
        if noabserr:
            error = relative
        else:
            error = (relative + absolute) / 2
        if maximum:
            error = np.max(error)
        return error

    def ones(self, *flatcoords):
        return yup_all_ones(max(self.p.degree, self.q.degree), *flatcoords)

    def cook(self, ones, *, coeffs=None):
        if coeffs is not None:
            pcoeffs, qcoeffs = self._get_coeffs(coeffs)
        else:
            pcoeffs = self.p.coeffs
            qcoeffs = self.q.coeffs
        P = self.p.cook(ones, coeffs=pcoeffs)
        Q = self.q.cook(ones, coeffs=qcoeffs)
        with np.errstate(divide="ignore"):
            return P / Q

    def approximate(self, real, span, ones, noabserr=False):
        # Optimise via least squares, since its significantly more stable
        # than a minimise-max-error optimiser.
        def diff(coeffs):
            pcoeffs, qcoeffs = self._get_coeffs(coeffs)
            P = self.p.cook(ones, coeffs=pcoeffs)
            Q = self.q.cook(ones, coeffs=qcoeffs)
            with np.errstate(divide="ignore"):
                approx = P / Q
            relative = (approx - real) / span
            absolute = approx / real - 1
            # We could just return delta here and it would optimise just
            # fine to fit the data points we've given, but it doesn't
            # take any heed of poles over our input range in that case.
            # It's entirely possible (im saying that cause shit happened
            # to me) for a solution to be produced which fits the points
            # very well but has the slimmest fucking asymptote right in
            # the middle. To prevent this, we penalise small denominators
            # (which shouldnt lose any solutions, since the numerator can
            # just make up any difference?).
            penalty = 1.0e-6 / np.abs(Q)
            if noabserr:
                return 2*relative + penalty
            return relative + absolute + penalty
        coeffs = self.initial_coeffs()
        res = scipy.optimize.least_squares(diff, coeffs, method="lm")
        coeffs = res.x
        self.set_coeffs(coeffs)
        # Return max difference.
        return self._error(real, span, self.cook(ones), noabserr=noabserr)


    def havealook(self, real, *coords, mask=None, threed=False, figtitle=None, noabserr=False):
        flatcoords = [x.ravel() for x in coords]
        if self.dims != len(coords):
            raise ValueError(f"expected {self.dims} dims, got {len(coords)}")
        if self.dims != 1 and self.dims != 2:
            raise ValueError(f"huh {self.dims}")
        ones = self.ones(*flatcoords)
        approx = self.cook(ones).reshape(coords[0].shape)
        span = np.max(real) - np.min(real)
        error = 100 * self._error(real, span, approx, maximum=False, noabserr=noabserr)

        if mask is not None:
            real[~mask] = 0.0
            approx[~mask] = 0.0
            error[~mask] = 0.0

        if len(coords) == 1 or threed:
            plotme = [
                ([
                    (real, "real", ("b", "Blues")),
                    (approx, "approx", ("orange", "Oranges")),
                ], "Approximation [real=blue, approx=orange]"),
                (error, "error"),
            ]
        else:
            plotme = [
                (real, "Real"),
                (approx, "Approximation"),
                (error, "error"),
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
    def of(cls, real, *coords, spec=None, blitz=False, starting_cost=2, noabserr=False):
        dims = len(coords)
        flatcoords = [x.ravel() for x in coords]
        real = real.ravel()

        if any((x == 0.0).any() for x in flatcoords):
            raise Exception("zero!")
        if any(((x > 0.0) != (x[0] > 0.0)).any() for x in flatcoords):
            raise Exception("zero?")
        span = np.max(real) - np.min(real)

        padto = 55
        N = 6
        onesN = yup_all_ones(N, *flatcoords)

        if spec is None:
            idxs = cls._all_idx_tuples(dims, blitz=blitz, starting_cost=starting_cost)
        else:
            idxs = [spec]

        # Iterate through the rat polys in cost order, where the cost of
        # the rat poly is just the sum of each polys cost.
        best = float("inf")
        for pidxs, qidxs in idxs:
            self = cls(dims, pidxs, qidxs)
            maxdegree = max(self.p.degree, self.q.degree)
            if maxdegree <= N:
                ones = onesN
            else:
                ones = yup_all_ones(maxdegree, *flatcoords)

            s = f"{pidxs} / {qidxs}"
            s += " " * (padto - len(s))
            print(s, end="\r")
            error = self.approximate(real, span, ones, noabserr=noabserr)
            if error >= max(best, 0.02):
                continue
            start = f"{pidxs} / {qidxs} .."
            start += "." * (padto - len(start))
            start += f" {100 * error:.4g}%"
            print(start)
            if spec is not None:
                return self
            best = min(best, error)

    def __repr__(self, short=True):
        return f"({self.p.__repr__(short)}) / ({self.q.__repr__(short)})"

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
                    s += f"    f64 {name} = ;\n"

        if q.idxs == [0]:
            coeffs = p.coeffs / q.coeffs[0]
            # just normal poly.
            ones = []
            for i, coeff in zip(p.idxs, coeffs):
                ones.append(abs(coeff) == 1.0 and xnames[i])
                if ones[-1]:
                    continue
                s += f"    f64 c{i} = {ftoa(coeff)};\n"
            cs = [f"c{i}" for i in p.idxs]
            xs = [xnames[i] for i in p.idxs]
            s += f"    return {poly(ones, cs, xs)};"
            return s
        if p.idxs == [0]:
            # inverse poly.
            numer = float(p.coeffs[0] / q.coeffs[-1])
            coeffs = q.coeffs / q.coeffs[-1]
            ones = []
            s += f"    f64 n0 = {ftoa(numer)};\n"
            for i, coeff in zip(q.idxs, coeffs):
                ones.append(abs(coeff) == 1.0 and xnames[i])
                if ones[-1]:
                    continue
                s += f"    f64 d{i} = {ftoa(coeff)};\n"
            cs = [f"d{i}" for i in q.idxs]
            xs = [xnames[i] for i in q.idxs]
            s += f"    f64 Num = {'-'*(numer < 0)}n0;\n"
            s += f"    f64 Den = {poly(ones, cs, xs)};\n"
            s += "    return Num / Den;"
            return s

        pones = []
        for i, coeff in zip(p.idxs, p.coeffs):
            pones.append(abs(coeff) == 1.0 and xnames[i])
            if pones[-1]:
                continue
            s += f"    f64 n{i} = {ftoa(coeff)};\n"
        qones = []
        for i, coeff in zip(q.idxs, q.coeffs):
            qones.append(abs(coeff) == 1.0 and xnames[i])
            if qones[-1]:
                continue
            s += f"    f64 d{i} = {ftoa(coeff)};\n"
        pcs = [f"n{i}" for i in p.idxs]
        qcs = [f"d{i}" for i in q.idxs]
        pxs = [xnames[i] for i in p.idxs]
        qxs = [xnames[i] for i in q.idxs]
        s += f"    f64 Num = {poly(pones, pcs, pxs)};\n"
        s += f"    f64 Den = {poly(qones, qcs, qxs)};\n"
        s += f"    return Num / Den;"
        return s




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


def do(func, X, Y=None, maskf=None, plotme=True, **of_kwargs):
    haskw = not not of_kwargs
    print(func.__name__ + ": "*(haskw) + repr(of_kwargs)*haskw)
    Xs = (X.ravel(),) if Y is None else (X.ravel(), Y.ravel())
    bounds = [(X.min(), X.max()) for X in Xs]
    real = func(*Xs)
    if maskf is not None:
        mask = maskf(*Xs, training=True)
        Xs = [X[mask] for X in Xs]
        real = real[mask]
    ratpoly = RationalPolynomial.of(real, *Xs, **of_kwargs)
    if Y is None:
        # can easily check that 1d has no poles over input range.
        xmin, xmax = bounds[0]
        roots = np.roots(ratpoly.q.as_numpy())
        if ((xmin <= roots) & (roots <= xmax)).any():
            raise ValueError(f"poly wanna pole: {ratpoly.__repr__(False)}")
    if plotme:
        N = 600 if Y is None else 100
        fines = [np.linspace(lo, hi, N) for lo, hi in bounds]
        fines = np.meshgrid(*fines)
        realfine = func(*[x.ravel() for x in fines]).reshape(fines[0].shape)
        maskfine = maskf(*fines) if maskf is not None else None
        noabserr = of_kwargs.get("noabserr", False)
        ratpoly.havealook(realfine, *fines, mask=maskfine, figtitle=func.__name__,
                noabserr=noabserr)
    print(func.__name__, "->", ratpoly)
    print(ratpoly.cython())
    print()
def peek1d(func, X):
    Y = func(X)
    fig = plt.figure(figsize=(8, 5))
    fig.suptitle(func.__name__)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X, Y)
    plt.tight_layout()
    plt.show()
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
    plt.show()
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

def nox():
    from CoolProp.CoolProp import PropsSI

    # nox triple point:     182.34 K  0.08785 MPa
    # nox critical point:   309.55 K  7.245 MPa
    # all nox mixture or vap temperatures reasonably within 183 .. 309 K.
    # all nox mixture or vap pressures reasonably within 0.09 .. 7.2 MPa.

    def nox_in_T(conc=None, strength=0.0, N=160):
        # note 182.34K is nox triple point.
        # note 309.55K is nox critical point.
        return concspace(183, conc, 309, strength=strength, N=N)
    def nox_in_P(conc=None, strength=0.0, N=160):
        # note 87.85kPa is nox triple point.
        # note 72.45MPa is nox critical point.
        return concspace(0.09, conc, 7.2, strength=strength, N=N)
    def nox_in_rho(conc=None, strength=0.0, N=160):
        # density of vapour reasonably within 1..325 kg/m^3.
        return concspace(1, conc, 325, strength=strength, N=N)
    def nox_in_TP(N=60, Tconc=None, Tstrength=0.0, Pconc=None, Pstrength=0.0,
            trim_corner=False):
        X = nox_in_T(conc=Tconc, strength=Tstrength, N=N)
        Y = nox_in_P(conc=Pconc, strength=Pstrength, N=N)
        X, Y = np.meshgrid(X, Y)
        # but its never gonna be higher pressure than Psat for a temp.
        def mask(X, Y, training=False):
            Psat = PropsSI("P", "T", X.ravel(), "Q", 1, "N2O")
            Psat = Psat.reshape(Y.shape)
            Psat /= 1e6
            mask = (Y < Psat)
            # i do not fucking know what is going on in critical-point corner
            # of cp but its so cooked.
            if training and trim_corner:
                mask &= (Y < 6.6)
            # if training:
            #     mask &= (Y > Psat - 4)
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
    # do(nox_T_sat, nox_in_P())
    do(nox_T_sat, nox_in_P(), spec=[(0, 1, 2), (0, 1, 2)])

    def nox_rho_satliq(T):
        return PropsSI("D", "T", T, "Q", 0, "N2O")
    # do(nox_rho_satliq, nox_in_T(conc=309, strength=2.0))
    do(nox_rho_satliq, nox_in_T(conc=309, strength=2.0), spec=[(0, 1, 2), (0, 1, 2)])

    def nox_rho_satvap(T):
        return PropsSI("D", "T", T, "Q", 1, "N2O")
    # so ill behaved :(
    # do(nox_rho_satvap, nox_in_T(conc=309, strength=3.0))
    do(nox_rho_satvap, nox_in_T(conc=309, strength=6.0), spec=[(0, 1, 2, 4), (0, 1, 4)])

    def nox_P_satliq(T):
        return PropsSI("P", "T", T, "Q", 0, "N2O")
    # do(nox_P_satliq, nox_in_T())
    do(nox_P_satliq, nox_in_T(), spec=[(1, 3, 5), (0,)])

    def nox_P(T, rho):
        return PropsSI("P", "T", T, "D", rho, "N2O")
    # peek(nox_P, *nox_in_Trho())
    # do(nox_P, *nox_in_Trho())
    do(nox_P, *nox_in_Trho(), spec=[(5, 7, 9), (1,)])

    # Note that zero entropy is not noteworthy, so disable cost from comparing
    # absolute proportions.
    def nox_s_satliq(P):
        return PropsSI("S", "P", P * 1e6, "Q", 0, "N2O")
    # do(nox_s_satliq, nox_in_P(), noabserr=True, blitz=1.5)
    do(nox_s_satliq, nox_in_P(), noabserr=True, spec=[(0, 1, 4, 5), (0, 1, 2)])
    def nox_s_satvap(P):
        return PropsSI("S", "P", P * 1e6, "Q", 1, "N2O")
    # do(nox_s_satvap, nox_in_P(), noabserr=True, blitz=1.5)
    do(nox_s_satvap, nox_in_P(), noabserr=True, spec=[(0, 1, 2, 3), (0, 1, 2)])

    def nox_cp(T, P):
        return PropsSI("C", "T", T, "P", P * 1e6, "N2O")
    # idk wtf is happening around the critical point for this but its cooked.
    # peek(nox_cp, *nox_in_TP(N=300, trim_corner=True))
    # do(nox_cp, *nox_in_TP(N=140, trim_corner=True), blitz=1.0)
    do(nox_cp, *nox_in_TP(N=140, trim_corner=True), spec=[(1, 2, 3), (1, 2, 3, 5)])

    def nox_cv_satliq(T):
        return PropsSI("O", "T", T, "Q", 0, "N2O")
    # do(nox_cv_satliq, nox_in_T(), blitz=1.5)
    do(nox_cv_satliq, nox_in_T(), spec=[(0, 1, 3), (0, 2, 3)])
    def nox_cv_satvap(T):
        return PropsSI("O", "T", T, "Q", 1, "N2O")
    # do(nox_cv_satvap, nox_in_T(), blitz=1.5)
    do(nox_cv_satvap, nox_in_T(), spec=[(1, 2, 3), (0, 1)])

    def nox_cv(T, P):
        return PropsSI("O", "T", T, "P", P * 1e6, "N2O")
    # peek(nox_cv, *nox_in_TP(N=300))
    # do(nox_cv, *nox_in_TP(N=140), blitz=4.0)
    do(nox_cv, *nox_in_TP(), spec=[(2, 3, 4, 5), (0, 1, 4)])

    def nox_h_satliq(T):
        return PropsSI("H", "T", T, "Q", 0, "N2O")
    # do(nox_h_satliq, nox_in_T(conc=309, strength=2.0), noabserr=True)
    do(nox_h_satliq, nox_in_T(conc=309, strength=2.0), noabserr=True, spec=[(0, 1, 3, 4), (0, 1)])

    def nox_h_satvap(T):
        return PropsSI("H", "T", T, "Q", 1, "N2O")
    # do(nox_h_satvap, nox_in_T(), noabserr=True)
    do(nox_h_satvap, nox_in_T(), noabserr=True, spec=[(0, 1, 4), (0, 1, 2)])

    def nox_h(T, rho):
        return PropsSI("H", "T", T, "D", rho, "N2O")
    # peek(nox_h, *nox_in_Trho())
    # do(nox_h, *nox_in_Trho(), noabserr=True, blitz=1.5)
    do(nox_h, *nox_in_Trho(), noabserr=True, spec=[(1, 3), (0, 1, 2)])

    def nox_u_satliq(T):
        return PropsSI("U", "T", T, "Q", 0, "N2O")
    # do(nox_u_satliq, nox_in_T(conc=309, strength=4.0), noabserr=True, blitz=1.5)
    do(nox_u_satliq, nox_in_T(conc=309, strength=4.0), noabserr=True, spec=[(0, 1, 2, 3), (0, 1)])
    def nox_u_satvap(T):
        return PropsSI("U", "T", T, "Q", 1, "N2O")
    # do(nox_u_satvap, nox_in_T(conc=309, strength=4.0), noabserr=True)
    do(nox_u_satvap, nox_in_T(conc=309, strength=4.0), noabserr=True, spec=[(0, 1, 4), (0, 1, 2)])

    def nox_u(T, rho):
        return PropsSI("U", "T", T, "D", rho, "N2O")
    # peek(nox_u, *nox_in_Trho())
    # do(nox_u, *nox_in_Trho(), noabserr=True, blitz=1.5)
    do(nox_u, *nox_in_Trho(), noabserr=True, spec=[(1, 2, 3), (0, 1, 2)])

    def nox_Z(T, rho):
        return PropsSI("Z", "T", T, "D", rho, "N2O")
    # peek(nox_Z, *nox_in_Trho())
    # do(nox_Z, *nox_in_Trho())
    do(nox_Z, *nox_in_Trho(), spec=[(0, 2, 4, 5), (0,)])



def cea():
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
        # chamber pressure always lower than tank, just use same bounds.
        return concspace(0.09, conc, 7.2, strength=strength, N=N)
    def cea_in_ofr(conc=None, strength=0.0, N=120):
        # ox-fuel ratio reasonably below 13.
        return concspace(0.5, conc, 13, strength=strength, N=N)
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

    # do(cea_Tcomb_high, *cea_Tcomb_in_high(), blitz=4.0, starting_cost=27)
    do(cea_Tcomb_high, *cea_Tcomb_in_high(), spec=[(0, 1, 2, 4, 5), (1, 4, 5, 7, 8)])

    # do(cea_Tcomb_low, *cea_Tcomb_in_low(), blitz=4.0, starting_cost=27)
    do(cea_Tcomb_low, *cea_Tcomb_in_low(), spec=[(0, 1, 2), (0, 1, 2, 4)])


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
            return np.ones(X.shape, dtype=bool)
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

    # do(cea_Cp_high, *cea_Cp_in_high(), blitz=4.0, starting_cost=30)
    do(cea_Cp_high, *cea_Cp_in_high(N=100), spec=[(1, 4, 6, 7, 8, 9), (0, 1, 2, 4, 8, 9)])

    # do(cea_Cp_low_left, *cea_Cp_in_low_left(), blitz=4.0, starting_cost=26)
    do(cea_Cp_low_left, *cea_Cp_in_low_left(N=100), spec=[(0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 7, 8, 9)])

    # do(cea_Cp_low_right, *cea_Cp_in_low_right(), blitz=3.0, starting_cost=24)
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
    # do(cea_Mw, *cea_in_Pofr(N=40), blitz=1.0)
    do(cea_Mw, *cea_in_Pofr(N=100), spec=[(0, 1, 3, 4, 5), (1, 2, 3, 5)])



def main():

    nox()
    cea()
    plt.show()

    """
Canon output:


nox_T_sat: {'spec': [(0, 1, 2), (0, 1, 2)]}
(0, 1, 2) / (0, 1, 2) ................................. 0.4313%
nox_T_sat -> (1.0537 + 5 x + x^2) / (0.0063155 + 0.021209 x + 0.0024803 x^2)
    f64 x1 = ;
    f64 x2 = ;
    f64 n0 = +1.053682149480346;
    f64 n1 = +5.000029230803551;
    f64 d0 = +0.0063155487831695655;
    f64 d1 = +0.021208661457396805;
    f64 d2 = +0.002480330679198678;
    f64 Num = n0 + n1*x1 + x2;
    f64 Den = d0 + d1*x1 + d2*x2;
    return Num / Den;

nox_rho_satliq: {'spec': [(0, 1, 2), (0, 1, 2)]}
(0, 1, 2) / (0, 1, 2) ................................. 1.143%
nox_rho_satliq -> (1.2231e+05 - 703.83 x + x^2) / (78.843 - 0.39477 x + 0.00045763 x^2)
    f64 x1 = ;
    f64 x2 = ;
    f64 n0 = +122309.80087089837;
    f64 n1 = -703.8266867155928;
    f64 d0 = +78.84320429264461;
    f64 d1 = -0.3947682772066382;
    f64 d2 = +0.0004576329029363305;
    f64 Num = n0 + n1*x1 + x2;
    f64 Den = d0 + d1*x1 + d2*x2;
    return Num / Den;

nox_rho_satvap: {'spec': [(0, 1, 2, 4), (0, 1, 4)]}
(0, 1, 2, 4) / (0, 1, 4) .............................. 0.7277%
nox_rho_satvap -> (-5.6166e+09 + 7.5962e+07 x - 2.8333e+05 x^2 + x^4) / (-1.0235e+08 + 4.1206e+05 x - 0.0027648 x^4)
    f64 x1 = ;
    f64 x2 = ;
    f64 x4 = ;
    f64 n0 = -5616563626.488913;
    f64 n1 = +75961789.44613385;
    f64 n2 = -283334.26225138275;
    f64 d0 = -102348183.29992932;
    f64 d1 = +412063.5030797808;
    f64 d4 = -0.002764818539542846;
    f64 Num = n0 + n1*x1 + n2*x2 + x4;
    f64 Den = d0 + d1*x1 + d4*x4;
    return Num / Den;

nox_P_satliq: {'spec': [(1, 3, 5), (0,)]}
(1, 3, 5) / (0,) ...................................... 0.707%
nox_P_satliq -> (7.3557e+08 x - 52268 x^3 + x^5) / (2.104e+05)
    f64 x1 = ;
    f64 x3 = ;
    f64 x5 = ;
    f64 c1 = +3496.03701234121;
    f64 c3 = -0.248418557185819;
    f64 c5 = +4.752823512296013e-06;
    return c1*x1 + c3*x3 + c5*x5;

nox_P: {'spec': [(5, 7, 9), (1,)]}
(5, 7, 9) / (1,) ...................................... 0.7166%
nox_P -> (-1002.4 y^2 + 3.7138 x^2 y + y^3) / (0.019689 x)
    f64 x1 = ;
    f64 y2 = ;
    f64 x2y1 = ;
    f64 y3 = ;
    f64 n5 = -1002.3823520461443;
    f64 n7 = +3.7137717864729005;
    f64 d1 = +0.019689134939176935;
    f64 Num = n5*y2 + n7*x2y1 + y3;
    f64 Den = d1*x1;
    return Num / Den;

nox_s_satliq: {'noabserr': True, 'spec': [(0, 1, 4, 5), (0, 1, 2)]}
(0, 1, 4, 5) / (0, 1, 2) .............................. 1.187%
nox_s_satliq -> (-355.88 + 3779.5 x - 13.185 x^4 + x^5) / (2.6823 + 6.0341 x - 0.70098 x^2)
    f64 x1 = ;
    f64 x2 = ;
    f64 x4 = ;
    f64 x5 = ;
    f64 n0 = -355.8755188448979;
    f64 n1 = +3779.467529165695;
    f64 n4 = -13.18529083615938;
    f64 d0 = +2.6822755391518727;
    f64 d1 = +6.034067534443412;
    f64 d2 = -0.7009834045855325;
    f64 Num = n0 + n1*x1 + n4*x4 + x5;
    f64 Den = d0 + d1*x1 + d2*x2;
    return Num / Den;

nox_s_satvap: {'noabserr': True, 'spec': [(0, 1, 2, 3), (0, 1, 2)]}
(0, 1, 2, 3) / (0, 1, 2) .............................. 1.123%
nox_s_satvap -> (106.3 + 275.88 x - 45.721 x^2 + x^3) / (0.049355 + 0.16833 x - 0.022834 x^2)
    f64 x1 = ;
    f64 x2 = ;
    f64 x3 = ;
    f64 n0 = +106.29753009318405;
    f64 n1 = +275.8845939008316;
    f64 n2 = -45.72111883807674;
    f64 d0 = +0.04935480191796069;
    f64 d1 = +0.16833373623393996;
    f64 d2 = -0.022833528970010734;
    f64 Num = n0 + n1*x1 + n2*x2 + x3;
    f64 Den = d0 + d1*x1 + d2*x2;
    return Num / Den;

nox_cp: {'spec': [(1, 2, 3), (1, 2, 3, 5)]}
(1, 2, 3) / (1, 2, 3, 5) .............................. 7.335%
nox_cp -> (-159.82 x - 4999.7 y + x^2) / (-0.16875 x - 9.7861 y + 0.0011045 x^2 + 0.3327 y^2)
    f64 x1 = ;
    f64 y1 = ;
    f64 x2 = ;
    f64 y2 = ;
    f64 n1 = -159.82009434320338;
    f64 n2 = -4999.721561359963;
    f64 d1 = -0.16874605625391056;
    f64 d2 = -9.78607781596831;
    f64 d3 = +0.001104532552674854;
    f64 d5 = +0.3326972587207335;
    f64 Num = n1*x1 + n2*y1 + x2;
    f64 Den = d1*x1 + d2*y1 + d3*x2 + d5*y2;
    return Num / Den;

nox_cv_satliq: {'spec': [(0, 1, 3), (0, 2, 3)]}
(0, 1, 3) / (0, 2, 3) ................................. 0.8151%
nox_cv_satliq -> (9.9999e+07 - 4.1729e+05 x + x^3) / (67167 - 1.6299 x^2 + 0.0030158 x^3)
    f64 x1 = ;
    f64 x2 = ;
    f64 x3 = ;
    f64 n0 = +99998624.02836193;
    f64 n1 = -417287.50625049777;
    f64 d0 = +67167.41562085839;
    f64 d2 = -1.6299051411280265;
    f64 d3 = +0.0030157641836329814;
    f64 Num = n0 + n1*x1 + x3;
    f64 Den = d0 + d2*x2 + d3*x3;
    return Num / Den;

nox_cv_satvap: {'spec': [(1, 2, 3), (0, 1)]}
(1, 2, 3) / (0, 1) .................................... 0.8734%
nox_cv_satvap -> (1.0532e+06 x - 3536 x^2 + x^3) / (3.1998e+05 - 986.82 x)
    f64 x1 = ;
    f64 x2 = ;
    f64 x3 = ;
    f64 n1 = +1053181.5139038756;
    f64 n2 = -3536.041105280067;
    f64 d0 = +319981.2745750608;
    f64 d1 = -986.8229772784155;
    f64 Num = n1*x1 + n2*x2 + x3;
    f64 Den = d0 + d1*x1;
    return Num / Den;

nox_cv: {'spec': [(2, 3, 4, 5), (0, 1, 4)]}
(2, 3, 4, 5) / (0, 1, 4) .............................. 3.673%
nox_cv -> (404.17 y + 0.011623 x^2 - 1.6303 xy + y^2) / (-0.62477 + 0.0071391 x - 0.00053361 xy)
    f64 x1 = ;
    f64 y1 = ;
    f64 x2 = ;
    f64 x1y1 = ;
    f64 y2 = ;
    f64 n2 = +404.17179888559093;
    f64 n3 = +0.0116226190297697;
    f64 n4 = -1.6302745906659892;
    f64 d0 = -0.6247719294046308;
    f64 d1 = +0.007139106757436227;
    f64 d4 = -0.0005336079270844953;
    f64 Num = n2*y1 + n3*x2 + n4*x1y1 + y2;
    f64 Den = d0 + d1*x1 + d4*x1y1;
    return Num / Den;

nox_h_satliq: {'noabserr': True, 'spec': [(0, 1, 3, 4), (0, 1)]}
(0, 1, 3, 4) / (0, 1) ................................. 0.9087%
nox_h_satliq -> (-1.5548e+12 + 1.0249e+10 x - 53363 x^3 + x^4) / (7.176e+06 - 22685 x)
    f64 x1 = ;
    f64 x3 = ;
    f64 x4 = ;
    f64 n0 = -1554844073854.8723;
    f64 n1 = +10248538986.948492;
    f64 n3 = -53363.27024534557;
    f64 d0 = +7175993.87640348;
    f64 d1 = -22685.089801817525;
    f64 Num = n0 + n1*x1 + n3*x3 + x4;
    f64 Den = d0 + d1*x1;
    return Num / Den;

nox_h_satvap: {'noabserr': True, 'spec': [(0, 1, 4), (0, 1, 2)]}
(0, 1, 4) / (0, 1, 2) ................................. 1.102%
nox_h_satvap -> (3.6007e+10 - 1.4583e+08 x + x^4) / (1.3309e+05 - 784.8 x + 1.1478 x^2)
    f64 x1 = ;
    f64 x2 = ;
    f64 x4 = ;
    f64 n0 = +36006860815.96613;
    f64 n1 = -145833527.21804488;
    f64 d0 = +133090.43360716323;
    f64 d1 = -784.7985028166469;
    f64 d2 = +1.1478421530010148;
    f64 Num = n0 + n1*x1 + x4;
    f64 Den = d0 + d1*x1 + d2*x2;
    return Num / Den;

nox_h: {'noabserr': True, 'spec': [(1, 3), (0, 1, 2)]}
(1, 3) / (0, 1, 2) .................................... 1.852%
nox_h -> (72.326 x + x^2) / (-0.051735 + 0.00095899 x + 0.00030575 y)
    f64 x1 = ;
    f64 y1 = ;
    f64 x2 = ;
    f64 n1 = +72.32632139879598;
    f64 d0 = -0.051735087047641605;
    f64 d1 = +0.000958993783893223;
    f64 d2 = +0.0003057500026811124;
    f64 Num = n1*x1 + x2;
    f64 Den = d0 + d1*x1 + d2*y1;
    return Num / Den;

nox_u_satliq: {'noabserr': True, 'spec': [(0, 1, 2, 3), (0, 1)]}
(0, 1, 2, 3) / (0, 1) ................................. 1.489%
nox_u_satliq -> (-5.7129e+12 + 4.8367e+10 x - 9.4711e+07 x^2 + x^3) / (1.7576e+07 - 54576 x)
    f64 x1 = ;
    f64 x2 = ;
    f64 x3 = ;
    f64 n0 = -5712868562742.414;
    f64 n1 = +48367409598.93219;
    f64 n2 = -94710942.69677454;
    f64 d0 = +17576470.543787282;
    f64 d1 = -54575.615844086264;
    f64 Num = n0 + n1*x1 + n2*x2 + x3;
    f64 Den = d0 + d1*x1;
    return Num / Den;

nox_u_satvap: {'noabserr': True, 'spec': [(0, 1, 4), (0, 1, 2)]}
(0, 1, 4) / (0, 1, 2) ................................. 1.969%
nox_u_satvap -> (4.0973e+10 - 1.6163e+08 x + x^4) / (1.6352e+05 - 930.74 x + 1.3045 x^2)
    f64 x1 = ;
    f64 x2 = ;
    f64 x4 = ;
    f64 n0 = +40973028771.259094;
    f64 n1 = -161630532.82151473;
    f64 d0 = +163516.09730925123;
    f64 d1 = -930.7444896485703;
    f64 d2 = +1.3044812440250384;
    f64 Num = n0 + n1*x1 + x4;
    f64 Den = d0 + d1*x1 + d2*x2;
    return Num / Den;

nox_u: {'noabserr': True, 'spec': [(1, 2, 3), (0, 1, 2)]}
(1, 2, 3) / (0, 1, 2) ................................. 1.307%
nox_u -> (133.3 x - 47.122 y + x^2) / (-0.05689 + 0.0012335 x + 0.00017801 y)
    f64 x1 = ;
    f64 y1 = ;
    f64 x2 = ;
    f64 n1 = +133.30350610563823;
    f64 n2 = -47.121524818758765;
    f64 d0 = -0.05688971632749395;
    f64 d1 = +0.0012334671429795158;
    f64 d2 = +0.00017800897800483042;
    f64 Num = n1*x1 + n2*y1 + x2;
    f64 Den = d0 + d1*x1 + d2*y1;
    return Num / Den;

nox_Z: {'spec': [(0, 2, 4, 5), (0,)]}
(0, 2, 4, 5) / (0,) ................................... 1.801%
nox_Z -> (4.1802e+05 - 4210 y + 9.9798 xy + y^2) / (4.196e+05)
    f64 y1 = ;
    f64 x1y1 = ;
    f64 y2 = ;
    f64 c0 = +0.9962244155069389;
    f64 c2 = -0.010033321729222494;
    f64 c4 = +2.3783799563418678e-05;
    f64 c5 = +2.383200854104837e-06;
    return c0 + c2*y1 + c4*x1y1 + c5*y2;

cea_Tcomb_high: {'spec': [(0, 1, 2, 4, 5), (1, 4, 5, 7, 8)]}
(0, 1, 2, 4, 5) / (1, 4, 5, 7, 8) ..................... 2.699%
cea_Tcomb_high -> (-14.287 - 32.406 x + 3.4742 y + 18.31 xy + y^2) / (0.0083002 x + 0.0013854 xy + 0.00041962 y^2 - 4.2153e-06 x^2 y + 0.00023245 xy^2)
    f64 x1 = ;
    f64 y1 = ;
    f64 x1y1 = ;
    f64 y2 = ;
    f64 x2y1 = ;
    f64 x1y2 = ;
    f64 n0 = -14.286699845712628;
    f64 n1 = -32.4062851919402;
    f64 n2 = +3.4742076078850297;
    f64 n4 = +18.309776531751787;
    f64 d1 = +0.008300237024458222;
    f64 d4 = +0.001385352578317833;
    f64 d5 = +0.00041962485459847375;
    f64 d7 = -4.215277067475584e-06;
    f64 d8 = +0.00023245123290958759;
    f64 Num = n0 + n1*x1 + n2*y1 + n4*x1y1 + y2;
    f64 Den = d1*x1 + d4*x1y1 + d5*y2 + d7*x2y1 + d8*x1y2;
    return Num / Den;

cea_Tcomb_low: {'spec': [(0, 1, 2), (0, 1, 2, 4)]}
(0, 1, 2) / (0, 1, 2, 4) .............................. 3.831%
cea_Tcomb_low -> (4.6259 + 4.241 x + y) / (0.0062646 + 0.0038335 x - 0.00065063 y - 0.00050479 xy)
    f64 x1 = ;
    f64 y1 = ;
    f64 x1y1 = ;
    f64 n0 = +4.625880807530353;
    f64 n1 = +4.240967496861169;
    f64 d0 = +0.0062645868710667785;
    f64 d1 = +0.003833484131106042;
    f64 d2 = -0.0006506284584737187;
    f64 d4 = -0.0005047873316636391;
    f64 Num = n0 + n1*x1 + y1;
    f64 Den = d0 + d1*x1 + d2*y1 + d4*x1y1;
    return Num / Den;

cea_Cp_high: {'spec': [(1, 4, 6, 7, 8, 9), (0, 1, 2, 4, 8, 9)]}
(1, 4, 6, 7, 8, 9) / (0, 1, 2, 4, 8, 9) ............... 5.2%
cea_Cp_high -> (3582.9 x - 907.66 xy + 0.86387 x^3 - 3.093 x^2 y + 70.736 xy^2 + y^3) / (0.061154 + 1.9517 x - 0.012086 y - 0.47145 xy + 0.031758 xy^2 + 0.00023038 y^3)
    f64 x1 = ;
    f64 y1 = ;
    f64 x1y1 = ;
    f64 x3 = ;
    f64 x2y1 = ;
    f64 x1y2 = ;
    f64 y3 = ;
    f64 n1 = +3582.944641587491;
    f64 n4 = -907.659602547744;
    f64 n6 = +0.8638673972095523;
    f64 n7 = -3.09295057735193;
    f64 n8 = +70.73570153378617;
    f64 d0 = +0.06115388267778262;
    f64 d1 = +1.9517349532470427;
    f64 d2 = -0.012085905000346658;
    f64 d4 = -0.47145131429742476;
    f64 d8 = +0.03175824988676035;
    f64 d9 = +0.0002303751013953221;
    f64 Num = n1*x1 + n4*x1y1 + n6*x3 + n7*x2y1 + n8*x1y2 + y3;
    f64 Den = d0 + d1*x1 + d2*y1 + d4*x1y1 + d8*x1y2 + d9*y3;
    return Num / Den;

cea_Cp_low_left: {'spec': [(0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 7, 8, 9)]}
(0, 1, 2, 3, 4, 5) / (0, 1, 2, 4, 5, 7, 8, 9) ......... 6.587%
cea_Cp_low_left -> (2.8701 + 1.4321 x - 2.9517 y - 0.24204 x^2 - 0.35194 xy + y^2) / (0.00035573 + 0.00018934 x - 0.00050345 y - 7.7265e-05 xy + 0.00023584 y^2 - 2.1318e-05 x^2 y + 1.852e-05 xy^2 + 2.3367e-05 y^3)
    f64 x1 = ;
    f64 y1 = ;
    f64 x2 = ;
    f64 x1y1 = ;
    f64 y2 = ;
    f64 x2y1 = ;
    f64 x1y2 = ;
    f64 y3 = ;
    f64 n0 = +2.8700843792143695;
    f64 n1 = +1.4321259855091084;
    f64 n2 = -2.951652427295875;
    f64 n3 = -0.2420415412480568;
    f64 n4 = -0.35193895657900276;
    f64 d0 = +0.0003557287035334117;
    f64 d1 = +0.00018934292736992974;
    f64 d2 = -0.0005034451842726294;
    f64 d4 = -7.726545211497027e-05;
    f64 d5 = +0.00023583511725994315;
    f64 d7 = -2.1317881064428632e-05;
    f64 d8 = +1.85195099174797e-05;
    f64 d9 = +2.336730752102996e-05;
    f64 Num = n0 + n1*x1 + n2*y1 + n3*x2 + n4*x1y1 + y2;
    f64 Den = d0 + d1*x1 + d2*y1 + d4*x1y1 + d5*y2 + d7*x2y1 + d8*x1y2 + d9*y3;
    return Num / Den;

cea_Cp_low_right: {'spec': [(0, 1, 2, 5), (0, 1, 2, 4, 5, 8)]}
(0, 1, 2, 5) / (0, 1, 2, 4, 5, 8) ..................... 3.88%
cea_Cp_low_right -> (4.2959 + 0.28908 x - 3.3675 y + y^2) / (0.00059786 + 7.1828e-05 x - 0.00074156 y - 4.2894e-05 xy + 0.00037331 y^2 + 1.8653e-05 xy^2)
    f64 x1 = ;
    f64 y1 = ;
    f64 x1y1 = ;
    f64 y2 = ;
    f64 x1y2 = ;
    f64 n0 = +4.295929211019145;
    f64 n1 = +0.28907793370196055;
    f64 n2 = -3.367483987949862;
    f64 d0 = +0.0005978622331052173;
    f64 d1 = +7.182771715581137e-05;
    f64 d2 = -0.0007415570025024972;
    f64 d4 = -4.289368017656845e-05;
    f64 d5 = +0.00037331217091959564;
    f64 d8 = +1.8653015095784888e-05;
    f64 Num = n0 + n1*x1 + n2*y1 + y2;
    f64 Den = d0 + d1*x1 + d2*y1 + d4*x1y1 + d5*y2 + d8*x1y2;
    return Num / Den;

cea_Mw: {'spec': [(0, 1, 3, 4, 5), (1, 2, 3, 5)]}
(0, 1, 3, 4, 5) / (1, 2, 3, 5) ........................ 3.314%
cea_Mw -> (0.21024 + 0.74869 x + 0.0037435 x^2 + 0.13463 xy + y^2) / (61.586 x + 68.421 y + 0.031774 x^2 + 30.142 y^2)
    f64 x1 = ;
    f64 y1 = ;
    f64 x2 = ;
    f64 x1y1 = ;
    f64 y2 = ;
    f64 n0 = +0.21024398321396112;
    f64 n1 = +0.7486891502942208;
    f64 n3 = +0.0037434970654549094;
    f64 n4 = +0.1346327815444957;
    f64 d1 = +61.5856277715907;
    f64 d2 = +68.42132631506553;
    f64 d3 = +0.03177433731792165;
    f64 d5 = +30.142213791419902;
    f64 Num = n0 + n1*x1 + n3*x2 + n4*x1y1 + y2;
    f64 Den = d1*x1 + d2*y1 + d3*x2 + d5*y2;
    return Num / Den;

    """


if __name__ == "__main__":
    main()

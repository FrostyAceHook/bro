"""
Helpers to create function approximations (for static use, not dynamically).
Creates a c snippet which takes some number of floating-point arguments and
calculates a single floating-point return.
"""

import contextlib
import functools
import itertools
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
import CEA_Wrap

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class FileLock:
    def __init__(self, filename):
        self.filename = filename
        self.handle = None

    def __enter__(self):
        self.handle = open(self.filename, "a+")
        try:
            if os.name == "nt":
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_LOCK, 0x7fff_ffff)
            else:
                fcntl.flock(self.handle, fcntl.LOCK_EX)
        except:
            self.handle.close()
            raise
        return self.handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if os.name == 'nt':
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 0x7fff_ffff)
            else:
                fcntl.flock(self.handle, fcntl.LOCK_UN)
        finally:
            self.handle.close()


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

    @staticmethod
    def _error(real, span, approx, maximum=True, noabserr=False):
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
        N = max(1, 6 // dims)
        ones = yup_all_ones(N, *flatcoords)

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
            if maxdegree > N:
                N = maxdegree
                ones = yup_all_ones(N, *flatcoords)

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

    def code(self):
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



class LUT:
    def __init__(self, *spaces):
        self.ndim = len(spaces)
        assert all(len(x) == 3 for x in spaces)
        self.N = [x[2] for x in spaces]
        self.axes = [np.linspace(*space) for space in spaces]
        self.data = np.empty(self.N, dtype=float)

    def match(self, f):
        for ijk, point in zip(np.ndindex(*self.N), itertools.product(*self.axes)):
            self.data[ijk] = f(*point)
        self._interp = scipy.interpolate.RegularGridInterpolator(self.axes, self.data)

    def __call__(self, *points):
        points = np.broadcast_arrays(*points)
        points = np.vstack(points).T
        return self._interp(points)



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
    print(ratpoly.code())
    print()

def peek1d(func, X):
    Y = func(X)
    fig = plt.figure(figsize=(8, 5))
    fig.suptitle(func.__name__)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X, Y)
    plt.tight_layout()

def peek2d(func, X, Y, maskf=None):
    Z = func(X.ravel(), Y.ravel()).reshape(X.shape)
    if maskf is not None:
        Z[~maskf(X, Y)] = np.nan
    fig = plt.figure(figsize=(8, 5))
    fig.suptitle(func.__name__)
    ax = fig.add_subplot(1, 1, 1)
    cont = ax.contourf(X, Y, Z, levels=100, cmap="viridis")
    fig.colorbar(cont, ax=ax)
    plt.tight_layout()

def peek3d(*args):
    fig = plt.figure(figsize=(8, 5))
    for i, (func, X, Y) in enumerate(args):
        Z = func(X.ravel(), Y.ravel()).reshape(X.shape)
        Z[Z < 0] = 0.0
        ax = fig.add_subplot(len(args), 1, 1 + i)
        cont = ax.contourf(X, Y, Z, levels=100, cmap="viridis")
        ax.set_title(func.__name__)
        fig.colorbar(cont, ax=ax)
    plt.tight_layout()

def compare2d(X, Y, realf, approxf, trim_corners=False):
    real = realf(X, Y)
    span = np.max(real) - np.min(real)
    approx = approxf(X, Y)
    error = 100 * RationalPolynomial._error(real, span, approx, maximum=False)
    if trim_corners:
        for idx in [(0,0), (0,-1), (-1,0), (-1,-1)]:
            real[idx] = np.nan
            approx[idx] = np.nan
            error[idx] = np.nan
    fig = plt.figure(figsize=(8, 5))
    for i, data in enumerate([real, approx, error]):
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
    # peek2d(nox_P, *nox_in_Trho())
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
    # peek2d(nox_cp, *nox_in_TP(N=300, trim_corner=True))
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
    # peek2d(nox_cv, *nox_in_TP(N=300))
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
    # peek2d(nox_h, *nox_in_Trho())
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
    # peek2d(nox_u, *nox_in_Trho())
    # do(nox_u, *nox_in_Trho(), noabserr=True, blitz=1.5)
    do(nox_u, *nox_in_Trho(), noabserr=True, spec=[(1, 2, 3), (0, 1, 2)])

    def nox_Z(T, rho):
        return PropsSI("Z", "T", T, "D", rho, "N2O")
    # peek2d(nox_Z, *nox_in_Trho())
    # do(nox_Z, *nox_in_Trho())
    do(nox_Z, *nox_in_Trho(), spec=[(0, 2, 4, 5), (0,)])






class CEA_Result:
    """
    Example .inp nasacea input.
    problem rocket equilibrium
        p(bar) = 30.00000
        o/f = 6.00000
        sup = 4.00000
    react
        fuel=PARAFFIN wt=100.00000  t,k=298.15000 C 32 H 66 h,kc/mol=-224.20000
        ox=N2O wt=100.00000  t,k=298.15000 N 2 O 1 h,kc/mol=15.50000
    output trans
        plot p t isp ivac m mw cp gam o/f cf rho son mach phi h cond pran ae pip
    end
    """

    # All CEA_Wrap rocket result properties (note some units are changed by us to be base si):
    NAMES = [
        # "prod_c", # Chamber products (not saved)
        # "prod_t", # Throat products (not saved)
        # "prod_e", # Exit products (not saved)
        "p", # Pressure, Pa
        "t_p", # Throat pressure
        "c_p", # Chamber pressure
        "t", # Temperature, Kelvin
        "t_t", # Throat temperature
        "c_t", # Chamber temperature
        "h", # Enthalpy, J/kg
        "t_h", # Throat enthalpy
        "c_h", # Chamber enthalpy
        "rho", # Density, kg/m^3
        "t_rho", # Throat density
        "c_rho", # Chamber density
        "son", # Sonic velocity, m/s
        "t_son", # Throat sonic velocity
        "c_son", # Chamber sonic velocity
        "visc", # Burned gas viscosity, Pa*s
        "t_visc", # Throat viscosity
        "c_visc", # Chamber viscosity
        "cond", # Burned gas thermal conductivity, W/(m*K)
        "t_cond", # Throat thermal conductivity
        "c_cond", # Chamber thermal conductivity
        "pran", # Burned gas Prandtl number
        "t_pran", # Throat Prandtl number
        "c_pran", # Chamber Prandtl number
        "mw", # Molecular weight of all products, kg/mol
        "t_mw", # Throat molecular weight
        "c_mw", # Chamber molecular weight
        "cp", # Constant-pressure specific heat capacity, J/(kg*K)
        "t_cp", # Throat cp
        "c_cp", # Chamber cp
        "gammas", # isentropic exponent (name from nasacea paper p1)
                  # isentropic ratio of specific heats (name from cea_wrap)
        "t_gammas", # Throat gammas
        "c_gammas", # Chamber gammas
        "gamma", # Real ratio of specific heats
        "t_gamma", # Throat gamma
        "c_gamma", # Chamber gamma
        "isp", # Ideal ISP (ambient pressure = exit pressure), s
        "t_isp", # Throat ISP
        "ivac", # Vacuum ISP, s
        "t_ivac", # Throat vacuum ISP
        "cf", # Ideally expanded thrust coefficient
        "t_cf", # Throat CF
        "dLV_dLP_t", # (dLV/dLP)t
        "t_dLV_dLP_t", # Throat dLV/dLP
        "c_dLV_dLP_t", # Chamber dLV/dLP
        "dLV_dLT_p", # (dLV/dLT)p
        "t_dLV_dLT_p", # Throat dLV/dLT
        "c_dLV_dLT_p", # Chamber dLV/dLT
        "cstar", # Characteristic velocity in chamber, m/s
        "mach", # Mach number at exhaust
    ]
    def __init__(self, data):
        self.data = data
    @classmethod
    def from_cea(cls, cea):
        data = np.empty(len(cls.NAMES), dtype=np.float32)
        for name, i in cls.MAPPING.items():
            # fix some stupid non-si values.
            data[i] = getattr(cea, name)
            if name in {"p", "t_p", "c_p"}:
                data[i] *= 1e5 # bar -> Pa
            elif name in {"h", "t_h", "c_h"}:
                data[i] *= 1e3 # kJ/kg -> J/kg
            elif name in {"mw", "t_mw", "c_mw"}:
                data[i] *= 1e-3 # kg/kmol -> kg/mol
            elif name in {"cp", "t_cp", "c_cp"}:
                data[i] *= 1e3 # kJ/(kg*K) -> J/(kg*K)
        return cls(data)
    def __getattr__(self, name):
        if name not in type(self).NAMES:
            return super().__getattribute__(name)
        return self.data[type(self).MAPPING[name]]
CEA_Result.MAPPING = {n: i for i, n in enumerate(CEA_Result.NAMES)}
CEA_Result.NAMES = set(CEA_Result.NAMES)




class CEA:
    CACHE_PATH = os.path.join(os.path.dirname(__file__), "approximator_cea_cache.npz")
    LOCK_PATH = os.path.join(os.path.dirname(__file__), "approximator_cea_cache.lock")

    def __init__(self):
        # Compositions and enthalpy of formation of paraffin and N2O from Hybrid
        # Rocket Propulsion Handbook, Karp & Jens.
        fuel_comp = CEA_Wrap.ChemicalRepresentation(
                " C 32 H 66", # need leading space to fix CEA_Wrap bug lmao.
                hf=-224.2, hf_unit="kc")
        fuel = CEA_Wrap.Fuel("PARAFFIN", temp=298.15, chemical_representation=fuel_comp)
        ox_comp = CEA_Wrap.ChemicalRepresentation(" N 2 O 1",
                hf=15.5, hf_unit="kc")
        ox = CEA_Wrap.Oxidizer("N2O", temp=298.15, chemical_representation=ox_comp)
        # Dummy initial params.
        self.problem = CEA_Wrap.RocketProblem(pressure=20, pressure_units="bar",
                materials=[fuel, ox], o_f=6, ae_at=8,
            )
        self.cache = {}

    def keyof(self, P, ofr, eps):
        P = round(float(P), 4)
        ofr = round(float(ofr), 3)
        eps = round(float(eps), 3)
        return P, ofr, eps

    def __call__(self, P, ofr, eps): # expects P in MPa
        key = self.keyof(P, ofr, eps)
        if key not in self.cache:
            P, ofr, eps = key
            self.problem.set_pressure(P * 10) # mpa -> bar
            self.problem.set_o_f(ofr)
            self.problem.set_ae_at(eps)
            cea = self.problem.run()
            self.cache[key] = CEA_Result.from_cea(cea)
        return self.cache[key]

    def __getitem__(self, name):
        def f(P, ofr, eps):
            return getattr(self(P, ofr, eps), name)
        f.__name__ = f"cea.{name}"
        return np.vectorize(f)

    def load(self, lock=True):
        filelock = FileLock(self.LOCK_PATH) if lock else contextlib.nullcontext()
        with filelock:
            if os.path.exists(self.CACHE_PATH):
                data = np.load(self.CACHE_PATH)
                keys = data["keys"]
                values = data["values"]
                cache = {self.keyof(*k): CEA_Result(v) for k, v in zip(keys, values)}
                self.cache = cache | self.cache
    def save(self):
        with FileLock(self.LOCK_PATH):
            print("saving cea cache...")
            self.load(lock=False)
            keys = np.array(list(self.cache.keys()), dtype=np.float32)
            values = np.array(list(x.data for x in self.cache.values()), dtype=np.float32)
            np.savez(self.CACHE_PATH, keys=keys, values=values, allow_pickle=False)
            print("saved cea cache.")
            print()

    def __enter__(self):
        self.load()
    def __exit__(self, etype, evalue, etb):
        self.save()
        return False

    # chamber pressure always lower than tank, but just use same bounds. also just assume no
    # comb at pressure below 90kPa.
    P_low = 0.09
    P_high = 7.2
    def in_P(self, N=120, conc=None, strength=0.0):
        return concspace(self.P_low, conc, self.P_high, strength=strength, N=N)
    # ox-fuel ratio reasonably below 13, and below 0.5 can safely assume no comb.
    ofr_low = 0.5
    ofr_high = 13.0
    def in_ofr(self, N=120, conc=None, strength=0.0):
        return concspace(self.ofr_low, conc, self.ofr_high, strength=strength, N=N)
    # nozzle exit area/throat area reasonably low for us since we have low altitude and low chamber pressure.
    eps_low = 1.2
    eps_high = 12.0
    def in_eps(self, N=120, conc=None, strength=0.0):
        return concspace(self.eps_low, conc, self.eps_high, strength=strength, N=N)

    def in_Peps(self, N=70, Pconc=None, Pstrength=0.0, epsconc=None, epsstrength=0.0):
        X = self.in_P(N, Pconc, Pstrength)
        Y = self.in_eps(N, epsconc, epsstrength)
        return np.meshgrid(X, Y)
    def in_Pofr(self, N=70, Pconc=None, Pstrength=0.0, ofrconc=None, ofrstrength=0.0):
        X = self.in_P(N, Pconc, Pstrength)
        Y = self.in_ofr(N, ofrconc, ofrstrength)
        return np.meshgrid(X, Y)
    def in_ofreps(self, N=70, ofrconc=None, ofrstrength=0.0, epsconc=None, epsstrength=0.0):
        X = self.in_ofr(N, ofrconc, ofrstrength)
        Y = self.in_eps(N, epsconc, epsstrength)
        return np.meshgrid(X, Y)

    def in_Pofreps(self, N=30):
        return np.meshgrid(self.in_P(N), self.in_ofr(N), self.in_eps(N))

CEA = CEA()




def cea():

    # cc temperature is independant of epsilon. Also its a huge pain to approx,
    # so we split into a high-ofr approx and a low-ofr approx which combine to
    # cover the whole input space. This should have negligible performance
    # impacts because the motor generally stays in high ofr at the beginning then
    # is in low ofr for the rest (so very predictable branch).
    def cea_T_cc_in_high(N=70):
        X = CEA.in_P(N=N)
        Y = concspace(4, 4, CEA.ofr_high, strength=1.0, N=N)
        return np.meshgrid(X, Y)
    def cea_T_cc_in_low(N=70):
        X = CEA.in_P(N=N, conc=0.1, strength=1.0)
        Y = concspace(CEA.ofr_low, CEA.ofr_low, 4, strength=1.0, N=N)
        return np.meshgrid(X, Y)
    def cea_T_cc_high(P, ofr):
        return CEA["c_t"](P, ofr, 2)
    def cea_T_cc_low(P, ofr): # just get a different __name__
        return CEA["c_t"](P, ofr, 2)
    # peek2d(cea_T_cc_high, *CEA.in_Pofr())

    # do(cea_T_cc_high, *cea_T_cc_in_high(), blitz=4.0, starting_cost=27)
    do(cea_T_cc_high, *cea_T_cc_in_high(), spec=[(0, 1, 2, 4, 5), (1, 4, 5, 7, 8)])

    # do(cea_T_cc_low, *cea_T_cc_in_low(), blitz=4.0, starting_cost=27)
    do(cea_T_cc_low, *cea_T_cc_in_low(), spec=[(0, 1, 2), (0, 1, 2, 4)])


    # cc cp is also indep of eps. Also after having a peek at cp, id say "looks fucking grim mate".
    # lets split over ofr again. HOLY its a doozy. split low again into left and right.
    def cea_cp_cc_in_high(N=50):
        X = CEA.in_P(N=N)
        Y = concspace(4, 4, CEA.ofr_high, strength=2.0, N=N)
        return np.meshgrid(X, Y)
    def cea_cp_cc_in_low_left(N=40):
        X = concspace(CEA.P_low, 1, 1, strength=2.0, N=N)
        Y = concspace(CEA.ofr_low, 4, 4, strength=2.0, N=N)
        X, Y = np.meshgrid(X, Y)
        def mask(X, Y, training=False):
            return np.ones(X.shape, dtype=bool)
            if training:
                return np.ones(X.shape, dtype=bool)
            return (X > 0.15)
        return X, Y, mask
    def cea_cp_cc_in_low_right(N=40):
        X = concspace(1, CEA.P_high, CEA.P_high, strength=1.0, N=N)
        Y = concspace(CEA.ofr_low, 4, 4, strength=2.0, N=N)
        X, Y = np.meshgrid(X, Y)
        def mask(X, Y, training=False):
            return np.ones(X.shape, dtype=bool)
        return X, Y, mask
    def cea_cp_cc_high(P, ofr):
        return CEA["c_cp"](P, ofr, 2)
    def cea_cp_cc_low_left(P, ofr):
        return CEA["c_cp"](P, ofr, 2)
    def cea_cp_cc_low_right(P, ofr):
        return CEA["c_cp"](P, ofr, 2)
    # peek2d(cea_cp_cc_high, *CEA.in_Pofr())

    # do(cea_cp_cc_high, *cea_cp_cc_in_high(), blitz=4.0, starting_cost=30)
    do(cea_cp_cc_high, *cea_cp_cc_in_high(N=70), spec=[(1, 4, 6, 7, 8, 9), (0, 1, 2, 4, 8, 9)])

    # do(cea_cp_cc_low_left, *cea_cp_cc_in_low_left(), blitz=4.0, starting_cost=26)
    do(cea_cp_cc_low_left, *cea_cp_cc_in_low_left(N=70), spec=[(0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 7, 8, 9)])

    # do(cea_cp_cc_low_right, *cea_cp_cc_in_low_right(), blitz=3.0, starting_cost=24)
    do(cea_cp_cc_low_right, *cea_cp_cc_in_low_right(N=70), spec=[(0, 1, 2, 5), (0, 1, 2, 4, 5, 8)])


    # cc Mw indep of eps.
    def cea_Mw(P, ofr):
        return CEA["c_mw"](P, ofr, 2)
    # peek(cea_Mw, *CEA.in_Pofr())
    # do(cea_Mw, *CEA.in_Pofr(N=40), blitz=1.0)
    do(cea_Mw, *CEA.in_Pofr(N=70), spec=[(0, 1, 3, 4, 5), (1, 2, 3, 5)])





    def map_Peps(f, ofr=6, vectorise=None):
        vf = lambda P, eps: f(P, ofr, eps)
        if vectorise is True or vectorise is None and isinstance(f, np.vectorize):
            vf = np.vectorize(vf)
        vf.__name__ = f.__name__ + f"  | x=P, y=eps, ofr={ofr}"
        return vf
    def map_Pofr(f, eps=3, vectorise=None):
        vf = lambda P, ofr: f(P, ofr, eps)
        if vectorise is True or vectorise is None and isinstance(f, np.vectorize):
            vf = np.vectorize(vf)
        vf.__name__ = f.__name__ + f"  | x=P, y=ofr, eps={eps}"
        return vf
    def map_ofreps(f, P=2, vectorise=None):
        vf = lambda ofr, eps: f(P, ofr, eps)
        if vectorise is True or vectorise is None and isinstance(f, np.vectorize):
            vf = np.vectorize(vf)
        vf.__name__ = f.__name__ + f"  | x=ofr, y=eps, P={P}"
        return vf



    # "random" values for testing.
    mdot = 0.4
    A_throat = 0.025**2 * np.pi / 4
    P_a = 1e5
    # P_a = 0.0
    g0 = 9.80665

    def thurst_ideal(P, ofr, eps):
        cea = CEA(P, ofr, eps)
        return mdot * cea.cstar * cea.cf
    def thurst_no_vel_change(P, ofr, eps):
        cea = CEA(P, ofr, eps)
        Ve = cea.isp * g0
        # Also works:
        # Ve = cea.mach * cea.son
        # Ve = np.sqrt(2 * (cea.c_h - cea.h))
        return mdot * Ve + (cea.p - P_a) * A_throat*eps
    def thurst_isp_vaccuum(P, ofr, eps):
        cea = CEA(P, ofr, eps)
        Ve_vac = cea.ivac * g0
        return mdot * Ve_vac - P_a * A_throat*eps
    def thurst_rocketcea_formula(P, ofr, eps):
        cea = CEA(P, ofr, eps)
        cf = cea.isp * g0 / cea.cstar - eps * P_a / (P*1e6)
        return mdot * cea.cstar * cf


    if 0:
        for name, func in locals().items():
            if not name.startswith("thurst"):
                continue
            peek3d(
                [map_Peps(func), *CEA.in_Peps()],
                [map_Pofr(func), *CEA.in_Pofr()],
                [map_ofreps(func), *CEA.in_ofreps()],
            )
        return


    if 1:
        class Bias:
            # Biases a linearly-spaced interval to be not that, so that a transformation can
            # be applied to inputs of a lut but the lut lookup itself can still be using a
            # regularly spaced rectangular grid for fast lookup (allowing the fit of the lut
            # to be better on average, since data may change more densely in certain regions).
            # https://www.desmos.com/calculator/okiztovx6y
            @classmethod
            def P_from(cls, i):
                A = CEA.P_low
                B = CEA.P_high
                a0 = -1.4
                a1 = 0.5 * (A + B - np.sqrt((A + B)**2 - 4*(A*B + B*a0 - 4*a0)))
                a2 = a0 / (a1 - A)
                return a0 / (i - a2) + a1
                return i
            @classmethod
            def P_to(cls, P):
                A = CEA.P_low
                B = CEA.P_high
                a0 = -1.4
                a1 = 0.5 * (A + B - np.sqrt((A + B)**2 - 4*(A*B + B*a0 - 4*a0)))
                a2 = a0 / (a1 - A)
                return a0 / (P - a1) + a2
                return P
            @classmethod
            def ofr_from(cls, j):
                return j
                A = CEA.ofr_low
                B = CEA.ofr_high
                a0 = -15.0
                a1 = 0.5 * (A + B - np.sqrt((A + B)**2 - 4*(A*B + B*a0 - 4*a0)))
                a2 = a0 / (a1 - A)
                return a0 / (j - a2) + a1
            @classmethod
            def ofr_to(cls, ofr):
                return ofr
                A = CEA.ofr_low
                B = CEA.ofr_high
                a0 = -15.0
                a1 = 0.5 * (A + B - np.sqrt((A + B)**2 - 4*(A*B + B*a0 - 4*a0)))
                a2 = a0 / (a1 - A)
                return a0 / (ofr - a1) + a2
            @classmethod
            def eps_from(cls, k):
                A = CEA.eps_low
                B = CEA.eps_high
                a0 = -3.3
                a1 = 0.5 * (A + B - np.sqrt((A + B)**2 - 4*(A*B + B*a0 - 4*a0)))
                a2 = a0 / (a1 - A)
                return a0 / (k - a2) + a1
                return k
            @classmethod
            def eps_to(cls, eps):
                A = CEA.eps_low
                B = CEA.eps_high
                a0 = -3.3
                a1 = 0.5 * (A + B - np.sqrt((A + B)**2 - 4*(A*B + B*a0 - 4*a0)))
                a2 = a0 / (a1 - A)
                return a0 / (eps - a1) + a2
                return eps

            @classmethod
            def in_P(cls, N=120):
                return cls.P_to(CEA.in_P(N=N))
            @classmethod
            def in_ofr(cls, N=120):
                return cls.ofr_to(CEA.in_ofr(N=N))
            @classmethod
            def in_ofr(cls, N=120):
                return cls.eps_to(CEA.in_eps(N=N))

            @classmethod
            def in_Peps(cls, N=70):
                P, eps = CEA.in_Peps(N=N)
                return cls.P_to(P), cls.eps_to(eps)
            @classmethod
            def in_Pofr(cls, N=70):
                P, ofr = CEA.in_Pofr(N=N)
                return cls.P_to(P), cls.ofr_to(ofr)
            @classmethod
            def in_ofreps(cls, N=70):
                ofr, eps = CEA.in_ofreps(N=N)
                return cls.ofr_to(ofr), cls.eps_to(eps)

            @classmethod
            def in_Pofreps(cls, N=30):
                P, ofr, eps = CEA.in_Pofreps(N=N)
                return cls.P_to(P), cls.ofr_to(ofr), cls.eps_to(eps)

        lut = LUT(
                (Bias.P_to(CEA.P_low), Bias.P_to(CEA.P_high), 2),
                (Bias.ofr_to(CEA.ofr_low), Bias.ofr_to(CEA.ofr_high), 10),
                (Bias.eps_to(CEA.eps_low), Bias.eps_to(CEA.eps_high), 2),
            )
        def unbiased_lut(P, ofr, eps):
            i, j, k = Bias.P_to(P), Bias.ofr_to(ofr), Bias.eps_to(eps)
            return lut(i, j, k)

        def biased_ivac(i, j, k):
            return CEA["ivac"](Bias.P_from(i), Bias.ofr_from(j), Bias.eps_from(k))
        lut.match(biased_ivac)

        def error(P, ofr, eps):
            P = P.ravel()
            ofr = ofr.ravel()
            eps = eps.ravel()
            approx = unbiased_lut(P, ofr, eps)
            real = CEA["ivac"](P, ofr, eps)
            return 100.0 * np.abs(approx / real - 1)
        print("Ivac using LUT")
        print(f"{np.prod(lut.N)} elements [{','.join(map(str, lut.N))}]")
        print(f"{np.max(error(*CEA.in_Pofreps(N=50))):.3g} %error")

        with np.printoptions(precision=np.finfo(float).precision, suppress=False):
            if lut.N[0] == 3:
                print("P (unbiased):")
                print(Bias.P_from(lut.axes[0]))
            elif lut.N[0] > 3:
                print("P (biased):")
                print(lut.axes[0])
            if lut.N[1] == 3:
                print("ofr (unbiased):")
                print(Bias.ofr_from(lut.axes[1]))
            elif lut.N[1] > 3:
                print("ofr (biased):")
                print(lut.axes[1])
            if lut.N[2] == 3:
                print("eps (unbiased):")
                print(Bias.eps_from(lut.axes[2]))
            elif lut.N[2] > 3:
                print("eps (biased):")
                print(lut.axes[2])
            print("data:")
            print(lut.data)
            print()

        if 0:
            maxerr = 0.0
            maxP = None
            for P in CEA.in_P(N=50):
                ofr, eps = CEA.in_ofreps(N=50)
                err = np.max(error(P, ofr, eps))
                if err > maxerr:
                    maxerr = err
                    maxP = P
            maxerr = 0.0
            maxofr = None
            for ofr in CEA.in_ofr(N=50):
                P, eps = CEA.in_Peps(N=50)
                err = np.max(error(P, ofr, eps))
                if err > maxerr:
                    maxerr = err
                    maxofr = ofr
            maxerr = 0.0
            maxeps = None
            for eps in CEA.in_eps(N=50):
                P, ofr = CEA.in_Pofr(N=50)
                err = np.max(error(P, ofr, eps))
                if err > maxerr:
                    maxerr = err
                    maxeps = eps
            def error_Pofr(P, ofr, eps=maxeps):
                return error(P, ofr, eps)
            def error_Peps(P, eps, ofr=maxofr):
                return error(P, ofr, eps)
            def error_ofreps(ofr, eps, P=maxP):
                return error(P, ofr, eps)
            peek3d(
                [map_Pofr(CEA["ivac"], eps=maxeps), *CEA.in_Pofr(N=50)],
                [map_Pofr(unbiased_lut, eps=maxeps, vectorise=False), *CEA.in_Pofr(N=50)],
                [error_Pofr, *CEA.in_Pofr(N=50)],
            )
            peek3d(
                [map_Peps(CEA["ivac"], ofr=maxofr), *CEA.in_Peps(N=50)],
                [map_Peps(unbiased_lut, ofr=maxofr, vectorise=False), *CEA.in_Peps(N=50)],
                [error_Peps, *CEA.in_Peps(N=50)],
            )
            peek3d(
                [map_ofreps(CEA["ivac"], P=maxP), *CEA.in_ofreps(N=50)],
                [map_ofreps(unbiased_lut, P=maxP, vectorise=False), *CEA.in_ofreps(N=50)],
                [error_ofreps, *CEA.in_ofreps(N=50)],
            )


            for ofr in CEA.in_ofr(N=50):
                if ofr < 2.0 or ofr > 9.5:
                    continue
                P, eps = CEA.in_Peps(N=50)
                peek3d(
                    [map_Peps(CEA["ivac"], ofr=ofr), *CEA.in_Peps(N=50)],
                    [map_Peps(unbiased_lut, ofr=ofr, vectorise=False), *CEA.in_Peps(N=50)],
                    [lambda P, eps: error_Peps(P, eps, ofr), *CEA.in_Peps(N=50)],
                )


def atmos():
    def atmos_get_Pa(altitude):
        g0 = 9.80665  # m/s2
        R = 8.3144598 # J/mol/K
        M = 0.0289644 # kg/mol

        # Define layers: base altitude (m), base temp (K), base pressure (Pa), lapse rate (K/m)
        layers = [
            (0,       288.15, 101325.0,    -0.0065),   # Troposphere
            (11000,   216.65, 22632.1,      0.0),      # Tropopause
            (20000,   216.65, 5474.89,      0.001),    # Lower Stratosphere
            (32000,   228.65, 868.02,       0.0028),
            (47000,   270.65, 110.91,       0.0),
            (51000,   270.65, 66.94,       -0.0028),
            (71000,   214.65, 3.96,        -0.002),    # Mesosphere
            (84852,   186.87, 0.3734,       0.0)       # Upper Mesosphere
        ]

        for i in range(len(layers) - 1):
            h_base, T_base, P_base, L = layers[i]
            h_next = layers[i + 1][0]
            if altitude < h_next:
                h = altitude
                break
        else:
            # If above last defined layer, extrapolate with last known values
            h_base, T_base, P_base, L = layers[-1]
            h = altitude

        if L == 0:
            # Isothermal layer
            pressure = P_base * math.exp(-g0 * M * (h - h_base) / (R * T_base))
        else:
            # Gradient layer
            T = T_base + L * (h - h_base)
            pressure = P_base * (T / T_base) ** (-g0 * M / (R * L))

        return pressure

    altitude_off = 10_000
    altitude = concspace(altitude_off, altitude_off, altitude_off + 30_000, N=100, strength=5.0)
    def atmos_Pa(x):
        return np.vectorize(atmos_get_Pa)(x - altitude_off)
    # peek1d(Pa, altitude)
    # do(atmos_Pa, altitude)
    do(atmos_Pa, altitude, spec=[(0, 1, 2), (0, 1, 2)])



def main():

    nox()

    with CEA:
        cea()

    atmos()

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

cea_T_cc_high: {'spec': [(0, 1, 2, 4, 5), (1, 4, 5, 7, 8)]}
(0, 1, 2, 4, 5) / (1, 4, 5, 7, 8) ..................... 2.698%
cea_T_cc_high -> (-14.288 - 32.429 x + 3.4748 y + 18.316 xy + y^2) / (0.0082956 x + 0.0013866 xy + 0.00041964 y^2 - 4.2175e-06 x^2 y + 0.00023248 xy^2)
    f64 x1 = ;
    f64 y1 = ;
    f64 x1y1 = ;
    f64 y2 = ;
    f64 x2y1 = ;
    f64 x1y2 = ;
    f64 n0 = -14.288432205175953;
    f64 n1 = -32.42867599262846;
    f64 n2 = +3.474751778815847;
    f64 n4 = +18.31561738023227;
    f64 d1 = +0.008295564068249973;
    f64 d4 = +0.0013866109147024212;
    f64 d5 = +0.0004196407872643308;
    f64 d7 = -4.217470292937784e-06;
    f64 d8 = +0.00023248363130312795;
    f64 Num = n0 + n1*x1 + n2*y1 + n4*x1y1 + y2;
    f64 Den = d1*x1 + d4*x1y1 + d5*y2 + d7*x2y1 + d8*x1y2;
    return Num / Den;

cea_T_cc_low: {'spec': [(0, 1, 2), (0, 1, 2, 4)]}
(0, 1, 2) / (0, 1, 2, 4) .............................. 3.829%
cea_T_cc_low -> (4.6262 + 4.2415 x + y) / (0.0062651 + 0.003834 x - 0.00065075 y - 0.00050486 xy)
    f64 x1 = ;
    f64 y1 = ;
    f64 x1y1 = ;
    f64 n0 = +4.626249969485239;
    f64 n1 = +4.2414906007771505;
    f64 d0 = +0.006265099088905423;
    f64 d1 = +0.0038339663921447076;
    f64 d2 = -0.0006507452327885072;
    f64 d4 = -0.0005048607265830429;
    f64 Num = n0 + n1*x1 + y1;
    f64 Den = d0 + d1*x1 + d2*y1 + d4*x1y1;
    return Num / Den;

cea_cp_cc_high: {'spec': [(1, 4, 6, 7, 8, 9), (0, 1, 2, 4, 8, 9)]}
(1, 4, 6, 7, 8, 9) / (0, 1, 2, 4, 8, 9) ............... 4.683%
cea_cp_cc_high -> (3697.2 x - 936.45 xy + 0.92708 x^3 - 3.2824 x^2 y + 73.081 xy^2 + y^3) / (0.061171 + 2.0171 x - 0.012091 y - 0.48726 xy + 0.032836 xy^2 + 0.00022932 y^3)
    f64 x1 = ;
    f64 y1 = ;
    f64 x1y1 = ;
    f64 x3 = ;
    f64 x2y1 = ;
    f64 x1y2 = ;
    f64 y3 = ;
    f64 n1 = +3697.2499992698126;
    f64 n4 = -936.4542475610451;
    f64 n6 = +0.9270759227273586;
    f64 n7 = -3.2824223104827146;
    f64 n8 = +73.08080264333545;
    f64 d0 = +0.06117125463913653;
    f64 d1 = +2.017096527865246;
    f64 d2 = -0.012091433831058363;
    f64 d4 = -0.4872595016883579;
    f64 d8 = +0.03283582431629867;
    f64 d9 = +0.00022931797763498784;
    f64 Num = n1*x1 + n4*x1y1 + n6*x3 + n7*x2y1 + n8*x1y2 + y3;
    f64 Den = d0 + d1*x1 + d2*y1 + d4*x1y1 + d8*x1y2 + d9*y3;
    return Num / Den;

cea_cp_cc_low_left: {'spec': [(0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 7, 8, 9)]}
(0, 1, 2, 3, 4, 5) / (0, 1, 2, 4, 5, 7, 8, 9) ......... 6.058%
cea_cp_cc_low_left -> (2.8962 + 1.3803 x - 2.9592 y - 0.22743 x^2 - 0.33389 xy + y^2) / (0.00035841 + 0.00018935 x - 0.00050605 y - 8.3745e-05 xy + 0.00023873 y^2 - 1.9483e-05 x^2 y + 2.0741e-05 xy^2 + 2.2767e-05 y^3)
    f64 x1 = ;
    f64 y1 = ;
    f64 x2 = ;
    f64 x1y1 = ;
    f64 y2 = ;
    f64 x2y1 = ;
    f64 x1y2 = ;
    f64 y3 = ;
    f64 n0 = +2.8961711401281796;
    f64 n1 = +1.380267785409765;
    f64 n2 = -2.959235197532152;
    f64 n3 = -0.22742870022761344;
    f64 n4 = -0.33389437020718044;
    f64 d0 = +0.00035840685034168327;
    f64 d1 = +0.0001893546685822136;
    f64 d2 = -0.0005060456120694603;
    f64 d4 = -8.374514776956192e-05;
    f64 d5 = +0.00023872791254781774;
    f64 d7 = -1.948261122047713e-05;
    f64 d8 = +2.07406108782002e-05;
    f64 d9 = +2.276707960740409e-05;
    f64 Num = n0 + n1*x1 + n2*y1 + n3*x2 + n4*x1y1 + y2;
    f64 Den = d0 + d1*x1 + d2*y1 + d4*x1y1 + d5*y2 + d7*x2y1 + d8*x1y2 + d9*y3;
    return Num / Den;

cea_cp_cc_low_right: {'spec': [(0, 1, 2, 5), (0, 1, 2, 4, 5, 8)]}
(0, 1, 2, 5) / (0, 1, 2, 4, 5, 8) ..................... 3.875%
cea_cp_cc_low_right -> (4.2606 + 0.28647 x - 3.3469 y + y^2) / (0.00059783 + 7.1533e-05 x - 0.00074669 y - 4.2802e-05 xy + 0.00037648 y^2 + 1.8535e-05 xy^2)
    f64 x1 = ;
    f64 y1 = ;
    f64 x1y1 = ;
    f64 y2 = ;
    f64 x1y2 = ;
    f64 n0 = +4.2606478633549125;
    f64 n1 = +0.286474125290797;
    f64 n2 = -3.346916554110677;
    f64 d0 = +0.0005978252239739312;
    f64 d1 = +7.15329555395512e-05;
    f64 d2 = -0.0007466904302128527;
    f64 d4 = -4.2802043949574985e-05;
    f64 d5 = +0.0003764818378455317;
    f64 d8 = +1.8535400055226076e-05;
    f64 Num = n0 + n1*x1 + n2*y1 + y2;
    f64 Den = d0 + d1*x1 + d2*y1 + d4*x1y1 + d5*y2 + d8*x1y2;
    return Num / Den;

cea_Mw: {'spec': [(0, 1, 3, 4, 5), (1, 2, 3, 5)]}
(0, 1, 3, 4, 5) / (1, 2, 3, 5) ........................ 3.286%
cea_Mw -> (0.21183 + 0.73935 x + 0.0044777 x^2 + 0.13399 xy + y^2) / (60.847 x + 68.522 y + 0.086728 x^2 + 30.139 y^2)
    f64 x1 = ;
    f64 y1 = ;
    f64 x2 = ;
    f64 x1y1 = ;
    f64 y2 = ;
    f64 n0 = +0.21182839008473092;
    f64 n1 = +0.7393450040280863;
    f64 n3 = +0.004477743899937738;
    f64 n4 = +0.13398511532142252;
    f64 d1 = +60.84686567413254;
    f64 d2 = +68.52163215322909;
    f64 d3 = +0.08672790956340412;
    f64 d5 = +30.13870865656756;
    f64 Num = n0 + n1*x1 + n3*x2 + n4*x1y1 + y2;
    f64 Den = d1*x1 + d2*y1 + d3*x2 + d5*y2;
    return Num / Den;

Ivac using LUT
40 elements [2,10,2]
3.67 %error
ofr (biased):
[ 0.5                1.888888888888889  3.277777777777778
  4.666666666666666  6.055555555555555  7.444444444444445
  8.833333333333332 10.222222222222221 11.61111111111111
 13.               ]
data:
[[[126.1773681640625  174.53619384765625]
  [157.9001007080078  213.9551544189453 ]
  [182.354736328125   235.49440002441406]
  [201.67176818847656 262.7115173339844 ]
  [206.6666717529297  275.6676940917969 ]
  [205.1580047607422  280.71356201171875]
  [202.00814819335938 280.4179382324219 ]
  [198.87869262695312 275.9123229980469 ]
  [196.01426696777344 270.4587097167969 ]
  [193.40469360351562 265.0764465332031 ]]

 [[131.6615753173828  182.00814819335938]
  [164.42405700683594 223.80224609375   ]
  [182.57899475097656 240.7645263671875 ]
  [202.70132446289062 263.05810546875   ]
  [211.09072875976562 277.2680969238281 ]
  [213.28236389160156 284.69927978515625]
  [211.08053588867188 287.3496398925781 ]
  [207.4923553466797  282.7420959472656 ]
  [203.761474609375   275.45361328125   ]
  [200.18348693847656 268.7767639160156 ]]]

atmos_Pa: {'spec': [(0, 1, 2), (0, 1, 2)]}
(0, 1, 2) / (0, 1, 2) ................................. 1.459%
atmos_Pa -> (4.1634e+09 - 1.2755e+05 x + x^2) / (53824 - 3.397 x + 6.7098e-05 x^2)
    f64 x1 = ;
    f64 x2 = ;
    f64 n0 = +4163420288.511369;
    f64 n1 = -127550.45177127342;
    f64 d0 = +53823.7243981808;
    f64 d1 = -3.3969727188689047;
    f64 d2 = +6.709829651497073e-05;
    f64 Num = n0 + n1*x1 + x2;
    f64 Den = d0 + d1*x1 + d2*x2;
    return Num / Den;

    """


if __name__ == "__main__":
    main()

from functools import partial
from itertools import islice, takewhile
from more_itertools import last, nth


def func(x):
    return x**3 - 5 * x + 3


def d_func(x):
    return 3 * x**2 - 5


def newton_old(x0, f, df):
    seq = [x0]
    x = x0
    for _ in range(10):
        x = x - f(x) / df(x)
        seq.append(x)
    return seq


# functional approach
def newton_update(x, f, df):
    return x - f(x)/df(x)


def iterate(x, func):
    while True:
        yield x
        x = func(x)


def newton_seq(x, f, df):
    update = partial(newton_update, f=f, df=df)
    yield from iterate(x, update)


def take10(seq):
    return list(islice(seq, 0, 10))


def newton_same_as_old(x0, f, df):
    return take10(newton_seq(x0, f, df))


def above_tolerance(tol, f):
    return lambda x: abs(f(x)) > tol


def newton_until_converged(x, f, df, tol=1e-5):
    yield from takewhile(
            above_tolerance(tol=tol, f=f),
            newton_seq(x, f, df))


def find_root_newton(x, f, df, tol=1e-5):
    return last(newton_until_converged(x, f, df, tol=tol))


def steps_to_converge(x, f, df, tol=1e-5):
    return sum(1 for _ in newton_until_converged(x, f, df, tol))

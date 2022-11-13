import numpy as np
from scipy import optimize
from copy import deepcopy
from random import random
import matplotlib.pyplot as plt

def trunc_squared(f):
    return lambda x: max(f(x)**2, 0)

def f1(x):
    return 2*x[0]**2 + x[1]**2 + 6*x[0] + 5*x[1] + x[2]**2

def f2(x):
    return 3*(x[0] - 2)**2 + 2*(x[1] - 5)**2 + 5*x[2]**2

def g1(x):
    return 3*x[0] + x[1] + x[2] - 8

def g2(x):
    return 5*x[1] - x[0] - 10

def penalty(Gs):
    return lambda x: sum(trunc_squared(g)(x) for g in Gs)

def convoluted(Fs, perfect, Gs, Ws):
    return lambda x: sum(Ws[i]*(Fs[i](x) - perfect[i])**2 for i in range(len(Fs))) + penalty(Gs)(x)


def gradient_descent(f, x0, eps_1=1e-4, eps_2=1e-4, eps_grad=1e-4, n_limit=1e3):

    k = 0
    x = x0
    end = False
    f_directed = lambda x_, diff_ : lambda a: f(x_ + a*diff_)

    while True:
        grad_f = optimize.approx_fprime(x, f, eps_grad)

        if k > n_limit or np.linalg.norm(grad_f) < eps_1:
            # print(k)
            return x, k


        x_old = deepcopy(x)
        # left, right = svenn(f_directed(x, -grad_f), 0)
        # print(left, right)
        # a = half(f_directed(x, -grad_f), left, right)
        # print(x, grad_f)
        a = np.max(optimize.minimize(f_directed(x, -grad_f), 1).x)
        # print(a, grad_f, x)
        # print(a)
        x -=  a * grad_f

        if np.linalg.norm(x-x_old) < eps_1 and np.abs(f(x)-f(x_old)) < eps_2:
            if end:
                # print(k)
                return x, k
            else:
                end = True
                k += 1
                continue
        k += 1
        end = False

def perfect_point(Fs, Gs):
    f_perfect = [Fs[i](gradient_descent(func, np.ones(3, dtype=np.float64))[0]) for i, func in enumerate(Fs)]
    # print(f)
    pareto_front = []

    for min_point in range(50):
        separator = random()
        Ws = [separator, 1 - separator]
        f = convoluted(Fs, f_perfect, Gs, Ws)
        # print(f(np.zeros(3, dtype=np.float64)))
        x0 = np.array([-1, 3, 4], dtype=np.float64)
        min_point, _ = gradient_descent(f, x0)
        pareto_front.append(min_point)

    return np.array(pareto_front)



def main():
    # np.seterr(all='raise')
    Fs = [f1, f2]
    Gs = [g1, g2]
    pareto_set = perfect_point(Fs, Gs)
    # print(pareto_set)

    F1 = [f1(point) for point in pareto_set]
    F2 = [f2(point) for point in pareto_set]
    # print(F1, F2)
    plt.scatter(F1, F2)
    plt.title('Pareto front')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.show()


if __name__=='__main__':
    main()

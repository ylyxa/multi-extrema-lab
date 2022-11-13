import numpy as np
from random import random, shuffle, randint
from itertools import tee
from scipy import optimize
from copy import deepcopy
from random import random
import matplotlib.pyplot as plt

alpha = -10
beta = 10
n = 3

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

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def f1(x):
    return 2*x[0]**2 + x[1]**2 + 6*x[0] + 5*x[1] + x[2]**2

def f2(x):
    return 3*(x[0] - 2)**2 + 2*(x[1] - 5)**2 + 5*x[2]**2

def g1(x):
    return 3*x[0] + x[1] + x[2] - 8 <= 0

def g2(x):
    return 5*x[1] - x[0] - 10 <= 0

def trunc_squared(f):
    return lambda x: max(f(x)**2, 0)

def penalty(Gs):
    return lambda x: sum(trunc_squared(g)(x) for g in Gs)

def convoluted(Fs, perfect, Gs, Ws):
    return lambda x: sum(Ws[i]*(Fs[i](x) - perfect[i])**2 for i in range(len(Fs)))

def fitness(f, x):
    return 1/f(x)

def generate_population(Mp):
    global alpha, beta, n
    pop = []
    for i in range(Mp):
        p = np.random.uniform(alpha, beta, 3)
        # p = np.array([(beta - alpha)*p0[j] + alpha for j in range(3)])
        pop.append(p)
        # print(p)
    return pop

def reproduce(pair):
    offsprings = []
    for i in range(30):
        c = random()
        # print(pair)
        X = c * pair[0] + (1-c) * pair[1]
        if g1(X) and g2(X):
            offsprings.append(X)
        Y = c * pair[1] + (1-c) * pair[0]
        if g1(Y) and g2(Y):
            offsprings.append(Y)
        if len(offsprings) >= 2:
            return offsprings[:2]
    if len(offsprings):
        return offsprings[0], pair[1]
    else:
        return pair

def genetic_single(f, Np=1000, Mp=30, Pc=0.3, Pm=0.3):
    global n
    pop = generate_population(Mp)
    # print(pop)
    for _ in range(Np):
        # print(len(pop))
        cumul_fitness = []
        cumul_fitness.append(fitness(f, pop[0]))
        # print(len(cumul_fitness), len(p0op))
        for i in range(1, Mp):
            cumul_fitness.append(cumul_fitness[i-1] + fitness(f, pop[i]))
        new_pop = []
        # print(cumul_fitness[-1])
        for i in range(Mp):
            r = random()*cumul_fitness[-1]
            selected = min(filter(lambda i: cumul_fitness[i] >= r, range(Mp)))
            new_pop.append(pop[selected])
        pop = new_pop

        parents = []
        nonparents = []
        for p in pop:
            if random() < Pc:
                parents.append(p)
            else:
                nonparents.append(p)
        shuffle(parents)
        if len(parents) % 2 != 0:
            nonparents.append(parents.pop())

        pairs = pairwise(parents)
        for i, pair in enumerate(pairs):
            if i % 2 != 0:
                continue
            X, Y = reproduce(pair)
            parents[i] = X
            parents[i+1] = Y

        pop = parents + nonparents

        for p in pop:
            if random() < Pm:
                p[randint(0, n-1)] = np.random.uniform(alpha, beta, 1)

    return min(pop, key=f)

def genetic(Fs, Gs):
    f_perfect = [Fs[i](gradient_descent(func, np.ones(3, dtype=np.float64))[0]) for i, func in enumerate(Fs)]
    # print(f)
    pareto_front = []

    for min_point in range(15):
        separator = random()
        Ws = [separator, 1 - separator]
        f = convoluted(Fs, f_perfect, Gs, Ws)
        # print(f(np.zeros(3, dtype=np.float64)))
        x0 = np.array([-1, 3, 4], dtype=np.float64)
        min_point = genetic_single(f)
        pareto_front.append(min_point)

    return np.array(pareto_front)

def main():
    # np.seterr(all='raise')
    Fs = [f1, f2]
    Gs = [g1, g2]
    pareto_set = genetic(Fs, Gs)
    # print(pareto_set)

    F1 = [f1(point) for point in pareto_set]
    F2 = [f2(point) for point in pareto_set]
    # print(F1, F2)
    plt.scatter(F1, F2)
    plt.title('Pareto front')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.show()

if __name__ == '__main__':
    main()

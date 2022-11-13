import numpy as np
from random import random, shuffle, randint
from itertools import tee

alpha = 0
beta = 20
n = 3

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def f(x):
    global n
    # return sum((x[i]-10)**2 for i in range(n))
    return sum((x[i])**2 - 10*np.cos(2*np.pi*x[i]) + 10 for i in range(n))

def fitness(x):
    # print(f(x))
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

def genetic(f, Np=1000, Mp=30, Pc=0.3, Pm=0.3):
    global n
    pop = generate_population(Mp)
    # print(pop)
    for _ in range(Np):
        # print(len(pop))
        cumul_fitness = []
        cumul_fitness.append(fitness(pop[0]))
        # print(len(cumul_fitness), len(p0op))
        for i in range(1, Mp):
            cumul_fitness.append(cumul_fitness[i-1] + fitness(pop[i]))
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
            c = random()
            # print(pair)
            X = c * pair[0] + (1-c) * pair[1]
            Y = c * pair[1] + (1-c) * pair[0]
            parents[i] = X
            parents[i+1] = Y

        pop = parents + nonparents

        for p in pop:
            if random() < Pm:
                p[randint(0, n-1)] = np.random.uniform(alpha, beta, 1)

    return min(pop, key=f)


def main():
    print(genetic(f))

if __name__ == '__main__':
    main()

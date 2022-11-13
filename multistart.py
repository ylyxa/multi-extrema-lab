import numpy as np
from copy import deepcopy
from scipy import optimize

def shekel(a,b,c):
    A = [[4,4,4],
         [4, 4, 4],
         [4,4,4]]
    C = [c/3 for i in range(3)]
    return lambda x: - a * sum((1. / (C[i] + b * sum((x[j] - aij)**2 for j, aij in enumerate(A[i])))) for i in range(3))

def rho(x, y):
    return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2)

def nearest_neighbor(points, f, delta=1):
    flag = True
    while flag:
        if len(points) <= 1:
            break
        flag = False
        min_dist = 1000000000
        min_i = -1
        min_j = -1
        # points = list(filter(lambda x: x[0]<=8 and x[1]<=8 and x[2]<=8, points))
        for i in range(len(points)-1):
            for j in range(i+1, len(points)):
                if rho(points[i], points[j]) < min_dist:
                    min_dist = rho(points[i], points[j])
                    min_i = i
                    min_j = j
        # print(min_dist)
        if min_dist < delta:
            del points[j if f(points[min_i]) > f(points[min_j]) else i]
            flag = True
    # print(len(points))
    return points

def optimize_step(f, x0, eps_1=1e-5, eps_2=1e-5, eps_grad=1e-5, n_limit=1e4):
    x = x0
    f_directed = lambda x_, diff_ : lambda a: f(x_ + a*diff_)
    grad_f = optimize.approx_fprime(x, f, eps_grad)

    if np.linalg.norm(grad_f) < eps_1:
        # print(k)
        return x

    x_old = deepcopy(x)
    a = np.max(optimize.minimize(f_directed(x, -grad_f), 1).x)
    x -=  a * grad_f

    return x


def gradient_descent(f, x0, eps_1=1e-10, eps_2=1e-10, eps_grad=1e-5, n_limit=1e4):
    k = 0
    x = x0
    end = False
    f_directed = lambda x_, diff_ : lambda a: f(x_ + a*diff_)

    while True:
        grad_f = optimize.approx_fprime(x, f, eps_grad)

        if k > n_limit or np.linalg.norm(grad_f) < eps_1:
            # print(k)
            return x


        x_old = deepcopy(x)
        a = np.max(optimize.minimize(f_directed(x, -grad_f), 1).x)
        x -=  a * grad_f

        if np.linalg.norm(x-x_old) < eps_1 and np.abs(f(x)-f(x_old)) < eps_2:
            if end:
                # print(k)
                return x
            else:
                end = True
                k += 1
                continue
        k += 1
        end = False

def concurrent(f):
    points = []
    xs = np.random.uniform(0, 8, 100)
    ys = np.random.uniform(0, 8, 100)
    zs = np.random.uniform(0, 8, 100)
    for i in range(100):
        point = np.array([xs[i], ys[i], zs[i]])
        points.append(point)

    iterations = 1
    while True:
        # print(f'{iterations} iteration')
        iterations += 1
        # print(len(points))
        for i in range(len(points)):
            points[i] = optimize_step(f, points[i])
            # print(f'optimized point{i}')
        old_len = len(points)
        points = nearest_neighbor(points, f)
        if len(points) <= 1:
            return points
            break

    return [gradient_descent(f, i) for i in points]

def main():
    f = shekel(1, 1, 2)
    print(concurrent(f))

if __name__ == '__main__':
    main()

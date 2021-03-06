import json
import math
import random
import progressbar


def err_func(houses, x, a, b):
    return (a*x + b - houses[x]) ** 2


def normalize(set):
    max_value = max(set, key=lambda p: p[1])
    return list(map(lambda p: (p[0] / len(set), p[1]/max_value[1]), set))

values = []
BATCH_SIZE = 60
LEARNING_RATE = 1e-3
ITERATIONS = 1500000
for x in range(BATCH_SIZE):
    values.append((x, math.sin(math.pi * x / BATCH_SIZE) + random.random() / 10))

values = normalize(values)

polynomial_level = 3
coefficients = list(map(lambda _: 0, range(polynomial_level)))
errors = []
bar = progressbar.ProgressBar(max_value=ITERATIONS, redirect_stdout=True)
for i in range(ITERATIONS):
    bar.update(i)
    gradient = list(map(lambda _: 0, range(polynomial_level)))
    error = 0
    for x in range(len(values) - 1):
        y = 0
        for v in range(polynomial_level):
            y += coefficients[v] * (values[x][0] ** v)
        diff = y - values[x][1]
        for power in range(polynomial_level):
            gradient[power] += diff * (values[x][0] ** power)
        error += abs(diff)

    for c in range(polynomial_level):
        coefficients[c] -= LEARNING_RATE * gradient[c] / BATCH_SIZE
    if i % 1000 == 0:
        errors.append(error)
        print(coefficients)

calculated_line = []
for x in range(BATCH_SIZE):
    y = 0
    for v in range(polynomial_level):
        y += coefficients[v] * (x ** v)
    calculated_line.append(y)
with open('dump.json', mode='w') as f:
    f.write(json.dumps({
        "coefficients": coefficients,
        "points": values,
        "errors": errors,
    }))

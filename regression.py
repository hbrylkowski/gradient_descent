import random
from functools import reduce

import matplotlib.pyplot as plt


def err_func(houses, x, a, b):
    return (a*x + b - houses[x]) ** 2


def normalize(set):
    max_value = max(set)
    return list(map(lambda x: x/max_value, set))

houses = []
LEARNING_RATE = 0.0006
BATCH_SIZE = 100
BASE_PRICE = 10
for h in range(BATCH_SIZE):
    houses.append((BASE_PRICE + h * 2) + random.randrange(-10, 10))

houses = normalize(houses)

polynomial_level = 2
coefficients = list(map(lambda _: 0, range(polynomial_level)))
print(coefficients)
errors = []
prev_err = 1
min_index = 0
for i in range(30000):
    sums = list(map(lambda _: 0, range(polynomial_level)))
    error = 0
    for x in range(len(houses) - 1):
        y = 0
        for v in range(polynomial_level):
            y += coefficients[v] * (x ** v)
        # y = reduce(helper, range(polynomial_level))
        diff = y - houses[x]
        for power, s in enumerate(sums):
            sums[power] += diff * (x ** power)
        # sums = list(map(lambda s: s[1] + diff * x ** s[0], enumerate(sums)))
        # error += diff ** 2

    # errors.append(error)
    for c in range(polynomial_level):
        coefficients[c] -= LEARNING_RATE * sums[c] / BATCH_SIZE
    # coefficients = list(map(lambda _: _[0] - LEARNING_RATE * _[1] / BATCH_SIZE, zip(coefficients, sums)))

calculated_line = []
for x in range(BATCH_SIZE):
    y = 0
    for v in range(polynomial_level):
        y += coefficients[v] * (x ** v)
    calculated_line.append(y)
print(coefficients)
#plt.plot(errors)
plt.plot(calculated_line)
plt.plot(range(len(houses)), houses, 'ro', markersize=2)
#plt.axis([0, len(houses), 0, max(houses) + 10])
# plt.savefig('draw.png')
plt.show()

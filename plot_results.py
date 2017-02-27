import json
import matplotlib.pyplot as plt

with open('dump.json', mode='r') as f:
    file = json.loads(f.read())

coefficients = file["coefficients"]
errors = file["errors"]
points = file["points"]

calculated_line = []
for x in range(len(points)):
    y = 0
    for v in range(len(coefficients)):
        y += coefficients[v] * (x ** v)
    calculated_line.append(y)

plt.subplot(1, 2, 1)
plt.plot(errors)
plt.subplot(1, 2, 2)
plt.plot(calculated_line)
plt.plot(range(len(points)), points, 'ro', markersize=2)
plt.show()

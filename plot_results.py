import json
import matplotlib.pyplot as plt

with open('dump.json', mode='r') as f:
    file = json.loads(f.read())

coefficients = file["coefficients"]
errors = file["errors"]
points = file["points"]

calculated_line_x = []
calculated_line_y = []
for x in range(1000):
    x /= 1000
    calculated_line_x.append(x)
    y = 0
    for v in range(len(coefficients)):
        y += coefficients[v] * ((x) ** v)
    calculated_line_y.append(y)
plt.subplot(1, 2, 1)
plt.plot(errors)
plt.subplot(1, 2, 2)
plt.axis([0, 1, 0, 1])
ax = plt.gca()
ax.set_autoscale_on(False)
plt.plot(calculated_line_x, calculated_line_y)
plt.plot(list(map(lambda p: p[0], points)), list(map(lambda p: p[1], points)), 'ro', markersize=4)
plt.show()

import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3*x**2 - 4*x + 5

xs = np.arange(-5, 5, 0.25)
ys = f(xs)
plt.plot(xs,ys)
#plt.show()

h = 0.000001

x = 3.0
print((f(x + h) - f(x)) / h)

a = 2.0
b = -3.0
c = 10.0
d1 = a*b + c
c += h
d2 = a*b + c
print(d1)
print(d2)
print('slope, ', (d2 - d1)/ h)




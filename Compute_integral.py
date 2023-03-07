import numpy as np
from scipy.integrate import quad

def f(t):
    return 1/np.sqrt(2*np.pi)*np.exp(-t**2/2)

def g(x):
    return quad(lambda t: f(t), -np.inf, 0.00554496*(x-4.393637)/np.sqrt(0.9362382))[0]

def h(x):
    return 2/np.sqrt(2*0.9362382*np.pi)*np.exp(-(x-4.393637)**2/(2*0.9362382))*g(x)

result = quad(h, -np.inf, 1.5)[0]
print(f"-np.inf, 1.5 The numerical value of the integral is: {result:.3f}")

result = quad(h, 1.5, 2.5)[0]
print(f"1.5, 2.5 The numerical value of the integral is: {result:.3f}")

result = quad(h, 2.5, 3.5)[0]
print(f"2.5, 3.5 The numerical value of the integral is: {result:.3f}")

result = quad(h, 3.5, 4.5)[0]
print(f"3.5, 4.5 The numerical value of the integral is: {result:.3f}")

result = quad(h, 4.5, 5.5)[0]
print(f"4.5, 5.5 The numerical value of the integral is: {result:.3f}")

result = quad(h, 5.5, 6.5)[0]
print(f"5.5, 6.5 The numerical value of the integral is: {result:.3f}")

result = quad(h, 6.5, np.inf)[0]
print(f"6.5 np.inf The numerical value of the integral is: {result:.3f}")

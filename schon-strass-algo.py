import numpy as np
import time
import os

def FFT(a, w):
    n = len(a)
    if n == 1:
        return a
    a0 = FFT(a[::2], w**2)
    a1 = FFT(a[1::2], w**2)
    z = np.array([w**i * a1[i] for i in range(n//2)])
    return np.concatenate([a0 + z, a0 - z])

def multiply(x, y):
    n = 1 << (len(x) + len(y) - 2).bit_length()
    w = np.exp(2j * np.pi / n)
    fx = np.concatenate([x, np.zeros(n - len(x))])
    fy = np.concatenate([y, np.zeros(n - len(y))])
    fx = FFT(fx, w)
    fy = FFT(fy, w)
    z = np.fft.ifft(fx * fy).real.round().astype(int)
    carry = np.cumsum(z[::-1] // 10)[::-1]
    z = z % 10
    z[:-1] += carry[1:]
    z[-1] += carry[0] - carry[1]
    return z

def save_output(x, y, result):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = "output_{}.txt".format(timestamp)
    with open(filename, 'w') as f:
        f.write("Multiplication of {} and {}:\n".format(x, y))
        f.write("Result: {}\n".format(''.join(map(str, result[::-1]))))

def main():
    x = np.array(list(map(int, str(input("Enter the first number: ")))))[::-1]
    y = np.array(list(map(int, str(input("Enter the second number: ")))))[::-1]
    result = multiply(x, y)
    save_output(x[::-1], y[::-1], result)

if __name__ == "__main__":
    main()

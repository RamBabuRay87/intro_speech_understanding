import numpy as np

def minimum_Fs(f):
    '''
    Find the lowest sampling frequency that would avoid aliasing for a pure tone at f Hz.
    '''
    Fs = 2 * f   # Nyquist sampling theorem
    return Fs

def omega(f, Fs):
    '''
    Find the radial frequency (omega) that matches a given f and Fs.
    '''
    omega = 2 * np.pi * f / Fs
    return omega

def pure_tone(omega, N):
    '''
    Create a pure tone of N samples at omega radians/sample.
    '''
    n = np.arange(N)
    x = np.cos(omega * n)
    return x  

f = 10          # frequency in Hz
Fs = minimum_Fs(f)
w = omega(f, Fs)
x = pure_tone(w, 10)

print("Input frequency f =", f, "Hz")
print("Minimum sampling frequency Fs =", Fs, "Hz")
print("Radial frequency omega =", w, "rad/sample")
print("Pure tone samples:")
print(x)

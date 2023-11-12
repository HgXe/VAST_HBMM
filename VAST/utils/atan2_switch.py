import csdl
import numpy as np 

def atan2_switch(x, y, scale=10000.):
    # x > 0
    f1 = csdl.arctan(y/x) * (0.5*csdl.tanh(scale*(x-1.e-2)) + 0.5)
    # x < 0
    f2 = (csdl.arctan(y/x) + np.pi) * (0.5*csdl.tanh(scale*(-1.e-2-x)) + 0.5)*(0.5*csdl.tanh(scale*(y)) + 0.5) # y >= 0
    f3 = (csdl.arctan(y/x) - np.pi) * (0.5*csdl.tanh(scale*(-1.e-2-x)) + 0.5)*(0.5*csdl.tanh(scale*(-y)) + 0.5) # y < 0
    # x = 0
    f4 = np.pi/2 * (0.5*(csdl.tanh(scale*(x+1e-2)) - csdl.tanh(scale*(x-1e-2)))) * (0.5*csdl.tanh(scale*(y+1e-3)) + 0.5) # y >= 0
    f5 = -np.pi/2 * (0.5*(csdl.tanh(scale*(x+1e-2)) - csdl.tanh(scale*(x-1e-2)))) * (0.5*csdl.tanh(scale*(1e-3-y)) + 0.5) # y < 0

    return f1 + f2 + f3 + f4 + f5
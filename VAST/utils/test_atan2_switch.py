import csdl
import numpy as np
from VAST.utils.atan2_switch import atan2_switch

class CSDLSwitchTest(csdl.Model):
    def initialize(self):
        self.parameters.declare('input_shape')
    
    def define(self):
        input_shape = self.parameters['input_shape']
        x = self.declare_variable('x', shape=input_shape)
        y = self.declare_variable('y', shape=input_shape) 

        scale = 10000.
        theta = atan2_switch(x=x, y=y, scale=scale)
        self.register_output('theta', theta)
        # # x > 0
        # f1 = csdl.arctan(y/x) * (0.5*csdl.tanh(scale*(x-1.e-2)) + 0.5) * 180/np.pi
        # # x < 0
        # f2 = (csdl.arctan(y/x) + np.pi) * (0.5*csdl.tanh(scale*(-1.e-2-x)) + 0.5)*(0.5*csdl.tanh(scale*(y)) + 0.5)* 180/np.pi # y >= 0
        # f3 = (csdl.arctan(y/x) - np.pi) * (0.5*csdl.tanh(scale*(-1.e-2-x)) + 0.5)*(0.5*csdl.tanh(scale*(-y)) + 0.5)* 180/np.pi # y < 0
        # # x = 0
        # f4 = np.pi/2 * (0.5*(csdl.tanh(scale*(x+1e-2)) - csdl.tanh(scale*(x-1e-2)))) * (0.5*csdl.tanh(scale*(y+1e-3)) + 0.5)* 180/np.pi # y >= 0
        # f5 = -np.pi/2 * (0.5*(csdl.tanh(scale*(x+1e-2)) - csdl.tanh(scale*(x-1e-2)))) * (0.5*csdl.tanh(scale*(1e-3-y)) + 0.5)* 180/np.pi # y < 0

        # self.register_output('f1', f1)
        # self.register_output('f2', f2)
        # self.register_output('f3', f3)
        # self.register_output('f4', f4)
        # self.register_output('f5', f5)

        # self.register_output('theta', f1 + f2 + f3 + f4 + f5)
    
from python_csdl_backend import Simulator


x = np.array([0., 0., 1., -1.])
y = np.array([1., -1., 0., 0., ])

n = 10000
radius = 10
t = np.linspace(0,2*np.pi,n)
x, y = np.cos(t)*radius, np.sin(t)*radius

# x = np.array([0.5, -0.5, -0.5, 0.5])
# y = np.array([0.5, 0.5, -0.5, -0.5])
# expected results for theta (deg): 45, 135, -135, -45
theta_exp = np.arctan2(y,x) * 180/np.pi
input_shape = x.shape

model = CSDLSwitchTest(input_shape=input_shape)
sim = Simulator(model)

sim['x'] = x
sim['y'] = y

sim.run()
theta = sim['theta']

# print('f1:', sim['f1'])
# print('f2:', sim['f2'])
# print('f3:', sim['f3'])
# print('f4:', sim['f4'])
# print('f5:', sim['f5'])

theta_error = theta_exp - theta

print('expected theta (deg)', theta_exp)
print('theta (deg)', theta)
# print('error in theta: ', theta_exp - theta)

# print('====')
# print(x)
# print(y)
# print('====')

print(theta[4995:5002])

import matplotlib.pyplot as plt
plt.figure()
plt.plot(theta, 'k*', label='CSDL atan2 switch')
plt.plot(theta_exp, 'c', label='Expected')
plt.xlabel('index')
plt.ylabel('theta (degrees)')
plt.grid()
plt.legend()

plt.figure()
plt.plot(theta_error, 'k*', label='Theta error')
plt.xlabel('index')
plt.ylabel('error in theta (degrees)')
plt.grid()
plt.legend()

plt.show()
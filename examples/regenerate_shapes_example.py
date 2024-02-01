import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
#%%
from gamma_simulator import gamma_simulator
import matplotlib.pyplot as plt

simulator = gamma_simulator(dict_type='double_exponential',
                            verbose=True)
s = simulator.generate_signal()
shapes = simulator.re_regenerate_shape_dict()
                                                                        
plt.plot(shapes.T)
plt.show()

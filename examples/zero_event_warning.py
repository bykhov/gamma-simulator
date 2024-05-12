import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from gamma_simulator import gamma_simulator

simulator = gamma_simulator(lambda_value=0.001,
                            seed=44)
s = simulator.generate_signal()

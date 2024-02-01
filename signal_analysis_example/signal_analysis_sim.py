import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import seaborn as sns

from gamma_simulator import gamma_simulator

(param1weights, param1bins, param2weights, param2bins) = np.load('gamma_shape_parameters.npz').values()
(energy_bins, energy_weights) = np.load('energy_histogram.npz').values()
#%%
simulator = gamma_simulator(verbose=True,
                            source={'hist_energy': energy_bins,
                                    'hist_counts': energy_weights},  # Energy histogram
                            lambda_value=0.005,
                            signal_len=1000000,
                            dict_type='gamma',
                            dict_shape_params={'custom': True,
                                               'param1bins': param1bins,
                                               'param1weights': param1weights,
                                               'param2bins': param2bins,
                                               'param2weights': param2weights},
                            noise=5,
                            dict_size=100,
                            seed=42)
s = simulator.generate_signal()

#%%

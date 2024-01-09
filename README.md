# Gamma_simulator
Our script can be used to mimic the signal collected from the gamma source, which has a large number of adjustable features and has powerful simulation functions
## Use
### Environment
Make sure you have the following libraries in your environment
* numpy
* scipy
* matplotlib
* urllib  
You can use the following command to install the libaries
```bash
pip install numpy scipy matplotlib urllib
```
### Import
```python
from gamma_simulator import gamma_simulator
```

### Run
Step 1.Creat an instance
```python
simulator = gamma_simulator()
```
Step 2.Define parameters
```python
simulator = gamma_simulator(verbose=True,
                            verbose_plots={'shapes': True, 'signal': True},
                            source={'name': 'Co-60', 'weights': 1},
                            signal_len=1,  # "analog" signal of 1 second that are 1e7 samples
                            fs=10e5,
                            lambda_value=1e4,
                            dict_type='gamma',
                            dict_shape_params={'mean1':  1.1,
                                               'std1': 0.001,
                                               'mean2': 1e5,
                                               'std2': 1e3},
                            noise_unit='std',
                            noise=1e-3,
                            dict_size=10,
                            seed=42)
```
Step 3.Creat the signal
```python
signal = simulator.generate_signal()
```
## Parameter description
* source: Target elements and weights for simulation.
* signal_len: The length for signal.
* fs: Sampling frequency.
* dict_size: Pulse shape dictionary size.
* dict_type: Dictionary type (currently only double_exponential or gamma is supported).
* dict_shape_params: Dictionary shape parameters.
* noise: Noise level.
* noise_unit: Unit of noise（std or snr）.
* seed: Random number generation seed.
* verbose_plot:Corresponding drawing include the following  
  _Shapes : pulse shapes_  
  _Energy : The energy spectrum of the simulated element_  
  _Signal : The simulated signal_
### spectrum
* spectrum_range:The maximum range of the energy spectrum(KeV)
* spectrum_bin:The bin number of the X-axis of the energy spectrum
  

## Notice
If you are not familiar with shape parameters, use the following combination of parameters
```python
{dict_type='gamma',
dict_shape_params={'mean1':  1.1,
'std1': 0.001,
'mean2': 1e5,
'std2': 1e3}
```
or
```python
{dict_type='double_exponential',
dict_shape_params={'mean1': 1e-7, 
'std1': 1e-9,
'mean2': 1e-5,
'std2': 1e-7}
```
### Plot setting
Our simulator supports drawing a variety of graphs, including energy, shape, signal and spectrum.
* Energy：Ideal energy spectrum of the drawn signal source (simulator built-in database)
* Shape：Draws a dictionary set of all possible signal shapes
* Signal：When the length of the resulting signal is less than 2000, the generated signal is drawn, and when the length is greater than 2000, the first 2000 sampling points are drawn
* Spectrum：Energy spectrum of the analog signal, including pulse stacking

The default option is not to draw, if you need to draw, you need to change the specified value in the parameter definition to True
```
verbose_plots={'energy':True, 'shapes': True, 'signal': True, 'spectrum':True}
```
## Examples
```python
from gamma_simulator import gamma_simulator
  simulator = gamma_simulator(verbose=True,
                            verbose_plots={'shapes': True, 'signal': True},
                            source={'name': 'Co-60', 'weights': 1},
                            signal_len=1,  # "analog" signal of 1 second that are 1e7 samples
                            fs=10e6,
                            lambda_value=1e4,
                            dict_type='double_exponential',
                            dict_shape_params={'mean1': 1e-7,  # continuous-time parameters measured in seconds
                                               'std1': 1e-9,
                                               'mean2': 1e-5,
                                               'std2': 1e-7},
                            noise_unit='std',
                            noise=1e-3,
                            dict_size=10,
                            seed=42)
signal = simulator.generate_signal()
```
```python
from gamma_simulator import gamma_simulator
simulator = gamma_simulator(verbose=True,
                            verbose_plots={'shapes': True, 'signal': True},
                            source={'name': ['Co-60', 'I-125'], 'weights': [1, 2]},
                            signal_len=1,  # "analog" signal of 1 second that are 1e7 samples
                            fs=10e6,
                            lambda_value=1e4,
                            dict_type='gamma',
                            dict_shape_params={'mean1':  1.1,  # continuous-time parameters measured in seconds
                                               'std1': 0.001,
                                               'mean2': 1e5,
                                               'std2': 1e3},
                            noise_unit='std',
                            noise=1e-3,
                            dict_size=10,
                            seed=42)
signal = simulator.generate_signal()
```
 

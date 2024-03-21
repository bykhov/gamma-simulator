![logo](./fig/logo.png)
# Gamma_simulator
This is a gamma pulse simulator jointly developed by [Shamoon College of Engineering(SCE)](https://en.sce.ac.il/) in Israel and [Shanghai Advanced Research Institute](http://www.sari.cas.cn/),CAS in China.Here we will give a brief introduction to our software, including the what and why. For more specific implementation steps of the software, please refer to our [paper](). Of course,**if you are a pure user, please jump directly to [Use](#Use) to see how to use it**.

## Contents

- [Introduction](#Introduction)
  - [What is Gamma Simulator](#what-is-gamma-simulator)
  - [Why do we creat it](#why-do-we-creat-it)
- [Software structure](#software-structure)
  - [Macrostructure](#macrostructure)
  - [Implementation structure](#implementation-structure)
  - [Parameter description](#parameter-description)
  - [Main function](#main-function)
- [Use](#use)
  - [Install](#install)
  - [Import](#import)
  - [Run](#run)
- [Notice](#notice)
  - [Shape parameter](#shape-parameter)
  - [Plot setting](#plot-setting)
- [Examples](#examples)
- [Contributors](#contributors)
- [Known issue](#known-issue)
- [Todo](#todo)

## Introduction
### What is Gamma Simulator?

Gamma simulator is a gamma pulse simulator with parameter customization function, you can specify the type of radioactive source and pulse count rate and other characteristics, generate pulse signals that meet the corresponding characteristics

### Why do we creat it?

The original intention of the gamma simulator was to introduce deep learning into energy spectroscopy in the later stage. The use of deep learning to process pulse signals requires that the collected pulse signals have corresponding labels, which is impossible in commercial energy spectrometer. Therefore, we used the simulator to label the pulse signals while generating them, so as to facilitate the reference of deep learning methods. At the same time, simulators can greatly reduce the manpower, material and financial resources of the signal collection process, and can be used to preliminarily test signal processing methods

## Software structure
### Macrostructure
![mainflow](./fig/mainflow.png)
### Implementation structure
 ![Flow_software](./fig/Flow_software.png)
### Parameter description
|**Setting Parameters:**||type|Default value|
| --- | -----------|-----------|-----------|
| verbose   | Whether to output detailed information   |bool|False |
| verbose_plots   | Whether images need to be output   |dict|None|
| source   | The simulated radioactive source   |str or dict |'Co-60' |
| signal_len   | Length of time to simulate sampling(s)   |int or float|1024|
| fs   | Analog sampling rate   |float |1 |
| lambda_value   | Analog pulse count rate(cps)   |float|0.1|
| dict_type    | Shape type model of the simulated pulse   |str|'gamma'|
| dict_shape_params   | dict shape params   |dict|Please see [Notice](#notice)|
| noise_unit   | Unit of noise   |str|'std'|
| noise   | The magnitude of noise in the given unit   |float|0.01|
| dict_size   | Shape dictionary size due to jitter   | int|100 |
| seed   | The simulated random number seed   |int|None|

**The above parameters can be set and customized by users. The chart shows the default values of parameters and draws discrete pulse signals. For specific parameter Settings in applications, please refer to the [example section](#examples) ,more specific parameter Settings and parameter tests are presented in the [example folder](/examples)**

|**Shape parameters:**| |type | eg |
| --- | -----------|-----------|-----------|
| t_rise   | rise time of the shape   | float |4.560e-07 |
| t_fall   | fall time of the shape   | float |6.134e-05  |
| shape_len   | length of the shape in samples   | int | 61|
| shape_len_sec   | length of the shape in seconds   | float | 6.180e-05|
|**Events parameters:**| | | |
|hist_energy|The energy contained in the spectrum (the transverse axis of the desired spectrum)| array |[0.02,0.04,0.06,... ,1667.20] |
|hist_counts|The probability of generating energy corresponding to the simulated source(vertical axis of the desired energy spectrum)|array|[0.0002,0.001,0.0015,... ,0.0001]|
| n_events   | number of events in the signal   | int | 16054|
| times   | arrival times of the events   |array |[0.7e-05,1.8e-05,... ,0.92] |
| energies   | The energy sequence of the pulses produced in this simulation|array|[223.2,453.6,889.4,... ,635.4]|
| lambda_measured   | actual event rate   | float | 13672.2 |
| shape_param1, shape_param2   | shape parameters for each event   | array |two sequences of n_event values conforming to the Gaussian distribution of the given parameter|
|**Signal parameters:**| |
| signal_len   | The number of samples at a given frequency in the simulated time  | int or float| 1e6 |
| signal_len_sec   | The length of time of the analog signal(s)   | int or float| 1 |
| duty_cycle   | The proportion of the time to detect the signal to the total analog time  |float | 0.67 |
| pile_up_stat   | number of the pile-ups in the generated signal   | int | 4302 |
| measured_snr   | measured SNR of the generated signal (dB)   |float| 66.26 |

**These values are intermediate values generated during the simulation, so there are no default values. Here are some examples of these values to give the reader an idea of what these values are**

### Main function
|**Function name**|**input**|**output**|**Function action**|
| --- | ---|-----------|-----------|
| *load_weighted_spectrum*  |source |hist_energy hist_counts|The energy spectrum of the simulated source (both elemental and mixture)|
| *generate_energy_distribution*  |seed, n_events, hist_energy, hist_counts|energies| Generate *n_event* pulses energy sequences conforming to the probability density distribution of the target energy spectrum|
| *generate_arrival_times*   | seed, lambda_value|times| Generates a Poisson process event arrival time series with a specified count rate(lambda_value) |
| *generate_all_dict_shapes*  | dict_type,dict_shape_params, dict_size| shapes | Produces a shape dictionary of the specified shapes generated by the specified parameters as a selection library of energy shapes|
| *generate_signal_without_noise*   |times, energies,shapes_len, shapes | *signal without noise*| The independent pulses and shapes that we have generated are pieced together to form an analog noiseless signal|
| *generate_signal_with_noise*  |*signal without noise*, noise_unit, noise|signal|The noiseless signal is superimposed on the specified unit and size of noise to obtain a real analog signal|

**A large number of parameters mentioned above are used in the function introduction. In our real code, there is a high degree of integration and a large number of function reuse. Some inputs are not directly called in this function, but by calling other functions, and some output intermediate outputs are directly called by another function after the result is generated, so these temporary variables are not reflected in the description. So this function introduction is not particularly strict, but it is definitely the most suitable for readers to clarify the code logic of the introduction**
## Use
### Install
Make sure you have the following libraries in your environment
* numpy
* scipy
* matplotlib
* urllib  
(It doesn't matter that you don't have these, because these dependencies will be installed when you install the gamma-simulator package)

Please use the following command to install our program
```bash
pip install gamma-simulator
```
### Import
```python
from gamma_simulator.gamma_simulator import gamma_simulator
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
                            dict_shape_params={'mean1':  0.1,
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


## Notice
### Shape parameter
If you are not familiar with shape parameters, use the following combination of parameters
```python
{dict_type='gamma',
dict_shape_params={'mean1':  0.1,
'std1': 0.001,
'mean2': 1e5,
'std2': 1e3}
```
or
```python
{dict_type='double_exponential',
dict_shape_params={'mean1': 1e-5, 
'std1': 1e-7,
'mean2': 1e-7,
'std2': 1e-9}
```
### Plot setting
Our simulator supports drawing a variety of graphs, including energy, shape, signal and spectrum.
* Energy：Ideal energy spectrum of the drawn signal source (simulator built-in database)
* Shape：Draws a dictionary set of all possible signal shapes
* Signal：When the length of the resulting signal is less than 2000, the generated signal is drawn, and when the length is greater than 2000, the first 2000 sampling points are drawn


The default option is not to draw, if you need to draw, you need to change the specified value in the parameter definition to True
```
verbose_plots={'energy':True, 'shapes': True, 'signal': True}
```
## Examples

```python
from gamma_simulator.gamma_simulator import gamma_simulator
simulator = gamma_simulator(verbose=True,
                            verbose_plots={'shapes': True, 'signal': True},
                            source={'name': 'Co-60', 'weights': 1},
                            signal_len=1,  # "analog" signal of 1 second that are 1e7 samples
                            fs=10e6,
                            lambda_value=1e4,
                            dict_type='double_exponential',
                            dict_shape_params={'mean1': 1e-5,  # continuous-time parameters measured in seconds
                                               'std1': 1e-7,
                                               'mean2': 1e-7,
                                               'std2': 1e-9},
                            noise_unit='std',
                            noise=1e-3,
                            dict_size=10,
                            seed=42)
signal = simulator.generate_signal()
```



```python
from gamma_simulator.gamma_simulator import gamma_simulator
simulator = gamma_simulator(verbose=True,
                            verbose_plots={'energy': True, 'signal': True},
                            source={'name': ['Co-60', 'I-125'], 'weights': [1, 2]},
                            signal_len=1,  # "analog" signal of 1 second that are 1e7 samples
                            fs=10e6,
                            lambda_value=1e4,
                            dict_type='gamma',
                            dict_shape_params={'mean1':  0.1,  # continuous-time parameters measured in seconds
                                               'std1': 0.001,
                                               'mean2': 1e5,
                                               'std2': 1e3},
                            noise_unit='std',
                            noise=1e-3,
                            dict_size=10,
                            seed=42)
signal = simulator.generate_signal()
```

You can see the result in [examples](./examples)
## Contributors
[Dima Bykhovsky](https://github.com/bykhov),[Tom Trigano](https://github.com/TomTrigano),[Zikang Chen](https://github.com/ZikangC)
## Known issue
  
## Todo 

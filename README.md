![logo](./fig/logo.png)
# Gamma_simulator
     
This is a gamma pulse simulator jointly developed by [Shamoon College of Engineering(SCE)](https://en.sce.ac.il/) in Israel and [Shanghai Advanced Research Institute](http://www.sari.cas.cn/) in China.Here we will give a brief introduction to our software, including the what and why. For more specific implementation steps of the software, please refer to our [paper](). Of course,**if you are a pure user, please jump directly to [Use](#Use) to see how to use it**.For any questions about the software, you can leave a message or send an email to me, I will reply as soon as possilble

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

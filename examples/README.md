# Goal
This file is to test the function and robustness of the code, we can first review the settings parameter that users can adjust.
 
|**Setting Parameters:**||type|Default value|
| --- | -----------|-----------|-----------|
| verbose   | Whether to output detailed information   |bool|False |
| verbose_plots   | Whether images need to be output   |dict|None|
| source   | The simulated radioactive source   |str or dict |'Co-60' |
| signal_len   | Length of time to simulate sampling(s)   |int or float|1024|
| fs   | Analog sampling rate   |float |1 |
| lambda_value   | Analog pulse count rate(cps)   |float|0.1|
| dict_type    | Shape type model of the simulated pulse   |str|'gamma'|
| dict_shape_params   | dict shape params   |dict|Please see [Notice](./README.md#notice)|
| noise_unit   | Unit of noise   |str|'std'|
| noise   | The magnitude of noise in the given unit   |float|0.01|
| dict_size   | Shape dictionary size due to jitter   | int|100 |
| seed   | The simulated random number seed   |int|None|

 We will test and adjust the code according to these settings, so as to prove that our code is stable and effective
* *Default parameters test in [default_test](default_test.ipynb):* Having the default parameter configuration in our program means that the program should work perfectly without any parameter input. 

# Basic function tests

## [source](source.ipynb)

* In this test, the focus was on assessing pure substances and mixtures to ensure the simulator's capability in simulating pulse signals of mixtures.
  
## [signal_length](signal_length.ipynb)

* In this test, evaluations were performed with tests of different lengths. Excessive test length revealed potential issues related to insufficient memory space. Hence, it is recommended to limit the simulation duration to 1000 seconds at a frequency of 10e6.

## [fs](fs.ipynb)

* The variable **fs** represents the sampling rate. In this test, frequencies, including the standard frequency of 10e6, were examined to ensure the proper functioning of the simulator



## [dict_type](Pulse_shape.ipynb)

* The simulator has implemented two shape models. In this test, two shapes were randomly selected, while the remaining numerical parameters were also randomly changed to verify the correct execution of the program under different shape parameters


## [noise](noise.ipynb)

* Incorporated noise is a characteristic of each signal and can be introduced in various manners. During this test, we examined the inclusion of both normal distribution noise and fixed Signal-to-Noise Ratio (SNR) noise. This design ensures the versatility of our program, making it well-suited for users with diverse preferences

# Advanced functionality tests
* Getting out the shapes dictionary by `re_regenerate_shape_dict()` in [regenerate_shapes_example.py](regenerate_shapes_example.py).
* When the pulse count rate is very low, the warning is issued as show in [zero_event_warning.py](zero_event_warning.py). In this case, longer signal and/or higher event rate are recommended.
  


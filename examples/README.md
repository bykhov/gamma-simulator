<!--- Duplicated from the main readme.
This is a specific introduction examples of the file，First let's look at the parameters we can change mainly:
|Parameter name  |Parameter description|
| --- | -----------|
| verbose   | Whether to output detailed information   |
| verbose_plots   | Whether images need to be output   |
| source   | The simulated radioactive source   |
| signal_len   | Length of time to simulate sampling(s)   |
| fs   | Analog sampling rate   |
| lambda_value   | Analog pulse count rate(cps)   |
| dict_type    | Shape type model of the simulated pulse   |
| dict_shape_params   | dict shape params   |
| noise_unit   | Unit of noise   |
| noise   | The magnitude of noise in the given unit   |
| dict_size   | Shape dictionary size due to jitter   |
| seed   | The simulated random number seed   | -->

## Examples and tests
* *Default parameters test in [defult_test](defult_test.ipynb):* Having the default parameter configuration in our program means that the program should work perfectly without any parameter input. 
**Zikang** please add all the numeric class parameter outputs to this code.
### Advanced functionality tests
* Getting out the shapes dictionary by `re_regenerate_shape_dict()` in [regenerate_shapes_example.py](regenerate_shapes_example.py).
* When the pulse count rate is very low, the warning is issued as show in [zero_event_warning.py](zero_event_warning.py). In this case, longer signal and/or higher event rate are recommended.

# Defult_test



In this section, I mainly tested the program's performance with no parameter input and given some literal parameters, and tested whether the program ran properly with the default parameters in the double exponential model and the gamma model

# Pulse_shape

Our simulator implemented two shape models. In this test, I randomly selected the two shapes, while the rest of the numerical parameters were randomly changed to test that the program could run correctly under different shape parameters

# fs

fs represents the sampling rate. In this test, we tested the frequency including the common frequency 10e6 to ensure that the simulator can work normally

# noise

Each signal contains noise, and noise can be added in different ways. In this test, we tested the addition of both normal distribution noise and fixed SNR noise, so that our program can be more suitable for users with different habits

# signal_length

In this test, I gave tests of different lengths. When the length is too large, the problem of insufficient memory space will occur. It is recommended that the simulation should not exceed 1000 seconds at the frequency of 10e6.

# source

In this test I mainly tested pure substances and mixtures to ensure that our simulator was able to simulate the pulse of mixtures

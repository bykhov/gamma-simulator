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

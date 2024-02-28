## Examples and tests
* *Default parameters test in [default_test](default_test.ipynb):* Having the default parameter configuration in our program means that the program should work perfectly without any parameter input. 
**Zikang** please add all the numeric class parameter outputs to this code.
### Advanced functionality tests
* Getting out the shapes dictionary by `re_regenerate_shape_dict()` in [regenerate_shapes_example.py](regenerate_shapes_example.py).
* When the pulse count rate is very low, the warning is issued as show in [zero_event_warning.py](zero_event_warning.py). In this case, longer signal and/or higher event rate are recommended.

## [Default_test](default_test.ipynb)

* In this section, performance tests were conducted on the program with no parameter input as well as with some predefined literal parameters. The aim was to ensure the proper functioning of the program under default parameters in both the double exponential model and the gamma model.

## [Pulse_shape](Pulse_shape.ipynb)

* The simulator has implemented two shape models. In this test, two shapes were randomly selected, while the remaining numerical parameters were also randomly changed to verify the correct execution of the program under different shape parameters
## [fs](fs.ipynb)

* The variable **fs** represents the sampling rate. In this test, frequencies, including the standard frequency of 10e6, were examined to ensure the proper functioning of the simulator

## [noise](noise.ipynb)

* Incorporated noise is a characteristic of each signal and can be introduced in various manners. During this test, we examined the inclusion of both normal distribution noise and fixed Signal-to-Noise Ratio (SNR) noise. This design ensures the versatility of our program, making it well-suited for users with diverse preferences

## [signal_length](signal_length.ipynb)

* In this test, evaluations were performed with tests of different lengths. Excessive test length revealed potential issues related to insufficient memory space. Hence, it is recommended to limit the simulation duration to 1000 seconds at a frequency of 10e6.
* 
## [source](source.ipynb)

* In this test, the focus was on assessing pure substances and mixtures to ensure the simulator's capability in simulating pulse signals of mixtures.

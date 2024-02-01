## Goal
The goal of this example is to provide a simulation that reflects the properties of the experimentally measured signal. 

### Signal description
The signal is a raw HPGe measurement signal. The signal is 1 second long and has a sampling frequency of 10MHz. The signal is sampled by a 16-bit resolution A/D converter and is stored in `unit16` format. For convenience, the file is stored in Matlab's `mat` format. The excerpt of a signal is presented below. The plot illustrates a few essential characteristics of the signal:
* Bias level
* Example of a saturated peak
* Example of a detector's dead time.

<img src="signal_excerpt.png" width="400"/>

### Signal pre-processing
The signal pre-processing in [signal_analysis.ipynb](signal_analysis.ipynb) includes the following steps.
#### Bias removal
The bias was approximated by a threshold level that is slightly higher than the median value. All the values below the threshold were zeroed.

#### Pulse shapes segmentation
The pulse shapes were segmented by identification of up and down signal properties. Then, all the resulting segments were converted to `float` and stored for further processing. About $1.5\times 10^5$ segments were identified.

#### Remove saturated shapes
The saturation level corresponds to the highest value represented by `unit16` format. All the segments (about 4,000) with the highest possible values were removed.

#### Remove low-amplitude segments
* The segments with the peak amplitude below the predefined threshold were considered as noise and removed.
* Remove segments with too high and too low energies
* Remove too short-length segments
* Pile-up rejection of segments with more than one peak
* Remove segments where the maximum is in the second half of the segment that they are physically incorrect (only a few of these are present in the signal)

#### Summary
The resulting signal is presented below.

<img src="signal_segments_examples.png" width="400"/>

### Fitting
* All the segments were fitted with gamma shape MSE fit. The fitting examples are presented below. Note fitting takes a few minutes to run.
* 
<img src="signal_segments_fitting_gamma.png" width="300"/>

* Fitting results clean-up: The segments with abnormally high-cost function values and outliers of $\alpha$ and $\beta$ were removed.

* The resulting distribution of the fitted $\alpha$ and $\beta$ parameters is stored for further simulation and is presented below.

<img src="alpha_beta_param.png" width="400"/>

* The resulting energy histogram for 512 is stored for further simulation and is presented below. 

<img src="signal_energy_histogram.png" width="400"/>

### Simulation
The simulation is performed with parameters $\alpha$ and $\beta$ and the energy histogram that was precalculated from the experimentally measured signal in [signal_analysis_sim.py](signal_analysis_sim.py) file.

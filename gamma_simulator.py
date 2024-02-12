import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import urllib
import warnings
import urllib.request
import urllib.error


# Classes
class gamma_simulator:
    def __init__(self,
                 source: str or dict = None,
                 signal_len: int or float = 1024,
                 lambda_value: float = 0.1,
                 fs: float = 1,
                 dict_size: int = 100,
                 dict_type: str = 'gamma',  # 'double_exponential'
                 dict_shape_params=None,
                 noise: float = 0.01,
                 noise_unit='std',
                 seed: int = None,
                 enforce_edges: bool = True,
                 verbose: bool = False,
                 verbose_plots: dict = None):
        """Simulate a gamma signal with the following parameters:
        source: name of the source from the gamma spectrum database (https://gammadb.nuclearphoenix.xyz/)
            or a custom spectrum. The usage is one of the following:
            * name of the source, e.g. 'Co-60'
            * a dictionary with list of sources and list of their weights, e.g.
            {'name':['Co-60, I-125'], 'weights':[0.5, 0.7]}
            * a dictionary with custom spectrum, e.g.
            {'hist_energy': hist_energy, 'hist_counts': hist_counts}
            hist_energy: energy values [keV] (bins, x-axis)
            hist_counts: amount of counts for each hist_energy value (y-axis)
            Note: only one of source and custom_spectrum can be used!
        signal_len: length of the signal in seconds
        lambda_value: event rate in Hz (events per second)
        fs: sampling frequency in Hz
        dict_size: size of the dictionary
        dict_type: type of the dictionary. Currently, 'double_exponential' and 'gamma' are supported.
        dict_shape_params: dictionary parameters (dictionary of parameters)
           parameters of the shape are generated normally with the following parameters:
                mean1: mean value of the first parameter
                std1: standard deviation of the first parameter
                mean2: mean value of the second parameter
                std2: standard deviation of the second parameter
           for custom parameters distribution, use the following format:
                custom: True
                param1val: values of the first parameter
                param2val: values of the second parameter
                use 'None' as a dict_size to remove warning about the dictionary size
        noise: noise level according to noise_unit
        noise_unit: 'std' for standard deviation of per sample noise,
                    'snr' for signal-to-noise ratio in dB
        enforce_edges: if True, any pulse shape have its starting or ending points within the signal
        seed: random seed for reproducibility
        verbose: print general information
        verbose_plots: illustrative plots of the signal, shapes, energy spectrum
            dict: dictionary of plots to show, e.g. {'energy': True, 'shapes': True, 'signal': True}

        -------------------------------------------
        --- Useful properties ---
        Shape parameters:
        t_rise: rise time of the shape
        t_fall: fall time of the shape
        shape_len: length of the shape in samples
        shape_len_sec: length of the shape in seconds

        Events parameters:
        n_events: number of events in the signal
        times: arrival times of the events
        energies: energy values for each event
        lambda_measured: actual event rate
        shape_param1, shape_param2: shape parameters for each event
        param1dict, param2dict: shape parameters for each dictionary entry
        shape dictionary (shape, time) may be re-generated using re_regenerate_shape_dict()
        
        Signal parameters:
        signal_len: length of the signal in samples
        signal_len_sec: length of the signal in seconds
        duty_cycle: duty cycle of the signal
        pile_up_stat: number of the pile-ups in the generated signal
        measured_snr: measured SNR of the generated signal (dB)
        -------------------------------------------
        """
        if verbose_plots is None:
            verbose_plots = dict()
        verbose_plots['energy'] = verbose_plots.get('energy', False)
        verbose_plots['shapes'] = verbose_plots.get('shapes', False)
        verbose_plots['signal'] = verbose_plots.get('signal', False)
        if source is None:
            source = 'Co-60'
        self.verbose = verbose
        self.verbose_plots = verbose_plots
        self.seed = seed
        self.enforce_edges = enforce_edges
        # --- load the spectrum ---
        # hist_energy: energy values [keV] (bins, x-axis
        # hist_counts: amount of counts for each hist_energy value (y-axis)
        if isinstance(source, str):  # handle str of a single source as a dictionary
            source = {'name': source, 'weights': 1}
        if 'name' in source.keys() and 'weights' in source.keys():
            # load the spectrum from the database with isotope name = source
            source['weights'] = np.array(source['weights']).reshape(-1)
            if isinstance(source['name'], str):
                assert len(source['weights']) == 1, "Only one weight is expected for a single source"
            else:
                assert len(source['name']) == len(source['weights']), "Number of sources and weights must be the same"
            assert np.all(source['weights'] >= 0), "Weights must be positive"
            assert len(source['name']) > 0, "Number of sources must be greater than zero"
            self.hist_energy, self.hist_counts = self.load_weighted_spectrum(source)
            self.weights = source['weights']
        elif 'hist_energy' in source.keys() and 'hist_counts' in source.keys():
            # load the spectrum from the dictionary
            self.energy_desc = 'Custom'
            self.hist_energy = source['hist_energy']
            self.hist_counts = self.normalize_histogram(
                np.array(source['hist_counts'])
            )
            assert self.hist_energy.shape == self.hist_counts.shape, "Counts and energies must have the same shape"
            self.weights = np.array([1])
        else:
            raise ValueError(f'Unknown source type: {source}')
        if self.verbose_plots['energy']:
            self.verbose_plot_energy()

        self.fs = fs
        self.dt = 1 / self.fs  # sampling period
        # --- shape dictionary generation ---
        self.dict_size = dict_size
        assert dict_type in ['double_exponential', 'gamma'], "Unknown shape type"
        self.dict_type = dict_type
        if dict_shape_params is None:
            if self.dict_type == 'double_exponential':
                # discrete-time parameters !!!
                dict_shape_params = {'mean2': 2,  # tau1
                                     'std2': 0.1,
                                     'mean1': 10,  # tau2
                                     'std1': 0.1}
            elif self.dict_type == 'gamma':
                dict_shape_params = {'mean1': 1.1,  # alpha
                                     'std1': 0.001,
                                     'mean2': 0.1,  # beta
                                     'std2': 0.001}
        # check if custom parameters are used explicitly
        dict_shape_params['custom'] = dict_shape_params.get('custom', False)
        if dict_shape_params['custom']:
            self.dict_shape_params = self.custom_shape_params_stats(dict_shape_params)
            # update the dictionary size for custom parameters
            # issue a warning if dict_size is not equal to the number of parameters
            if self.dict_size is None and (len(self.dict_shape_params['param1val']) != self.dict_size):
                warnings.warn(f'Dictionary size is automatically set to the number of shape parameters.')
            self.dict_size = len(self.dict_shape_params['param1val'])
        else:
            self.dict_shape_params = dict_shape_params
        # non-random shape parameters
        if self.dict_shape_params['std1'] == self.dict_shape_params['std2'] == 0 and self.dict_size > 1:
            warnings.warn(f'Both std1 and std2 are zero, but dict_size is greater than 1.')

        # shape_len_sec: length of the shape in seconds
        # shape_len: length of the shape in samples
        # t_rise: rise time in seconds or samples
        self.shape_len, self.shape_len_sec, self.t_rise = self.evaluate_shape_len()
        self.t_fall = self.shape_len_sec - self.t_rise  # fall time
        # --- Time parameters ---
        # lambda_value: event rate in event/second
        self.lambda_value = lambda_value
        # lambda_n: event rate in event/sample
        self.lambda_n = lambda_value * self.dt  # = lambda_value / fs
        # signal_len_sec: length of the signal in seconds
        self.signal_len_sec = signal_len
        # signal_len: length of the signal in samples
        self.signal_len = int(self.signal_len_sec * self.fs)
        if self.signal_len < self.shape_len:
            raise ValueError(f'Signal length must be greater than the shape length,'
                             f' but signal_len = {self.signal_len} and shape_len = {self.shape_len}')
        if self.enforce_edges:
            self.signal_len_samples = self.signal_len
        else:
            # The shape is DO allowed to exceed the signal boundaries
            # The generated signal is extended by 2 * shape_len_sec and then truncated back to signal_len
            self.signal_len_samples = self.signal_len + self.shape_len * 2

        self.noise = noise
        assert noise_unit in ['std', 'snr'], "Unknown noise unit"
        self.noise_unit = noise_unit
        # calculate the duty cycle
        self.duty_cycle = self.lambda_value * self.shape_len_sec
        if self.dict_shape_params['custom']:
            self.param1dict, self.param2dict = self.dict_shape_params['param1val'], self.dict_shape_params['param2val']
        else:
            # parameters of the shape (per each dictionary entry)
            self.param1dict, self.param2dict = self.generate_normal_dict_shape_parameters()
        # parameters of the signal (will be calculated later)
        self.lambda_measured = None
        self.energies = None
        self.n_events = None
        self.times = None
        self.measured_snr = None
        self.shape_param1 = None  # shape parameters for each event
        self.shape_param2 = None  # shape parameters for each event

    def verbose_info(self):
        """Print general information about the simulated signal.
        """
        print('-- General information ------------------------------------------')
        print(f'Loaded spectrum for {self.energy_desc} source')
        if len(self.weights) != 1:
            print(f'Weights are {self.weights}')
        print(f'Energy spectrum between {self.hist_energy.min():.1f} and {self.hist_energy.max():.1f} keV '
              f'with {self.hist_energy.shape[0]} bins')
        if self.enforce_edges:
            print(f'Shapes are NOT allowed to exceed the signal boundaries')

        if self.fs == 1:  # discrete time
            print(f'Discrete-time parameters are used!')
            print(f'Signal length : {self.signal_len_sec} samples')
            print(f'Number of events: {self.n_events} (randomly generated)')
            print(f'Activity {self.lambda_value:.3f} event per sample and '
                  f'actual activity is {self.lambda_measured:.3f} events per sample')
            print(f'Shape model: {self.dict_type}')
            print(f'Number of {self.dict_type} shapes in the dictionary: {self.dict_size}')
            if self.dict_shape_params['custom']:
                print('Custom shape distribution is used.')
            else:
                print('Normal shape distribution is used.')
            print(f'Shape parameters:  {self.dict_shape_params["mean1"]:1.3f}±{self.dict_shape_params["std1"]:1.3f}'
                  f' and {self.dict_shape_params["mean2"]:1.3f}±{self.dict_shape_params["std2"]:1.3f}')
            print(f'Each shape has a length of {self.shape_len_sec * self.fs:.3f} samples '
                  f'rounded to {self.shape_len} samples')
            print(f'Rise time is {self.t_rise:.3f} samples and fall time is {self.t_fall:.3f} samples')
        else:  # continuous time
            print(f'Sampling frequency: {self.fs} samples per second')
            print(f'Signal length is {self.signal_len_sec} sec that '
                  f'are {int(self.signal_len_sec * self.fs)} samples')
            print(f'Events: {self.n_events} (randomly generated)')
            print(f'Activity {self.lambda_value:.3f} event per second and '
                  f'actual activity is {self.lambda_measured:.3f} events per second')
            print(f'Normalized lambda value: {self.lambda_n:.3e} events per sample')
            print(f'Shape model: {self.dict_type}')
            print(f'Number of {self.dict_type} shapes in the dictionary: {self.dict_size}')
            if self.dict_shape_params['custom']:
                print('Custom shape distribution is used.')
            else:
                print('Normal shape distribution is used.')
            if self.dict_type == 'double_exponential':
                print(f'Shape parameters: tau1 = {self.dict_shape_params["mean1"]} '
                      f'sec ±{self.dict_shape_params["std1"]:1.3e}'
                      f' ({self.dict_shape_params["mean1"] * self.fs:.2f} samples) '
                      f'and tau2 = {self.dict_shape_params["mean2"]} sec ±{self.dict_shape_params["std2"]:0.3e}'
                      f' ({self.dict_shape_params["mean2"] * self.fs:.2f} samples) ')
            elif self.dict_type == 'gamma':
                print(f'Shape parameters: '
                      f'alpha = {self.dict_shape_params["mean1"]} ±{self.dict_shape_params["std1"]:1.3e}'
                      f' and beta = {self.dict_shape_params["mean2"]} ±{self.dict_shape_params["std2"]:1.3e}')
            print(f'Each shape has a length of {self.shape_len_sec:1.3e} sec that are {self.shape_len} samples')
            print(f'Rise time is {self.t_rise:.3e} sec and fall time is {self.t_fall:.3e} sec')
        # duty cycle and pile-up probability
        print(f'Duty cycle is given by {self.duty_cycle:.2f}'
              f' with theoretical pile-up probability of {1 - np.exp(-self.duty_cycle):.3f}')
        if self.n_events != 0:
            print(f'Actual pile-up probability is {self.pile_up_stat() / self.n_events:.3f} with'
                  f' {self.n_events - self.pile_up_stat()} non-pile-up events'
                  f' out of {self.n_events} events')
        # noise
        if self.noise_unit == 'std':
            print(f'Noise level: ±{self.noise} per sample')
        elif self.noise_unit == 'snr':
            print(f'Noise level: {self.noise} dB')
        print(f'Measured SNR: {self.measured_snr:.2f} dB')
        # random seed
        if self.seed is None:
            print('Random seed is not defined')
        else:
            print(f'Pre-defined random seed {self.seed} is used')

    def pile_up_stat(self):
        """evaluate number of the pile-ups in the generated signal"""
        # calculate the time difference between events
        d_time = self.times[1:] - self.times[:-1]
        # pile-up happens when the time difference less than the shape length
        pile_ups = np.sum(d_time < self.shape_len)
        return pile_ups

    def verbose_plot_energy(self):
        """Plot the energy histogram"""
        plt.bar(self.hist_energy, self.hist_counts, width=1, alpha=0.5)
        plt.yscale('log')
        plt.grid(linestyle='--', linewidth=1, color='gray')
        plt.xlabel('Energy [keV]')
        plt.ylabel('Normalized counts')
        if self.energy_desc == 'Custom':
            plt.title(f'User-defined energy spectrum with {self.hist_energy.shape[0]} bins')
        else:
            plt.title(f'{self.energy_desc} energy spectrum with {self.hist_energy.shape[0]} bins')
        plt.show()

    # --- Energy spectrum -----------------------------------------------------
    @staticmethod
    def normalize_histogram(counts: np.ndarray) -> np.ndarray:
        """Normalize the spectrum to the number of counts
        The resulting histogram is normalized to 1
        counts: counts values
        return: normalized counts
        """
        counts = counts / counts.sum()
        return counts

    def load_spectrum_data(self, source) -> tuple[np.ndarray, np.ndarray]:
        """Load the gamma spectrum database from https://github.com/OpenGammaProject/Gamma-Spectrum-Database/
        return: tuple of hist_energy and counts, where counts are normalized to 1
        """
        url = ('https://raw.githubusercontent.com/bykhov/Gamma-Spectrum-Database/main/assets/spectra/'
               + source
               + '.html')
        try:
            html_page = urllib.request.urlopen(url).read()
        except urllib.error.HTTPError:
            raise ValueError(f'Unknown isotope name: {source}')

        # parse the html page and extract the spectrum
        page = str(html_page)
        start_idx = page.find('Clean Spectrum')
        x_idx = page.find('"x":', start_idx) + 5
        x_str = page[x_idx:page.find(']', x_idx)]
        x_str = x_str.replace('"', '')
        x = np.fromstring(x_str, sep=',')
        y_idx = page.find('"y":', x_idx) + 5
        y_str = page[y_idx:page.find(']', y_idx)]
        y_str = y_str.replace('"', '')
        y = np.fromstring(y_str, sep=',')
        y[y < 0] = 0  # remove weird negative values
        energy, counts = x, self.normalize_histogram(y)
        return energy, counts

    def load_weighted_spectrum(self, source) -> tuple[np.ndarray, np.ndarray]:
        """Load the gamma spectrum database of a list of sources with weights
        """
        self.energy_desc = source['name']
        if isinstance(source['name'], str):  # single source
            source['name'] = [source['name']]
        # load the sources from the dictionary
        for i, source_name in enumerate(source['name']):
            hist_energy, hist_counts = self.load_spectrum_data(source_name)
            hist_counts *= source['weights'][i]
            if i == 0:  # initialize variables
                total_hist_energy = hist_energy
                total_hist_counts = hist_counts
            else:  # update for each source
                total_hist_energy += hist_energy
                total_hist_counts += hist_counts
        total_hist_counts = self.normalize_histogram(total_hist_counts)
        return total_hist_energy, total_hist_counts

    # --- Time ---------------------------------------------------------------
    def generate_arrival_times(self,
                               outage_prob: float = 1e-12) -> np.ndarray:
        """Generate a sequence of events with a given number of samples and events rate.
        signal_len_samples: length of the signal in samples
        lambda_n: event rate in event/sample
        outage_prob: probability of missing events in a frame of length signal_len_samples (not used)
        return: array of arrival times
        """
        # vector implementation does not check for number of events through a loop (!)
        # cdf(outage_prob) is used to calculate the number of events to generate
        # the number of events is calculated according to the Poisson distribution

        np.random.seed(self.seed)
        # number of events in a frame to guarantee outage probability
        max_number_of_events = stats.poisson.ppf(1 - outage_prob, mu=self.lambda_n * self.signal_len_samples).astype(
            int)
        # generate events times
        times = np.cumsum(np.random.exponential(1 / self.lambda_n, max_number_of_events))
        # remove events after the end of the frame
        times = times[times < self.signal_len_samples - self.shape_len]
        return times

    def generate_energy_distribution(self) -> np.ndarray:
        """Generate a hist_energy distribution according to the number of events (per each arrival time).
        Each hist_energy value corresponds to the arrival time of an event.
        return: array of hist_energy values
        """

        # generate random hist_energy values
        np.random.seed(self.seed)
        energies = np.random.choice(self.hist_energy,
                                    size=self.n_events,
                                    replace=True,
                                    p=self.hist_counts)
        return energies

    # --- Shapes -------------------------------------------------------------
    @staticmethod
    def custom_shape_params_stats(dict_shape_params: dict) -> dict:
        """Evaluate the parameters of the shape from the custom distribution.
        dict_shape_params: dictionary of shape parameters
            param1val: values of the first parameter
            param2val: values of the second parameter
        return:
            dictionary of shape parameters mean1, std1, mean2, std2
            param1val and param2val are converted to numpy arrays
        """
        if len(dict_shape_params['param1val']) != len(dict_shape_params['param2val']):
            raise ValueError('Number of parameters for the first and second parameter must be the same')
        dict_shape_params['mean1'] = np.mean(dict_shape_params['param1val'])
        dict_shape_params['mean2'] = np.mean(dict_shape_params['param2val'])
        dict_shape_params['std1'] = np.std(dict_shape_params['param1val'])
        dict_shape_params['std2'] = np.std(dict_shape_params['param2val'])
        dict_shape_params['param1val'] = np.array(dict_shape_params['param1val'])
        dict_shape_params['param2val'] = np.array(dict_shape_params['param2val'])
        return dict_shape_params

    def evaluate_shape_len(self) -> tuple[int, float, float]:
        """Evaluate the average length of the shape for the simulation
        return:
            shape_len: length of the shape in samples
            shape_time: length of the shape in seconds
            t_rise: rise time in seconds (always)
        """
        if self.dict_type == 'double_exponential':
            assert self.dict_shape_params['mean1'] > self.dict_shape_params['mean2'], \
                "tau2 must be greater than tau1"
            shape_time = 6 * (self.dict_shape_params['mean1'] + 3 * self.dict_shape_params['std1'])
            # calculate the rise time
            tr = ((self.dict_shape_params["mean1"] * self.dict_shape_params["mean2"]) /
                  (self.dict_shape_params["mean1"] + self.dict_shape_params["mean2"]) *
                  np.log(self.dict_shape_params["mean1"] / self.dict_shape_params["mean2"]))
        elif self.dict_type == 'gamma':
            assert self.dict_shape_params['mean1'] > 0 and self.dict_shape_params['mean2'] > 0, \
                "alpha and beta must be positive"
            shape_time = stats.gamma.ppf(0.995, self.dict_shape_params['mean1'] - 1,
                                         scale=1 / self.dict_shape_params['mean2'])
            # calculate the rise time
            tr = self.dict_shape_params["mean1"] / self.dict_shape_params["mean2"]
        else:
            raise ValueError(f'Unknown shape type: {self.dict_type}')
        shape_len = int(shape_time * self.fs)
        return shape_len, shape_time, tr

    def generate_normal_dict_shape_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate random parameters for the double exponential shape.
        return: tuple of param1 and param2 are shape parameters for each dictionary entry
        """
        np.random.seed(self.seed)
        # generate parameters
        # param1values and param2values are parameters for each shape in the dictionary
        param1dict = np.random.normal(self.dict_shape_params['mean1'],
                                      self.dict_shape_params['std1'],
                                      self.dict_size)
        param2dict = np.random.normal(self.dict_shape_params['mean2'],
                                      self.dict_shape_params['std2'],
                                      self.dict_size)
        assert np.all(param1dict > 0), "First parameter of the shape must be positive - please check the parameters"
        assert np.all(param2dict > 0), "Second parameter of the shape must be positive - please check the parameters"
        if self.dict_type == 'double_exponential':
            assert np.all(param1dict > param2dict), "tau1 must be greater than tau2"
        return param1dict, param2dict

    def generate_random_shape_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate random parameters for the shape.
        return: tuple of shape_param1 and shape_param2 are shape parameters for each event
        """
        np.random.seed(self.seed)
        if self.dict_shape_params['custom']:
            x = self.dict_shape_params['param1val']
            y = self.dict_shape_params['param2val']
            idx = np.random.randint(0, len(x), size=self.n_events)
            self.shape_param1, self.shape_param2 = x[idx].reshape(-1, 1), y[idx].reshape(-1, 1)
        else:
            # assign the parameters to events
            # sample from the dictionary for each event
            # reshape to column vector for broadcasting
            self.shape_param1 = np.random.choice(self.param1dict, size=self.n_events, replace=True).reshape(-1, 1)
            self.shape_param2 = np.random.choice(self.param2dict, size=self.n_events, replace=True).reshape(-1, 1)

    @property
    def generate_all_dict_shapes(self) -> np.ndarray:
        """Generate a double exponential shape dictionary.
        Shape is generated for each event in the signal.
        """
        self.generate_random_shape_parameters()
        # define x-axis values:
        # dimension 0 is the number of shapes, dimension 1 is the number of samples
        # each start time is shifted by the arrival time of the event
        n = np.r_[0:self.shape_len] - np.reshape(self.times / self.dt % 1.0, (-1, 1))
        if self.dict_type == 'double_exponential':
            s = np.exp(-n * self.dt / self.shape_param2) - np.exp(-n * self.dt / self.shape_param1)
            s[:, 0] = 0  # set the first sample to zero
        elif self.dict_type == 'gamma':
            # n[:, 0] = 0
            s = stats.gamma.pdf(n * self.dt, self.shape_param1 - 1, scale=1 / self.shape_param2)
            # If the built-in function is used, the parameter is subtracted by one,
            # s = ((n * self.dt) ** self.shape_param1) * np.exp(-self.shape_param2 * n * self.dt)
        # normalize the shape
        s /= s.sum(axis=1, keepdims=True)
        # note: storing the dictionary may be memory consuming
        if self.verbose_plots['shapes']:
            self.verbose_plot_shapes()
        return s

    def re_regenerate_shape_dict(self) -> np.ndarray:
        """Re-generate the shape dictionary
        without time-shifts and without re-generating the shape parameters"""
        n = np.r_[0:self.shape_len]
        if self.dict_type == 'double_exponential':
            shape_dict = (np.exp(-n * self.dt / self.param1dict.reshape(-1, 1)) -
                          np.exp(-n * self.dt / self.param2dict.reshape(-1, 1)))
            shape_dict[:, 0] = 0  # set the first sample to zero
        elif self.dict_type == 'gamma':
            # n[:, 0] = 0
            shape_dict = stats.gamma.pdf(n * self.dt,
                                         self.param1dict.reshape(-1, 1) - 1,
                                         scale=1 / self.param2dict.reshape(-1, 1))
        else:
            raise ValueError(f'Unknown shape type: {self.dict_type}')
        return shape_dict

    def verbose_plot_shapes(self):
        # re-generate time-synchronized shapes for plotting
        s = self.re_regenerate_shape_dict()
        # plot the shapes
        if self.dict_size > 100:
            # plot only the first 100 shapes if the dictionary is too large
            plt.plot(s[:100, :].T, alpha=0.25)
            plt.title(f'Shapes in the pulse shapes dictionary (100 first shapes)')
        else:
            plt.plot(s.T, alpha=0.35)
            plt.title(f'Shapes in the pulse shapes dictionary ({self.dict_size} shapes '
                      f'with length of {self.shape_len} samples)', fontsize=9)
        plt.grid(linestyle='--', linewidth=1, color='gray')
        plt.xlabel('Discrete time [n]')
        plt.ylabel('Amplitude')
        plt.xlim([0, self.shape_len - 1])
        plt.show()

    # --- Signal -------------------------------------------------------------
    # Signal is generated without enforcing the edges and then the edges are truncated if needed
    def generate_signal_without_noise(self) -> np.ndarray:
        """Generate a noiseless signal.
        """
        # generate the dictionary
        shapes = self.generate_all_dict_shapes
        # generate the signal
        signal = np.zeros(self.signal_len_samples)
        for i in range(self.n_events):
            # for each event
            # add the shape to the signal
            signal[int(self.times[i]):int(self.times[i]) + self.shape_len] \
                += self.energies[i] * shapes[i, :]

        return signal

    def generate_signal_with_noise(self) -> np.ndarray:
        """Add noise to the signal.
        """
        np.random.seed(self.seed)
        # generate the noiseless signal
        signal = self.generate_signal_without_noise()
        # generate white Gaussian noise
        if self.noise_unit == 'std':
            noise_std = self.noise
        else:  # if self.noise_unit == 'snr':
            noise_std = 10 ** (-self.noise / 20) * np.linalg.norm(signal) / np.sqrt(len(signal))
        noise = np.random.normal(0, scale=noise_std, size=self.signal_len_samples)
        signal_norm = np.linalg.norm(signal)
        if self.n_events != 0:
            # measured SNR
            self.measured_snr = 20 * np.log10(np.linalg.norm(signal) / np.linalg.norm(noise))
        else:
            # no signal since there are no events
            self.measured_snr = -np.inf

        # add noise to the signal
        signal += noise
        return signal

    def generate_signal(self) -> np.ndarray:
        # times: event times
        # n_events: number of events in times vector
        self.times = self.generate_arrival_times()
        self.n_events = len(self.times)
        if self.n_events == 0:
            warnings.warn(f'Signal has {self.n_events} events. '
                          f'Longer signal and/or higher event rate are recommended.')
        self.lambda_measured = self.n_events / self.signal_len_sec  # actual event rate
        # energy: energy values for each event
        self.energies = self.generate_energy_distribution()

        signal = self.generate_signal_with_noise()
        if not self.enforce_edges:
            # remove the edges
            signal = signal[self.shape_len:-self.shape_len]
        # verbose information
        if self.verbose_plots['signal']:
            if len(signal) > 2000:
                # plot only the first 2000 samples is signal is too long
                plt.plot(signal[:2000])
                plt.title(f'Signal with additive noise (first 2000 samples)')
            else:
                plt.plot(signal)
                plt.title(f'Signal with additive noise')
            plt.xlabel('Time [n]')
            plt.ylabel('Signal amplitude')
            plt.title(f'Signal with additive noise')
            plt.axis('tight')
            plt.grid()
            plt.show()
        if self.verbose:
            self.verbose_info()
        return signal


if __name__ == '__main__':
    # simulator = gamma_simulator()
    # s = simulator.generate_signal()
    # simulator.verbose_info()
    # # # %%
    simulator = gamma_simulator(verbose=True,
                                verbose_plots={'signal': True, 'shapes': True},
                                source={'name': ['Co-60', 'I-125'], 'weights': [1, 2]},
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
    s = simulator.generate_signal()
    #
    # # %%
    # simulator = gamma_simulator(verbose=True,
    #                             verbose_plots={'shapes': True, 'signal': True},
    #                             source={'name': 'Co-60', 'weights': 1},
    #                             lambda_value=0.1,
    #                             dict_type='gamma',
    #                             noise_unit='std',
    #                             noise=1e-3,
    #                             dict_size=10,
    #                             enforce_edges=False,
    #                             seed=42)
    # s = simulator.generate_signal()

    # %%
    simulator = gamma_simulator(verbose=True,
                                verbose_plots={'shapes': True, 'signal': True},
                                source={'name': 'Co-60', 'weights': 1},
                                lambda_value=0.1,
                                dict_type='gamma',
                                dict_shape_params={'custom': True,
                                                   'param1val': [2, 3, 4],
                                                   'param2val': [0.4, 0.5, 0.66]},
                                noise=1e-3,
                                dict_size=None,
                                enforce_edges=False,
                                seed=44)
    s = simulator.generate_signal()

#%%

import unittest

import numpy as np

from data_processing import preprocessing


class PreprocessingTests(unittest.TestCase):
    """Tests for the preprocessing module in the data_processing package."""

    PATH_TO_SORTING_RESULTS = "/net/bs-filesvr02/export/group/hierlemann/intermediate_data/Maxtwo/phornauer/Chemogenetics/Large_dose_range/well014/sorter_output/"

    def test_load_sorting_results(self):
        # Test case 1: Test loading sorting results with default segment index
        sorting_results = preprocessing.load_sorting_results(
            self.PATH_TO_SORTING_RESULTS
        )
        self.assertIsNotNone(sorting_results)
        # Add more test cases for different scenarios

    def test_generate_unit_spike_train_list(self):
        # Test case 1: Test generating unit spike train list with default parameters
        spike_times = np.array([1, 2, 3, 4, 5])
        spike_templates = np.array([0, 1, 0, 1, 0])
        spike_train_list = preprocessing.generate_unit_spike_train_list(
            spike_times, spike_templates
        )
        self.assertIsNotNone(spike_train_list)
        # Add more test cases for different scenarios

    def test_find_significant_firing_rate_changes(self):
        # Test case 1: Test finding significant firing rate changes with default parameters
        significant_changes = preprocessing.find_significant_firing_rate_changes(
            self.PATH_TO_SORTING_RESULTS
        )
        self.assertIsNotNone(significant_changes)
        # Add more test cases for different scenarios

    def test_generate_firing_rate_bins(self):
        # Test case 1: Test generating firing rate bins with default bin size
        spike_times = np.array([1, 2, 3, 4, 5])
        firing_rate_bins = preprocessing.generate_firing_rate_bins(spike_times)
        self.assertIsNotNone(firing_rate_bins)
        # Add more test cases for different scenarios

    def test_bootstrap_firing_rates(self):
        # Test case 1: Test bootstrapping firing rates with pre and post binned data
        pre_binned = np.array([1, 2, 3, 4, 5])
        post_binned = np.array([6, 7, 8, 9, 10])
        bootstrap_results = preprocessing.bootstrap_firing_rates(
            pre_binned, post_binned
        )
        self.assertIsNotNone(bootstrap_results)
        # Add more test cases for different scenarios


if __name__ == "__main__":
    unittest.main()

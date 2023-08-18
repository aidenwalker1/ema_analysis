#!/usr/bin/python

# Contains functions to calculate features from sensor data.

# Written by Diane J. Cook, Washington State University.

# Copyright (c) 2020. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.


import math

import numpy as np
from scipy.stats import moment


def mean_absolute_deviation(data, axis=None):
    """ Compute mean absolute deviation of data.
    Input data is an array with an optional specified axis of computation.
    """
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


def median_absolute_deviation(data, axis=None):
    """ Compute median absolute deviation of data.
    Input data is an array with an optional specified axis of computation.
    """
    return np.median(np.absolute(data - np.mean(data, axis)), axis)


def zero_crossings(data, median):
    """ Compute zero crossings of the input array of data. Zero crossings is
    computed as the number of times the data value crosses the median as the
    sequence is traversed from beginning to end.
    """
    rel = 0
    count = 0

    for x in data:
        if x < median:
            if rel > 0:
                count += 1
            rel = -1
        elif x > median:
            if rel < 0:
                count += 1
            rel = 1
        else:
            rel = 0
    return count


def mean_crossings(data, mean):
    """ Compute zero crossings of the input array of data. Mean crossings is
    computed as the number of times the data value crosses the mean as the
    sequence is traversed from beginning to end.
    """
    rel = 0
    count = 0

    for x in data:
        if x < mean:
            if rel > 0:
                count += 1
            rel = -1
        elif x > mean:
            if rel < 0:
                count += 1
            rel = 1
        else:
            rel = 0
    return count


def moments(data):
    """ Calculate the 1st through 4th moments about the mean for the
    input array of data. These describe the shape of the data.
    """
    m1 = moment(data, 1)
    m2 = moment(data, 2)
    m3 = moment(data, 3)
    m4 = moment(data, 4)
    return m1, m2, m3, m4


def fft_features(data):
    """ Calculate the periodic components that express the input array of data
    using a Discrete Fourier Transform.
    """
    fft_feature = np.fft.fft(data)
    ceps = fft_feature[0].real
    lceps = len(fft_feature)
    psd = []
    psdtotal = 0.0
    energy = 0.0
    for i in range(lceps):
        r = fft_feature[i].real
        energy += r
        value = (r * r) / float(lceps) + 1e-08
        psd.append(value)
        psdtotal += value
    entropy = 0.0
    for i in range(lceps):
        entropy -= psd[i] * math.log(psd[i])
    msignal = np.mean(fft_feature).real
    vsignal = np.var(fft_feature).real
    return ceps, entropy, energy, msignal, vsignal


def interquartile_range(data):
    """ Calculate the interquartile range of the input array of data.
    This is the difference in values between the 25% value and 75% value of
    the sorted array of values.
    """
    newlist = sorted(data)
    num = len(newlist)
    x = newlist[(int(num / 4))]
    y = newlist[(int((3 * num) / 4))]
    return y - x


def skewness(data, mean):
    """ Calculate the amount of skewness, or leaning toward low or high values,
    in the input array of data.
    """
    sum1 = 0
    sum2 = 0
    for d in data:
        sum1 += (d - mean) ** 3
        sum2 += (d - mean) ** 2
    n1 = sum1 / len(data)
    n2 = sum2 / len(data)
    n2 = n2 ** 1.5
    if n2 == 0:
        return 0.0
    else:
        return n1 / n2


def kurtosis(data, mean, std):
    """ Calculate the amount of kurtosis, or peakedness/flatness,
    in the input array of data.
    """
    sum1 = 0
    for d in data:
        sum1 += (d - mean) ** 4
    n1 = sum1 / len(data)
    n2 = std ** 4
    if n2 == 0:
        return -3.0
    else:
        return (n1 / n2) - 3.0


def signal_energy(data):
    """ Calculate the signal energy of the data array. Because the data are
    discrete-time, energy is defined as the sum of the squared values.
    """
    signal_sum = 0
    for d in data:
        signal_sum += d ** 2
    return signal_sum


def log_signal_energy(data):
    """ Calculate the log signal energy of the data array. This is defined as
    the sum of log base 10 of the squared values.
    """
    signal_sum = 0
    for d in data:
        d2 = d * d
        if d2 != 0:
            signal_sum += math.log(d2, 10)
    return signal_sum


def signal_magnitude_area(x, y, z):
    """ Calculate the signal magnitude area of the input data array.
    """
    signal_sum = 0
    for i in range(len(x)):
        signal_sum += abs(x[i]) + abs(y[i]) + abs(z[i])
    return signal_sum


def correlation(x, y, mx, my):
    """ Two dimensions of data are input together with their means. This function
    calculates the correlation between the two data dimensions.
    """
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for i in range(len(x)):
        sum1 += (x[i] - mx) * (y[i] - my)
        sum2 += (x[i] - mx) ** 2
        sum3 += (y[i] - my) ** 2
    sum4 = sum2 * sum3
    if sum4 == 0:
        return 0.0
    sum4 = math.sqrt(sum4)
    return sum1 / sum4


def autocorrelation(x, mean):
    """ Calculate autocorrelation, or correlation of a data sequence with
    itself at different points in the sequence.
    """
    sum1 = 0
    sum2 = 0
    for i in range(1, len(x)):
        sum1 += (x[i - 1] - mean) * (x[i] - mean)
        sum2 += (x[i] - mean) ** 2
    sum1 /= (len(x) - 1)
    sum2 /= len(x)
    if sum2 == 0:
        return 0.0
    return sum1 / sum2


def heading_change_rate(course, distance):
    """ The input is sequences of course values and distance values
    (usually provided by location services). This function calculates the number
    of points in the sequence that indicate a change in direction, divided by
    the distance covered during the sequence.
    """
    if distance == 0:  # Avoid divide by zero
        return 0.0
    total = 0
    prevc = course[0]
    for c in course:
        if (prevc != c) and (c != -1.0):
            total += 1
            prevc = c
    return float(total) / distance


def stop_rate(latitude, longitude, distance):
    """ The input is sequences of latitude, longitude, and distance values
    (usually provided by location services). This function calculates the number
    of points in the sequence that have no change in location, divided by
    the distance covered during the sequence.
    """
    if distance == 0:  # Avoid divide by zero
        return 0.0
    total = 0
    prevlat = latitude[0]
    prevlong = longitude[0]
    for i in range(len(latitude)):
        if (prevlat == latitude[i]) and (prevlong == longitude[i]):
            total += 1
        prevlat = latitude[i]
        prevlong = longitude[i]
    return float(total) / distance


def trajectory(latitude, longitude):
    """ Calculate the overall trajectory indicated by the input sequences of
    latitude and longitude values.
    """
    minl = min(latitude)
    maxl = max(latitude)
    difflat = maxl - minl
    if difflat == 0.0:  # Avoid divide by zero
        return -1.57079633
    minl = min(longitude)
    maxl = max(longitude)
    difflong = maxl - minl
    slopepercent = difflong / difflat
    return math.atan(slopepercent)

import tsfresh.feature_extraction.feature_calculators as tcalc
from tsfresh.feature_extraction.settings import MinimalFCParameters
import scipy.signal as signal
import matplotlib.pyplot as plt
def feats(x) :
    if x.shape[0] == 0 :
        print('stop')
        return
    last = x[0]
    same = True
    for cur in x :
        if last != cur :
            same = False
    if same :
        x[0] = x[0] + 0.005
    
    window = signal.general_gaussian(51, p=3, sig=5)
    filtered = signal.fftconvolve(window, x)
    filtered = (np.average(x) / np.average(filtered)) * filtered
    filtered = np.roll(filtered, -25)[:x.shape[0]]

    param = {}
    param["attr"] = "rvalue"
    param["chunk_len"] = 5
    param["f_agg"] = "mean"
    param["maxlag"] = 5
    param["coeff"] = 3
    param["m"] = 5
    param["r"] = 3
    param["k"] = 5
    param = [param]
    feature_list = []
    feature1 = max(x)
    feature2 = min(x)
    feature3 = sum(x)
    feature4 = feature3 / x.shape[0]
    feature5 = np.var(x)
    feature6 = np.std(x)
    feature7 = tcalc.abs_energy(x)
    feature8 = tcalc.absolute_sum_of_changes(x)

    feature9 = tcalc.agg_autocorrelation(x, param)[0][1]
    feature10 = tcalc.agg_linear_trend(x, param)
    f = []
    for feature in feature10 :
        f.append(feature)
    #feature11 = tcalc.approximate_entropy(x, 5, 3) ###
    feature12 = 0
    try :
        feature12 = tcalc.ar_coefficient(x, param)[0][1]
    except:
        feature12 = 0
    param[0]["attr"] = "pvalue"
    param[0]["autolag"] = None
    feature13 = tcalc.augmented_dickey_fuller(x, param)[0][1]
    feature14 = tcalc.autocorrelation(x, 5)
    feature15 = tcalc.benford_correlation(x)
    feature16 = tcalc.binned_entropy(x, 4)
    feature17 = tcalc.c3(x, 5)
    feature18 = tcalc.cid_ce(x, normalize=False)
    feature19 = tcalc.count_above_mean(x)
    feature20 = tcalc.count_below_mean(x)
    feature21 = tcalc.fourier_entropy(x, 4)
    #feature22 = tcalc.friedrich_coefficients(x,param)[0][1]
    param[0]["aggtype"] = "variance"
    param[0]["attr"] = "real"
    feature23 = tcalc.fft_aggregated(x, param)
    f1 = []
    for feature in feature23 :
        f1.append(feature)
    feature24 = tcalc.fft_coefficient(x, param)
    f2 = []
    for feature in feature24 :
        f2.append(feature)
    param[0]["num_segments"] = 4
    param[0]["segment_focus"] = 2
    feature26 = tcalc.energy_ratio_by_chunks(x, param)[0][1]
    feature27 = tcalc.first_location_of_maximum(x)
    feature28 = tcalc.first_location_of_minimum(x)
    param[0]["q"] = 0.5
    feature29 = tcalc.index_mass_quantile(x, param)[0][1]
    feature30 = tcalc.kurtosis(x)
    feature31 = float(tcalc.large_standard_deviation(x, 0.25))
    feature32 = tcalc.lempel_ziv_complexity(x, 10)
    param[0]["attr"] = "rvalue"
    feature33 = tcalc.linear_trend(x, param)[0][1]
    feature34 = tcalc.longest_strike_above_mean(x)
    feature35 = tcalc.longest_strike_below_mean(x)
   # feature36 = tcalc.max_langevin_fixed_point(x, 3, 0.25) ##
    feature37 = tcalc.mean_change(x)
    feature38 = tcalc.mean_n_absolute_max(x, 10)
    feature39 = tcalc.mean_second_derivative_central(x)
    feature40 = tcalc.number_crossing_m(x,0.5) 
    feature41 = tcalc.number_cwt_peaks(x, 8)
    feature42 = tcalc.number_peaks(x, 4)
    param[0]["lag"] = 5
    feature43 = tcalc.partial_autocorrelation(x, param)[0][1]
    feature44 = tcalc.quantile(x, 0.25)
    feature45 = feature1 - feature2
    feature46 = tcalc.quantile(x, 0.5)
    feature47 = tcalc.quantile(x, 0.75)
    feature48 = tcalc.ratio_beyond_r_sigma(x, 0.25)
    feature49 = tcalc.root_mean_square(x)
    #feature50 = tcalc.sample_entropy(x) ##
    feature51 = tcalc.skewness(x)
    feature52 = tcalc.spkt_welch_density(x, param)
    f3 = []
    for feature in feature52 :
        f3.append(feature)
    param[0]["r"] = 0.5
    feature53 = float(tcalc.symmetry_looking(x, param)[0][1])
    feature54 = tcalc.time_reversal_asymmetry_statistic(x, 5)
    feature55 = tcalc.variation_coefficient(x)
    feature56 = tcalc.sum_of_reoccurring_data_points(x)
    ceps, entropy, energy, msignal, vsignal = fft_features(x)
    se = signal_energy(x)  # SMA
    
    lse = log_signal_energy(x)  # log SMA
    
    p = se / len(x)  # power
    iqr = interquartile_range(x)  # interquartile range

    feature_list.append(feature1)
    feature_list.append(feature2)
    feature_list.append(feature3)
    feature_list.append(feature4)
    feature_list.append(feature5)
    feature_list.append(feature6)
    feature_list.append(feature7)
    feature_list.append(feature8)
    feature_list.append(feature9)
    feature_list.append(f[0][1])
    #feature_list.append(feature11)
    feature_list.append(feature12)
    feature_list.append(feature13)
    feature_list.append(feature14)
    feature_list.append(feature15)
    feature_list.append(feature16)
    feature_list.append(feature17)
    feature_list.append(feature18)
    feature_list.append(feature19)
    feature_list.append(feature20)
    feature_list.append(feature21)
    #feature_list.append(feature22)
    feature_list.append(f1[0][1])
    feature_list.append(f2[0][1])
    #feature_list.append(feature25)
    feature_list.append(feature26)
    feature_list.append(feature27)
    feature_list.append(feature28)
    feature_list.append(feature29)
    feature_list.append(feature30)
    feature_list.append(feature31)
    feature_list.append(feature32)
    feature_list.append(feature33)
    feature_list.append(feature34)
    feature_list.append(feature35)
    #feature_list.append(feature36)
    feature_list.append(feature37)
    feature_list.append(feature38)
    feature_list.append(feature39)
    feature_list.append(feature40)
    feature_list.append(feature41)
    feature_list.append(feature42)
    feature_list.append(feature43)
    feature_list.append(feature44)
    feature_list.append(feature45)
    feature_list.append(feature46)
    feature_list.append(feature47)
    feature_list.append(feature48)
    feature_list.append(feature49)
    #feature_list.append(feature50)
    feature_list.append(feature51)
    feature_list.append(f3[0][1])
    feature_list.append(feature53)
    feature_list.append(feature54)
    feature_list.append(feature55)
    feature_list.append(feature56)
    feature_list.append(ceps)
    feature_list.append(entropy)
    feature_list.append(energy)
    feature_list.append(msignal)
    feature_list.append(lse)
    feature_list.append(iqr)

    return np.nan_to_num(feature_list).tolist()


def generate_statistical_features(x):
    """ Create a list of statistical features for a sequence of values
    corresponding to one type of sensor (e.g., acceleration, rotation, location).
    """
    feature_list = []

    if len(x) == 0:
        return [0] * 30
    feature = max(x)  # max
    feature_list.append(feature)
    feature = min(x)  # min
    feature_list.append(feature)
    feature = 0  # sum
    for i in x:
        if i != 'None':
            feature += i
    feature_list.append(feature)
    mean = feature / len(x)  # mean
    feature_list.append(mean)
    median = np.median(x)  # median
    feature_list.append(median)
    mav1 = np.median(list(map(abs, x)))  # mean absolute value
    feature_list.append(mav1)
    mav2 = np.median(list(map(abs, x)))  # median absolute value
    feature_list.append(mav2)
    variance = np.var(x)
    feature_list.append(variance)  # variance
    std = np.std(x)
    feature_list.append(std)  # standard deviation
    m1 = mean_absolute_deviation(x)  # mad1
    feature_list.append(m1)
    m2 = median_absolute_deviation(x)  # mad2
    feature_list.append(m2)
    zc = zero_crossings(x, median)  # zero crossings
    feature_list.append(zc)
    mc = mean_crossings(x, mean)  # mean crossings
    feature_list.append(mc)
    m1, m2, m3, m4 = moments(x)
    feature_list.append(m1)
    feature_list.append(m2)
    feature_list.append(m3)
    feature_list.append(m4)
    ceps, entropy, energy, msignal, vsignal = fft_features(x)
    feature_list.append(ceps)
    feature_list.append(entropy)
    feature_list.append(energy)
    feature_list.append(msignal)
    feature_list.append(vsignal)
    iqr = interquartile_range(x)  # interquartile range
    feature_list.append(iqr)
    if mean == 0:  # cv
        coefficient_of_variation = 0
    else:
        coefficient_of_variation = std / mean
    feature_list.append(coefficient_of_variation)
    skew = skewness(x, mean)
    feature_list.append(skew)  # skewness
    k = kurtosis(x, mean, std)  # kurtosis
    feature_list.append(k)
    se = signal_energy(x)  # SMA
    feature_list.append(se)
    lse = log_signal_energy(x)  # log SMA
    feature_list.append(lse)
    p = se / len(x)  # power
    feature_list.append(p)
    if len(x) > 1:  # autocorrelation
        ac = autocorrelation(x, mean)
    else:
        ac = 0
    feature_list.append(ac)
    return feature_list

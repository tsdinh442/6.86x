import project1 as p1
import utils
import random
import numpy as np

def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices

def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    z = label * np.dot(feature_vector, current_theta) + current_theta_0
    if z <= 1:
        return (1 - eta * L) * current_theta + eta * label * feature_vector, current_theta_0 + eta * label
    else:
        return (1 - eta * L) * current_theta, current_theta_0

    mult = 1 - eta * L
    if label * (np.dot(feature_vector, current_theta) + current_theta_0) <= 1:
        return (mult * current_theta) + (eta * label * feature_vector), current_theta_0 + (eta * label)
    return mult * current_theta, current_theta_0



def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    nsamples, nfeatures = feature_matrix.shape
    theta = np.zeros(nfeatures)
    theta_0 = 0
    count = 0
    for t in range(T):
        for i in get_order(nsamples):
            count += 1
            eta = 1.0 / np.sqrt(count)
            theta, theta_0 = pegasos_single_step_update(
                feature_matrix[i], labels[i], L, eta, theta, theta_0)
    return theta, theta_0

feature_vector = np.array([ 0.43266691, -0.30978726,  0.09753205,  0.24664983,  0.30397977,  0.30171255, -0.49425932,  0.05225855,  0.49198953,  0.19763639])
current_theta = np.array([ 0.02632698,  0.25972592, -0.30581519,  0.33575905, -0.05419816, -0.13310677, -0.06818656,  0.08042317, 0.2108675, 0.46193152])
current_theta_0 = 1.5502704486358234
label= 1
L= 0.8904689804521747
eta= 0.9395529245911732


feature_vector= np.array( [ 0.48307418, 0.04571974, 0.17169688, -0.46655922,  0.0847632,  -0.22578859, -0.23158738,  0.27373153,  0.04772714,  0.30584908])
label = 1
L = 0.40208780341235417
eta = 0.19655858198436083
current_theta = np.array( [-0.27726426,  0.24357834,  0.09891236,  0.12497626, -0.18094852, -0.02074998, -0.27669228,  0.0077205,  -0.1498352,  -0.43953108])
current_theta_0 = 1.2501710099249492


feature_vector = np.array( [ 0.2584929,   0.26495836,  0.49391554,  0.24479528,  0.15433678, -0.18628872,
  0.11731756,  0.42382695,  0.11403971, -0.42921903])
label = -1
L = 0.7096543280388292
eta = 0.4103165377312913
current_theta = np.array( [-0.28947436,  0.00194721,  0.31514556,  0.08235029,  0.4021951,  -0.3299868,
 -0.35892938, -0.05870122, -0.33360154,  0.09979787])
current_theta_0 = -0.5217362339013706

feature_matrix = np.array( [[ 0.1837462,   0.29989789, -0.35889786, -0.30780561, -0.44230703, -0.03043835, 0.21370063,  0.33344998 ,-0.40850817, -0.13105809],
                            [ 0.08254096,  0.06012654,  0.19821234,  0.40958367,  0.07155838, -0.49830717, 0.09098162,  0.19062183, -0.27312663,  0.39060785],
                            [-0.20112519, -0.00593087,  0.05738862,  0.16811148, -0.10466314, -0.21348009, 0.45806193, -0.27659307,  0.2901038,  -0.29736505],
                            [-0.14703536, -0.45573697, -0.47563745, -0.08546162, -0.08562345,  0.07636098, -0.42087389, -0.16322197, -0.02759763,  0.0297091 ],
                            [-0.18082261,  0.28644149, -0.47549449, -0.3049562,   0.13967768,  0.34904474, 0.20627692,  0.28407868,  0.21849356, -0.01642202]])
labels = np.array([-1, -1, -1,  1, -1])
T = 10
L = 0.1456692551041303


feature_matrix = np.array( [[ 0.32453673,  0.06082212,  0.27845097,  0.27124962, -0.48858134],
                            [-0.07490036, -0.2226942,   0.46808161, -0.15484728, -0.06555043],
                            [ 0.48089473,  0.11053774, -0.39253255, -0.45844357,  0.19818921],
                            [ 0.39728286,  0.14426349,  0.23446484, -0.46963688,  0.30978055],
                            [-0.2836313,  0.20048277,  0.10600686, -0.47812081,  0.24772569],
                            [-0.38813183, -0.39082381,  0.02482903,  0.46576666, -0.22720277],
                            [ 0.15482689, -0.16083218,  0.38637948, -0.14209394,  0.05076824],
                            [-0.1238048,  -0.1064888,  -0.28800396, -0.47983335,  0.31652173],
                            [ 0.31485345,  0.30679047, -0.1907081,  -0.0961867,   0.27954887],
                            [ 0.4024408,   0.2990748,   0.34148516, -0.311256,    0.13324454]])
labels = np.array( [-1, -1,  1,  1,  1,  1, -1, -1,  1,  1])
T = 10
L = 0.705513226934028

print(pegasos(feature_matrix, labels, T, L))

['-0.0451377', '0.0892342', '-0.0633491', '-0.0615329', '0.1293817']
import numpy as np
from scipy.interpolate import CubicSpline
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def DTW(ori, padded):
    """
    Dynamic Time Warping to calculate the similarity
    :return:
    """
    distance, _ = fastdtw(ori, padded, dist=euclidean)
    return distance


def Jittering(data: np.ndarray, mean=0.0, std=0.03):
    """
    :param data:    train data in shape [frames, channels]
    :return:
        add random noise from a Gaussian distribution with a mean 0 and std 0.03
        to the original data
    """
    assert len(data.shape) == 2, "data shape does not fit this function"
    noise = np.random.normal(mean, std, data.shape)
    new_data = data + noise
    return new_data


def Scaling(data: np.ndarray, mean=1.0, std=0.1):
    """
    :param data:    train data in shape [frames, channels]
    :return:
    """
    scalingFactor = np.random.normal(loc=mean, scale=std, size=(1, data.shape[1]))
    myNoise = np.matmul(np.ones((data.shape[0], 1)), scalingFactor)
    new_data = data * myNoise
    return new_data


def GenerateRandomCurves(data: np.ndarray, sigma=0.2, knot=4):
    """
    :param data:
    :param sigma:
    :param knot:
    :return:
        sample the knots on temporal dimension
        generate random noise whose dimension is as same as feature dimension for each knot
        cubic spline function for each channel of the feature
        use this function to interpolate the whole time series

        this can be seen as smooth noise
    """
    xx = (np.ones((data.shape[1], 1)) * (np.arange(0, data.shape[0], (data.shape[0] - 1) / (knot + 1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, data.shape[1]))
    x_range = np.arange(data.shape[0])
    out = None
    for i in range(data.shape[1]):
        cs = CubicSpline(xx[:, i], yy[:, i])
        sub_out = cs(x_range)
        out = sub_out if out is None else np.vstack([out, sub_out])

    return out.transpose()


def MagWarp(data: np.ndarray, sigma=0.2):
    new_data = data * GenerateRandomCurves(data, sigma)
    return new_data


def DistortTimesteps(data: np.ndarray, sigma=0.2):
    """
    :param data:
    :param sigma:
    :return:
        return a array with shape [frames, channels] which is same as data.
        for each column of the returned value, you can treat it as a set of
        new time points.
    """
    tt = GenerateRandomCurves(data, sigma)
    """
    'tt' is actually a smooth curve consisted of random noise
    with shape [frames, channels]
    """
    tt_cum = np.cumsum(tt, axis=0)
    """
    column sum,
    then we will normalize this to generate a new set of time points
    """
    # Make the last value to have X.shape[0]
    for i in range(data.shape[1]):
        t_scale = (data.shape[0] - 1) / tt_cum[-1, i]
        tt_cum[:, i] = tt_cum[:, i] * t_scale

    return tt_cum


def TimeWarp(data: np.ndarray, sigma=0.2):
    tt_new = DistortTimesteps(data, sigma)
    data_new = np.zeros(data.shape)
    x_range = np.arange(data.shape[0])
    for i in range(data.shape[1]):
        """
        since we get a new set of time points, use naive
        interpolation method to generate new data
        whose time starts from 0 and ends with data.shape[0]
        """
        data_new[:, i] = np.interp(x_range, tt_new[:, i], data[:, i])

    return data_new


# do not use this first
def Permutation(X, nPerm=4, minSegLength=10):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile:
        segs = np.zeros(nPerm + 1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0] - minSegLength, nPerm - 1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:] - segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii] + 1], :]
        X_new[pp:pp + len(x_temp), :] = x_temp
        pp += len(x_temp)

    return X_new


def Combanition(data):
    # 1. time_warping + Magnitude_Warping
    new_d1 = MagWarp(TimeWarp(data))
    new_d4 = TimeWarp(MagWarp(data))
    # 2. time_warping + Jittring
    new_d2 = Jittering(TimeWarp(data))
    new_d5 = TimeWarp(Jittering(data))
    # 3. time_warping + scaling
    new_d3 = Scaling(TimeWarp(data))
    new_d6 = TimeWarp(Scaling(data))

    return [new_d1, new_d2, new_d3, new_d4, new_d5, new_d6]


def augment(data: np.ndarray):
    """
    :param data:
                train data in shape [channels, frames, joints(segments)]
    :return:
                augmented data with the same shape of input
    """
    out = list()
    channels, frames, joints = data.shape
    n_data = np.reshape(data, (frames, joints, channels))
    n_data = np.reshape(n_data, (frames, -1))

    jit_aug1 = Jittering(data=n_data, mean=0.0, std=0.03)
    jit_aug2 = Jittering(data=n_data, mean=0.1, std=0.05)
    jit_aug3 = Jittering(data=n_data, mean=-0.1, std=0.05)
    sca_aug1 = Scaling(data=n_data, mean=1.0, std=0.1)
    sca_aug2 = Scaling(data=n_data, mean=0.98, std=0.12)
    sca_aug3 = Scaling(data=n_data, mean=1.02, std=0.08)

    jit_aug1 = np.reshape(jit_aug1, (frames, joints, channels))
    jit_aug1 = np.reshape(jit_aug1, (channels, frames, joints))

    jit_aug2 = np.reshape(jit_aug2, (frames, joints, channels))
    jit_aug2 = np.reshape(jit_aug2, (channels, frames, joints))

    jit_aug3 = np.reshape(jit_aug3, (frames, joints, channels))
    jit_aug3 = np.reshape(jit_aug3, (channels, frames, joints))

    sca_aug1 = np.reshape(sca_aug1, (frames, joints, channels))
    sca_aug1 = np.reshape(sca_aug1, (channels, frames, joints))

    sca_aug2 = np.reshape(sca_aug2, (frames, joints, channels))
    sca_aug2 = np.reshape(sca_aug2, (channels, frames, joints))

    sca_aug3 = np.reshape(sca_aug3, (frames, joints, channels))
    sca_aug3 = np.reshape(sca_aug3, (channels, frames, joints))

    out = list([jit_aug1, jit_aug2, jit_aug3, sca_aug1, sca_aug2, sca_aug3])

    return out

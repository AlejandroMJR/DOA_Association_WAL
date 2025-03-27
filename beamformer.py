import numpy as np
import scipy.signal as sp

def get_delays(micPos, thetas, c= 343):
    """
    Compute the delays for each microphone based on the desired steering angles.

    Args:
    - micPos: Array of microphone positions (shape: [n_mics, 2]).
    - thetas: Array of desired steering angles (radians).
    - c: Speed of sound in air (m/s).

    Returns:
    - delays: Array of delays (seconds) for each microphone and each direction.
    """
    if micPos.shape[1] == 3:
        micPos = micPos[:, :2]
    micPos = micPos - np.mean(micPos, axis=0)  # Center the microphone positions
    dir_vec = np.array([np.cos(thetas), np.sin(thetas)])
    delays = np.dot(micPos, dir_vec) / c
    return -delays.T

def mpdr_beamformer(signal, f, steeringDelays, returnWeights=False, regularization=1e-7):
    """
    Implements the Minimum Variance Distortionless Response (MVDR) beamformer.

    Args:
    - signal: Matrix of microphone signals (shape: [n_mics, n_samples]).
    - f: Carrier frequency of the signal (Hz).
    - steering_delays: Array of steering delays (seconds).
    - return_weights: If True, return the weights instead of the beamformed signal.
    - regularization: Regularization parameter to improve numerical stability.

    Returns:
    - w: MVDR beamformed signal, if return_weights is False. Otherwise, return the weights.
    or
    - beamformed signal
    """
    n_mics = signal.shape[0]
    # Steering vector in the desired direction
    s = np.exp(-2j * np.pi * steeringDelays * f) / np.sqrt(n_mics)

    # Compute the covariance matrix
    R = (signal @ signal.conj().T) / signal.shape[1]
    R += regularization * np.eye(n_mics)  # Add regularization to improve stability
    R_inv = np.linalg.inv(R)  # Inverse of the covariance matrix

    # Compute MVDR weights
    w = np.array([R_inv.dot(s[i]) / (s[i].conj().dot(R_inv).dot(s[i])) for i in range(s.shape[0])])
    if returnWeights:
        return w
    else:
        return w.conj().dot(signal)

def stft_beamformer(signal, fs, nfft, sd, overlap=0, window="boxcar", return_signal=True, average = True):
    """
    Apply a beamformer to broadband signals using the Short-Time Fourier Transform (STFT).

    Args:
    - signal: Matrix of broadband signals (shape: [n_mics, n_samples]).
    - fs: Sampling frequency (Hz).
    - nfft: Number of FFT points.
    - sd: Array of steering delays (seconds) for each direction.
    - overlap: Number of samples to overlap between adjacent frames.
    - window: Type of window to apply to each frame.
    - return_signal: If True, return the beamformed signal in the time domain.
    - average: If True, average the beamformed signal across all time bins.

    Returns:
    - bfo: Beamformed output signal (1D array).
    or
    - beamformed output power either averaged across time bins or not
    """
    if nfft/fs < (np.max(sd)-np.min(sd)):
        raise ValueError('nfft too small for this array')

    _, _, x = sp.stft(signal, nperseg=nfft, noverlap=overlap, window=window)
    bfo = np.zeros((sd.shape[0], x.shape[1], x.shape[2]), dtype=np.complex128)
    for i in range(x.shape[1]):
        f = i*float(fs)/nfft
        bfo[:,i,:] = mpdr_beamformer(x[:,i,:], f, sd)

    if return_signal:
        # Perform inverse STFT to get the time-domain signal
        bfo_time = np.zeros((sd.shape[0], signal.shape[1]), dtype=np.float64)
        for i in range(sd.shape[0]):
            _, bfo_t = sp.istft(bfo[i], nfft=nfft, noverlap=overlap, window=window)
            bfo_time[i] = bfo_t[:signal.shape[1]]
        # normalize signal from -1 to 1
        bfo_time /= np.max(np.abs(bfo_time))
        return bfo_time
    else:
        if average:
            return (np.abs(bfo) ** 2).sum(axis=-2).sum(axis=-1)
        else:
            return (np.abs(bfo) ** 2).sum(axis=-2)




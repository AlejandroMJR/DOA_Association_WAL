from beamformer import stft_beamformer, get_delays
from utils import get_embeddings_from_buffer

def broadband_beamform(micPos, signals, fs, nfft, anglesRad, overlap):
    """
    Apply broadband beamforming to each estimated DOA direction at each node

    Args:
    - micPos: Array of microphone positions (shape: [nNodes, nMics, 2]).
    - signals: Array of microphone signals (shape: [nNodes, nMics, nSamples]).
    - fs: Sampling frequency (Hz).
    - nfft: Number of FFT points.
    - anglesRad: Dictionary of estimated DOA angles (radians) per node (shape: [nNodes, nSources]).
    - overlap: Number of overlapping samples.
    """

    # Apply beamforming to each node
    nNodes = len(micPos)
    y = []
    for i in range(nNodes):
        steeringDelays = get_delays(micPos[i], anglesRad[i])
        y.append(stft_beamformer(signals[i], fs, nfft, steeringDelays, overlap=overlap, window="hann"))
    return y

def extract_embeddings(y, nNodes, fs):
    """
    Extract embeddings from the beamformed signals

    Args:
    - y: Array of beamformed signals (shape: [nNodes, nSources, nSamples]).
    - nNodes: Number of nodes.
    - fs: Sampling frequency (Hz).

    Returns:
    - embeddings: List of embeddings extracted from the beamformed signals.
    """
    embeddings = []
    for i in range(nNodes):
        for j in range(y[i].__len__()):
            embeddings.append(get_embeddings_from_buffer(y[i][j], fs))
            #sf.write(f"audios/beamformed/{i + 1}array_to{j + 1}source.wav", y[i][j], fs, format='wav')
    return embeddings
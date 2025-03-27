import numpy as np
import pyroomacoustics as pra
import librosa
import soundfile as sf
from utils import clear_folder

def _simulate(nNodes=3, micsPerNode=4, micSpacing=0.05, nSources=5, environmentSize=30, noisePower = 50, sourcePower= 80, arrayType="circular", saveSignals=False):
    """
    Simulates audio signals captured by a microphone array from multiple sound sourcePool in a free-field environment.

    Args:
        nNodes (int): Number of microphone nodes.
        micsPerNode (int): Number of microphones per node.
        micSpacing (float): Spacing between microphones in meters.
        nSources (int): Number of active sourcePool.
        environmentSize (float): Dimensions of the simulated environment.
        noisePower (float): Noise power in dB SPL.
        sourcePower (float): Source power in dB SPL.
        arrayType (str): Type of microphone array. Choose from 'linear', 'circular', or 'l'.

    Returns:
        fs (int): Sampling frequency of the audio signals.
        micPos (numpy.ndarray): Positions of the microphones [nNodes, micsPerNode, 2].
        signals_noise (numpy.ndarray): Simulated microphone signals with added noise [nNodes, micsPerNode, nSamples].
        srcPos (numpy.ndarray): Positions of the sound sourcePool [nSources, 2].
    """
    if nSources > 8:
        raise ValueError("The maximum number of sourcePool is 8")
    if saveSignals:
        clear_folder("audios/micSignals")
        clear_folder("audios/beamformed")

    # Get simulation setup
    micPos, srcPos = simulation_setup(nNodes, micsPerNode, micSpacing, nSources, environmentSize, arrayType = arrayType)

    # Load bird audio files and ensure consistent sampling rate
    source1, fs = librosa.load("audios/sourcePool/1source.wav", sr=None)
    source2, _ = librosa.load("audios/sourcePool/2source.wav", sr=fs)
    source3, _ = librosa.load("audios/sourcePool/3source.wav", sr=fs)
    source4, _ = librosa.load("audios/sourcePool/4source.wav", sr=fs)
    source5, _ = librosa.load("audios/sourcePool/5source.wav", sr=fs)
    source6, _ = librosa.load("audios/sourcePool/6source.wav", sr=fs)
    source7, _ = librosa.load("audios/sourcePool/7source.wav", sr=fs)
    source8, _ = librosa.load("audios/sourcePool/8source.wav", sr=fs)

    # List of all audio signals
    allSources = np.array([source1, source2, source3, source4, source5, source6, source7, source8])

    # Randomly sample nSources from all sourcePool
    selectedSources = np.random.choice(len(allSources), nSources, replace=False)
    sources = allSources[selectedSources]

    # Normalize source audio signals
    for i, source in enumerate(sources):
        if source.ndim == 2:  # Handle stereo audio
            source = source[:, 0]
        sources[i] = source / np.max(np.abs(source))
        if saveSignals:
            sf.write(f"audios/beamformed/{i + 1}source.wav", sources[i], fs, format='wav')
    sources = np.array(sources)

    # Normalize source power
    pRef = 20e-6  # Reference pressure in Pascals
    # Compute Scaling Factor
    pTarget = pRef * 10 ** (sourcePower / 20) # Convert SPL to actual pressure

    # Compute the RMS of the normalized signal
    pRmsNorm = [np.sqrt(np.mean(sources[i][sources[i]>0.3] ** 2)) for i in range(nSources)]
    pRmsNorm = np.array(pRmsNorm)
    # Compute the correct scaling factor
    scale = pTarget / pRmsNorm
    scale = scale[:, np.newaxis]

    sources = scale * sources  # Scale the source signals

    # Create free field simulation
    environment = pra.AnechoicRoom(dim=2, fs=fs)

    # Add sourcePool to the environment
    for i, source in enumerate(sources):
        environment.add_source(srcPos[i], signal=source)

    # Add microphone array to the room
    environment.add_microphone_array(
        pra.MicrophoneArray(micPos.T, environment.fs)
    )

    # Simulate the microphone signals
    environment.simulate()
    signals = environment.mic_array.signals

    # Keep signals to 3s duration
    dur = 3  # Duration in seconds
    durSamples = dur * fs
    signals = signals[:, :durSamples]
    signals = signals.reshape(nNodes, micsPerNode, -1)

    # Add Gaussian uncorrelated noise to the simulated signals
    pNoise = pRef * 10 ** (noisePower / 20)  # Convert dB SPL to pressure
    noise = np.random.normal(0, pNoise, signals.shape) # Generate noise
    signals = signals + noise

    # Normalize noisy signals
    for i in range(nNodes):
        for j in range(micsPerNode):
            signals[i, j] = signals[i, j] / np.max(np.abs(signals[i, j]))
            if saveSignals:
                sf.write(f"audios/micSignals/{i + 1}node_{j + 1}micSignal.wav", signals[i, j], fs, format='wav')

    return fs, micPos.reshape(nNodes,micsPerNode,-1), signals, srcPos


def simulation_setup(nNodes, micsPerNode, micSpacing, nSources, environmentSize, arrayType):
    """
    Generates the positions of the microphones and sound sourcePool in the simulated environment.

    Args:
        nNodes (int): Number of microphone nodes.
        micsPerNode (int): Number of microphones per node.
        micSpacing (float): Spacing between microphones in meters.
        nSources (int): Number of sound sourcePool.
        environmentSize (float): Dimensions of the simulated environment.
        arrayType (str): Type of microphone array. Choose from 'linear', 'circular', or 'l'.

    Returns:
        micPos (numpy.ndarray): Positions of the microphones [nNodes * micsPerNode, 2].
        srcPos (numpy.ndarray): Positions of the sound sourcePool [nSources, 2].
    """
    # Generate default node positions in a grid pattern in the room using nNodes and room_size
    nodeCenterPos = []
    gridSize = int(np.ceil(np.sqrt(nNodes)))  # Determine the grid size
    spacing = environmentSize / (gridSize + 1)  # Calculate spacing based on room size and grid size

    for i in range(nNodes):
        x_pos = spacing * (i % gridSize + 1)
        y_pos = spacing * (i // gridSize + 1)
        nodeCenterPos.append([x_pos, y_pos])
    nodeCenterPos = np.array(nodeCenterPos)

    #nodeCenterPos = np.array([[10,10],[20,20]])
    # Generate microphone positions based on the array type
    if arrayType == "linear":
        micPos = generate_linear_array(nNodes, nodeCenterPos, micsPerNode, micSpacing)
    elif arrayType == "circular":
        radius = micSpacing / (2 * np.sin(np.pi / micsPerNode))
        #radius = micSpacing
        micPos = generate_circular_array(nNodes, nodeCenterPos, micsPerNode, radius)
    elif arrayType == "l":
        micPos = generate_l_array(nNodes, nodeCenterPos, micsPerNode, micSpacing)
    else:
        raise ValueError("Invalid array type. Choose from 'linear', 'circular', or 'l'.")

    # Generate source positions in the room
    srcPos = np.random.rand(nSources, 2) * (environmentSize) # Randomly distribute sourcePool in the room
    return micPos, srcPos

def generate_linear_array(nNodes, nodeCenterPos, micsPerNode, micSpacing):
    """
    Generates a linear array of microphones for each node.
    
    Args:
        nNodes (int): Number of microphone nodes.
        nodeCenterPos (numpy.ndarray): Center positions of the microphone nodes [nNodes, 2].
        micsPerNode (int): Number of microphones per node.
        micSpacing (float): Spacing between microphones in meters.
        
    Returns:
        micPos (numpy.ndarray): Positions of the microphones [nNodes * micsPerNode, 2].
    """
    micPos = []
    #Generate uniform linear array of microphones for each node
    for node in range(nNodes):
        base_x, base_y = nodeCenterPos[node % len(nodeCenterPos)]  # Cycle through base positions
        for mic in range(micsPerNode):  # Configurable number of microphones per node
            micPos.append([base_x + mic * micSpacing, base_y])

    return np.array(micPos)

def generate_circular_array(nNodes, nodeCenterPos, micsPerNode, micSpacing):
    """
    Generates a circular array of microphones for each node.

    Args:
        nNodes (int): Number of microphone nodes.
        nodeCenterPos (numpy.ndarray): Center positions of the microphone nodes [nNodes, 2].
        micsPerNode (int): Number of microphones per node.
        micSpacing (float): Spacing between microphones in meters.

    Returns:
        micPos (numpy.ndarray): Positions of the microphones [nNodes * micsPerNode
    """
    micPos = []
    # Generate circular array of microphones for each node
    for node in range(nNodes):
        base_x, base_y = nodeCenterPos[node % len(nodeCenterPos)]  # Cycle through base positions
        for mic in range(micsPerNode):  # Configurable number of microphones per node
            angle = 2 * np.pi * mic / micsPerNode
            micPos.append([base_x + micSpacing * np.cos(angle), base_y + micSpacing * np.sin(angle)])

    return np.array(micPos)

def generate_l_array(nNodes, nodeCenterPos, micsPerNode, micSpacing):
    """
    Generates an L-shaped array of microphones for each node.

    Args:
        nNodes (int): Number of microphone nodes.
        nodeCenterPos (numpy.ndarray): Center positions of the microphone nodes [nNodes, 2].
        micsPerNode (int): Number of microphones per node.
        micSpacing (float): Spacing between microphones in meters.

    Returns:
        micPos (numpy.ndarray): Positions of the microphones [nNodes * micsPerNode, 2].
    """
    micPos = []
    # Generate L shaped array of microphones for each node
    for node in range(nNodes):
        base_x, base_y = nodeCenterPos[node % len(nodeCenterPos)]
        half_mics = micsPerNode // 2
        if micsPerNode % 2 == 1:
            half_mics += 1
        for mic in range(half_mics):
            micPos.append([base_x + (mic) * micSpacing, base_y])
        for mic in range(half_mics, micsPerNode):
            micPos.append([base_x, base_y + (mic - half_mics + 1) * micSpacing])
    return np.array(micPos)

import numpy as np
from birdnetlib import RecordingBuffer
from birdnetlib.analyzer import Analyzer
from itertools import product
import os
import glob
import matplotlib.pyplot as plt


def normalized_cosine_similarity(x, y):
    """
    Compute the cosine similarity between two vectors and normalize it to the range [0, 1].
    Args:
    - x: 1D numpy array.
    - y: 1D numpy array.
    Returns:
    - cosine_similarity: Cosine similarity between x and y normalized to the range [0, 1].
    """
    dotProduct = np.dot(x, y)
    normX = np.linalg.norm(x)
    normY = np.linalg.norm(y)
    cosine_similarity = (dotProduct / (normX * normY) + 1) / 2
    return cosine_similarity

def fill_sparse_dict(nodes, alpha):
    """
    Fills the similarity dictionary
    Args:
    - nodes: List of lists containing the feature vectors of each node.
    - alpha: Weight for the penalization of dummies.

    Returns:
    - similarities: Dictionary containing the similarity values for each possible association.
    """
    nodes = add_dummy(nodes)
    nNodes = len(nodes)
    similarities = {}
    indices = list(product(*[range(len(nodes[i])) for i in range(len(nodes))]))


    alpha = alpha/(nNodes-1) # Normalize alpha to the number of nodes
    w_n = np.array([1 - i * alpha for i in range(nNodes - 1, -1, -1)])
    for idx in indices:
        similarity = 0
        pair_count = 0
        # Compute the similarity for all pairs of nodes
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if nodes[i][idx[i]] is not None and nodes[j][idx[j]] is not None:
                    similarity += normalized_cosine_similarity(nodes[i][idx[i]], nodes[j][idx[j]])
                    pair_count += 1 # Count the number of valid pairs

        # Number of used nodes from the total number of valid pairs
        N = 1/2 + np.sqrt(1/4 + 2*pair_count)
        if pair_count > 0:
            similarity = similarity / pair_count * w_n[int(N-1)]
        similarities[idx] = similarity

    non_zero_similarities = [value for value in similarities.values() if value != 0]
    max_similarity = max(non_zero_similarities)
    for key in similarities:
        if similarities[key] == 0: # Fill similarities for associations with only one real feature
            similarities[key] = max_similarity * w_n[0]
    return similarities

def get_embeddings_from_buffer(data, rate):
    """
    Extract BirNet embeddings from a recording buffer.
    """
    analyzer = Analyzer()
    recording = RecordingBuffer(
        analyzer,
        data,
        rate,
    )
    recording.extract_embeddings()
    emb = np.asarray(recording.embeddings[0]["embeddings"])
    return emb

def add_error_to_angles(trueAngles, error):
    """
    Add error to the true angles using the Von Mises distribution.
    Args:
    - trueAngles: List of true angles in radians.
    - error: Error in degrees.

    Returns:
    - noisyAngles: List of noisy angles in radians.
    """
    if error is not None:
        error = np.deg2rad(error)
        noisyAngles = trueAngles.copy()

        for i in range(len(trueAngles)):
            while True:
                # Generate noisy angles using the Von Mises distribution
                noisyAngles[i] = np.random.normal(trueAngles[i], error, len(trueAngles[i]))
                # Normalize the angles to be within -pi to pi
                noisyAngles[i] = np.mod(noisyAngles[i] + np.pi, 2 * np.pi) - np.pi

                # Keep sourcePool with angular separation larger than array resolution
                overlap = False
                for j in range(len(noisyAngles[i])):
                    for k in range(j + 1, len(noisyAngles[i])):
                        # Circular difference calculation
                        angle_diff = np.abs(np.angle(np.exp(1j * (noisyAngles[i][j] - noisyAngles[i][k]))))
                        if angle_diff < 2 * error:
                            overlap = True
                            break
                    if overlap:
                        break

                if not overlap:
                    break
    return noisyAngles

def filter_sources(srcPos, micPos, tolerance, min_snr=None, noiseLevel=None, sourceLevel=None, snr_threshold=None):
    """
    Filters sourcePool for each node, adding missing detections as explained in the paper.

    Args:
    - srcPos: List of source positions.
    - micPos: List of microphone positions for each node.
    - tolerance: Tolerance for filtering sourcePool.
    - min_snr: Minimum SNR for filtering sourcePool.
    - noiseLevel: Noise level in dB.
    - sourceLevel: Source level in dB.
    - snr_threshold: SNR threshold for filtering sourcePool based on the SNR between sourcePool.

    Returns:
    - filtered_angles: Dictionary containing the filtered angles for each node.
    - filtered_sources: Dictionary containing the filtered sourcePool for each node.
    """
    nNodes = micPos.shape[0]
    arrayCenters = np.zeros((nNodes, 2))
    for i in range(nNodes):
        array = micPos[i]
        array_center = np.mean(array, axis=0)
        arrayCenters[i] = array_center

    trueAnglesRad = np.zeros((len(arrayCenters), len(srcPos)))

    for i in range(len(arrayCenters)):
        for j in range(len(srcPos)):
            angle = np.arctan2(srcPos[j][1] - arrayCenters[i][1], srcPos[j][0] - arrayCenters[i][0])
            trueAnglesRad[i, j] = angle

    filtered_angles = {}
    filtered_sources = {}

    for node in range(nNodes):
        node_angles = trueAnglesRad[node]
        node_pos = np.mean(micPos[node], axis=0)
        node_filtered_angles = []
        node_filtered_sources = []

        for i, angle in enumerate(node_angles):
            if min_snr is not None:
                if noiseLevel is None or sourceLevel is None:
                    raise ValueError("Noise and source levels are required for SNR filtering.")
                # Calculate SNR
                sourceSPL = sourceLevel - 20 * np.log10(np.linalg.norm(srcPos[i] - node_pos))
                snr = sourceSPL - noiseLevel
                if snr < min_snr:
                    continue  # Skip sourcePool with SNR below threshold

            keep = True
            for j, filtered_angle in enumerate(node_filtered_angles):
                if abs(angle - filtered_angle) <= tolerance:
                    # Compare distances and keep the closer source
                    if np.linalg.norm(srcPos[i] - node_pos) < np.linalg.norm(srcPos[node_filtered_sources[j]] - node_pos):
                        node_filtered_sources[j] = i
                        node_filtered_angles[j] = angle
                    keep = False
                    break

            if keep:
                node_filtered_sources.append(i)
                node_filtered_angles.append(angle)

        # Additional filtering based on SNR between sourcePool
        if snr_threshold is not None:
            final_filtered_sources = []
            final_filtered_angles = []
            for i, src_idx in enumerate(node_filtered_sources):
                keep = True
                for j, other_src_idx in enumerate(node_filtered_sources):
                    if i != j:
                        d1 = np.linalg.norm(srcPos[src_idx] - node_pos)
                        d2 = np.linalg.norm(srcPos[other_src_idx] - node_pos)
                        snr_between_sources = 20 * np.log10(d1 / d2)
                        if snr_between_sources < snr_threshold:
                            keep = False
                            break
                if keep:
                    final_filtered_sources.append(src_idx)
                    final_filtered_angles.append(node_filtered_angles[i])

            filtered_angles[node] = final_filtered_angles
            filtered_sources[node] = final_filtered_sources
        else:
            filtered_angles[node] = node_filtered_angles
            filtered_sources[node] = node_filtered_sources

    return filtered_angles, filtered_sources

def add_dummy(lists):
    """
    Add a dummy element to each list in the input list.
    Args:
    - lists: List of lists.

    Returns:
    - lists: List of lists with a dummy element added to each list.
    """
    for lst in lists:
        lst.append(None)
    return lists

def get_ground_truth_assignment(node_sources):
    """
    Get the ground truth assignment from the filtered sourcePool indices

    Args:
    - node_sources: Dictionary containing the filtered sourcePool for each node.

    Returns:
    - source_assignments_zero_based: List of tuples containing the source assignments for each node.
    """
    unique_sources = set()
    for sources in node_sources.values():
        unique_sources.update(sources)
    assignment = []

    for source in unique_sources:
        association = []
        for node in sorted(node_sources.keys()):
            # Find index of the source in the node's list or use None
            if source in node_sources[node]:
                association.append(node_sources[node].index(source))  # 0-based indexing
            else:
                association.append(None)
        assignment.append(tuple(association))

    return assignment

def remove_dummy_features(assignment, node_sources):
    """
    Replace dummy features with None in the assignment.

    Args:
    - assignment: List of tuples containing the associations.
    - node_sources: Dictionary containing the filtered sourcePool for each node.

    Returns:
    - corrected_associations: List of tuples containing the associations with None instead of dummy features.
    """
    valid_sources_per_node = {node: np.arange(len(sources)) for node, sources in node_sources.items()}
    corrected_associations = []
    for association in assignment:
        corrected_tuple = []
        for node_idx, source_idx in enumerate(association):
            if source_idx not in valid_sources_per_node[node_idx]:
                corrected_tuple.append(None)  # Replace dummy feature with None
            else:
                corrected_tuple.append(source_idx)
        corrected_associations.append(tuple(corrected_tuple))
    # if all features are dummies, remove the assignment
    corrected_associations = [assignment for assignment in corrected_associations if any(source is not None for source in assignment)]
    return corrected_associations

def plot_room(micPos, srcPos, roomDims):
    """
    Plot the room with the node centers and source positions.
    Args :
    - nodeCenters : List of node centers.
    - srcPos : List of source positions.
    - roomDims : Room dimensions.
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13

    # Calculate and plot node centers
    nodeCenters = np.mean(micPos, axis=1)
    for i, center in enumerate(nodeCenters):
        plt.plot(center[0], center[1], 'o', label=f'Node {i + 1} Center')
        #plt.text(center[0], center[1], f'Node {i + 1}', fontsize=12, ha='right')

    # Plot sources
    for i, src in enumerate(srcPos):
        plt.plot(src[0], src[1], 'x', label=f'Source {i + 1}')
        #plt.text(src[0], src[1], f'Source {i + 1}', fontsize=12, ha='right')

    plt.xlim([0, roomDims])
    plt.ylim([0, roomDims])
    plt.xlabel(' (m)', fontsize=16)
    plt.ylabel(' (m)', fontsize=16)
    plt.legend()
    #fig.savefig('Figures/scenario.eps', format='eps', dpi=1200)
    #fig.savefig('Figures/scenario.svg', format='svg', dpi=1200)
    plt.show()

def clear_folder(folder_path):
    files = glob.glob(os.path.join(folder_path, '*'))
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

import numpy as np
from simulate_signals import _simulate
from utils import add_error_to_angles, fill_sparse_dict, get_ground_truth_assignment, remove_dummy_features, filter_sources
from fmdap import solve_fractional_mdap
from embedding_extractor import broadband_beamform, extract_embeddings
from metrics import compute_rca, compute_rcpa
import time
import pickle


def random_simulation(nNodes=4, micsPerNode=4, micSpacing=0.05, nSources=4, environmentSize=30, noisePower=60, sourcePower=80, doaError = 3, alpha = 0.1):

    # Parameters for beamforming
    nfft = 256 * 4
    overlap = 128 * 4
    tolerance = max(np.deg2rad(4), np.deg2rad(doaError * 2))  # Node angular resolution

    fs, micPos, signals, srcPos = _simulate(nNodes=nNodes, micsPerNode=micsPerNode, environmentSize=environmentSize,
                                            micSpacing=micSpacing, nSources=nSources, noisePower=noisePower,
                                            sourcePower=sourcePower)

    # Filter sourcePool

    filtered_true_angles, filtered_sources = filter_sources(srcPos, micPos, tolerance, min_snr=0, noiseLevel=noisePower,
                                                            sourceLevel=sourcePower, snr_threshold=-10)
    # Check that all nodes have at least 1 source
    if not filtered_sources:
        print("No sourcePool detected. Skipping simulation")
        return -1, -1, -1, -1
    if min(len(sublist) for sublist in filtered_sources.values()) == 0:
        print("At least one node has no sourcePool. Skipping simulation")
        return -1, -1, -1, -1
    if max(len(sublist) for sublist in filtered_sources.values()) == 1:
        print("Only one source detected per node. Skipping simulation")
        return -1, -1, -1, -1
    filtered_error_angles = add_error_to_angles(filtered_true_angles, doaError)
    groundTruth = get_ground_truth_assignment(filtered_sources)

    y = broadband_beamform(micPos, signals, fs, nfft, filtered_error_angles, overlap)
    ##### Using birdnetlib
    embeddings = extract_embeddings(y, nNodes, fs)
    nodes = []
    start_idx = 0
    # Divide ss into nodes of the same shape as y
    for sublist in y:
        end_idx = start_idx + len(sublist)
        nodes.append(embeddings[start_idx:end_idx])
        start_idx = end_idx

    # Fill similarity tensor
    simDict = fill_sparse_dict(nodes, alpha)
    shape = [len(nodes[i]) for i in range(len(nodes))]

    # Solve the fractional MDAP
    assignment = solve_fractional_mdap(simDict, shape)

    assignment = remove_dummy_features(assignment, filtered_sources)

    # Compute metrics
    rca = compute_rca(groundTruth, assignment)
    rcpa = compute_rcpa(groundTruth, assignment)

    # Compute % of missing detections
    missing_detections = 0
    for i in range(nNodes):
        missing_detections += nSources - len(filtered_sources[i])
    missing_detections /= nSources * nNodes

    # Estimated number of sourcePool
    estimated_sources = len(assignment)
    real_sources = len(groundTruth)
    ratio = estimated_sources / real_sources
    ratio = min(ratio, 1 / ratio)

    return rca, rcpa, missing_detections * 100, ratio * 100


def run_simulations(nSources, noisePower, micsPerNode, nSims=200):
    """Runs a number of simulations and returns statistics."""
    np.random.seed(123) # To obtain same results as in the paper
    rcas, rcpas, missedDets, sourceRatios = [], [], [], []
    simCounter = 0

    while simCounter < nSims:
        rca, rcpa, missedDet, sourceRatio = random_simulation(
            nSources=nSources, noisePower=noisePower, micsPerNode=micsPerNode
        )
        if rca == -1:
            continue
        rcas.append(rca)
        rcpas.append(rcpa)
        missedDets.append(missedDet)
        sourceRatios.append(sourceRatio)
        simCounter += 1

    return {
        'mean_rca': np.mean(rcas),
        'mean_rcpa': np.mean(rcpas),
        'mean_missedDet': np.mean(missedDets),
        'mean_sourceRatio': np.mean(sourceRatios)
    }


if __name__ == '__main__':
    # Parameter sets
    micsPerNode = [2, 3, 4, 5, 6, 7, 8]
    noisePowers = [50, 55, 60]
    numSources = [3, 5]
    nSims = 200  # Number of simulations per configuration

    # Store results in a structured dictionary
    results = {
        'mean_rcas': [], 'mean_rcpas': [], 'mean_missedDets': [], 'mean_sourceRatios': [],
    }

    start_time = time.time()

    # Nested list comprehensions to run simulations for each configuration
    for mics in micsPerNode:
        doa_results = {key: [] for key in results}
        for noisePower in noisePowers:
            noise_results = {key: [] for key in results}
            for nSource in numSources:
                stats = run_simulations(nSources=nSource, noisePower=noisePower, micsPerNode=mics, nSims=nSims)

                # Aggregate results
                noise_results['mean_rcas'].append(stats['mean_rca'])
                noise_results['mean_rcpas'].append(stats['mean_rcpa'])
                noise_results['mean_missedDets'].append(stats['mean_missedDet'])
                noise_results['mean_sourceRatios'].append(stats['mean_sourceRatio'])

            # Append aggregated noise level results
            for key in results:
                doa_results[key].append(noise_results[key])

        # Append aggregated DOA error level results
        for key in results:
            results[key].append(doa_results[key])

    # Save results to a file
    with open('Results/doaError_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    elapsed_time = time.time() - start_time
    print(f"Simulation completed in {elapsed_time:.2f} seconds.")
    print("Results saved to 'Results/micsXNode_results.pkl'")



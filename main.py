import numpy as np
import argparse
import time
from simulate_signals import _simulate
from utils import add_error_to_angles, filter_sources, get_ground_truth_assignment, remove_dummy_features, \
    fill_sparse_dict
from fmdap import solve_fractional_mdap
from embedding_extractor import broadband_beamform, extract_embeddings
from metrics import compute_rca, compute_rcpa
import soundfile as sf

def main(args):
    # Independent parameters
    tolerance = max(np.deg2rad(4), np.deg2rad(args.doaError * 2))  # Node angular resolution

    # Simulation
    fs, micPos, signals, srcPos = _simulate(
        nNodes=args.nNodes,
        micsPerNode=args.micsPerNode,
        environmentSize=args.environmentSize,
        micSpacing=args.micSpacing,
        nSources=args.nSources,
        noisePower=args.noisePower,
        sourcePower=args.sourcePower,
        saveSignals=args.saveSignals
    )

    # Plot room
    if args.plotRoom:
        from utils import plot_room
        plot_room(micPos, srcPos, args.environmentSize)

    # Filter sourcePool
    filtered_true_angles, filtered_sources = filter_sources(
        srcPos, micPos, tolerance, min_snr=0,
        noiseLevel=args.noisePower, sourceLevel=args.sourcePower, snr_threshold=-10
    )
    filtered_error_angles = add_error_to_angles(filtered_true_angles, args.doaError)

    # Get ground truth
    groundTruth = get_ground_truth_assignment(filtered_sources)

    start_time_1 = time.time()

    # Beamforming and embedding extraction
    y = broadband_beamform(micPos, signals, fs, args.nfft, filtered_error_angles, args.overlap)
    if args.saveSignals:
        for i in range(args.nNodes):
            for j in range(y[i].__len__()):
                sf.write(f"audios/beamformed/{i + 1}node_to{filtered_sources[i][j] + 1}source.wav", y[i][j], fs, format='wav')
    embeddings = extract_embeddings(y, args.nNodes, fs)

    # Divide embeddings into nodes
    nodes = []
    start_idx = 0
    for sublist in y:
        end_idx = start_idx + len(sublist)
        nodes.append(embeddings[start_idx:end_idx])
        start_idx = end_idx

    start_time = time.time()

    # Fill similarity tensor
    simDict = fill_sparse_dict(nodes, args.alpha)
    shape = [len(nodes[i]) for i in range(len(nodes))]

    # Solve the fractional MDAP
    assignment = solve_fractional_mdap(simDict, shape)
    assignment = remove_dummy_features(assignment, filtered_sources)

    # Output the assignments and total similarity
    print("Estimated Assignments:")
    for association in assignment:
        print(association)
    print("Ground Truth Assignments:")
    for association in groundTruth:
        print(association)

    # Compute metrics
    rca = compute_rca(groundTruth, assignment)
    rcpa = compute_rcpa(groundTruth, assignment)

    print(f"rca (%): {rca}%")
    print(f"rcpa (%): {rcpa:.2f}%")

    end_time = time.time()
    elapsed_time_total = end_time - start_time_1
    print(f"Total elapsed time: {elapsed_time_total}")
    elapsed_time_mdap = end_time - start_time
    print(f"F-MDAP elapsed time: {elapsed_time_mdap}")
    print("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for the DOA AP.")
    parser.add_argument("--nNodes", type=int, default=4, help="Number of nodes")
    parser.add_argument("--micsPerNode", type=int, default=4, help="Microphones per node")
    parser.add_argument("--environmentSize", type=float, default=30, help="Environment size (meters)")
    parser.add_argument("--micSpacing", type=float, default=0.05, help="Microphone spacing (meters)")
    parser.add_argument("--nSources", type=int, default=3, help="Number of sourcePool")
    parser.add_argument("--doaError", type=float, default=2, help="Direction of Arrival estimation error (degrees)")
    parser.add_argument("--noisePower", type=float, default=50, help="Noise power level (db SLP)")
    parser.add_argument("--sourcePower", type=float, default=80, help="Source power level(db SLP)")
    parser.add_argument("--alpha", type=float, default=0.15, help="Weight for the similarity tensor (alpha in paper)")
    parser.add_argument("--nfft", type=int, default=256 * 4, help="Number of FFT points")
    parser.add_argument("--overlap", type=int, default=128 * 4, help="FFT overlap size")
    parser.add_argument("--saveSignals", type=bool, default=True, help="Save source/mixed/beamformed signals")
    parser.add_argument("--plotRoom", type=bool, default=True, help="Whether to plot the room or not")

    args = parser.parse_args()
    main(args)

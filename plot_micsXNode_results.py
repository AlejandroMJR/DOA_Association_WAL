import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines as mlines
import seaborn as sns
import pickle

# p = 0.1
# doaError = 3
micsPerNode = [2,3,4,5,6,7,8]
noisePowers = [50, 55, 60]
numSources = [3, 5]

with open('Results/micsXNode_results.pkl', 'rb') as f:
    results = pickle.load(f)

mean_rca = results["mean_rcas"]
mean_rca = np.array(mean_rca)

mean_rcpa = results["mean_rcpas"]
mean_rcpa = np.array(mean_rcpa)

palette = sns.color_palette("Set1", n_colors=3)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.figure()
plt.subplots_adjust(bottom=0.15, left=0.13)

# Create lists to store handles and labels for the legend
handles_noise = []
labels_noise = []
handles_source = []
labels_source = []

for j, source in enumerate(numSources):
    for i, snr in enumerate(noisePowers):
        label_noise = fr"$L_{{\text{{noise}}}} = {snr} $dB SPL"

        # Plot the data with lines and markers for the data points
        line, = plt.plot(micsPerNode, mean_rca[:, i, j],
                         linestyle='-' if source == 3 else '--', color=palette[i], marker='o')

        # Add a line for the noise level, but only add it once per noise level
        if label_noise not in labels_noise:
            # Add only the marker (not the line) for noise level to the legend
            marker = mlines.Line2D([], [], color=palette[i], marker='o', linestyle='None', label=label_noise)
            handles_noise.append(marker)
            labels_noise.append(label_noise)

# Add manual entries for source types with different line styles
handles_source.append(plt.Line2D([0], [0], color='black', linestyle='-'))  # For 2 sourcePool
handles_source.append(plt.Line2D([0], [0], color='black', linestyle='--'))  # For 4 sourcePool
labels_source = ['3 sourcePool', '5 sourcePool']

# Combine handles for the legend: first noise and then sourcePool
#plt.legend(handles=handles_noise + handles_source, labels=labels_noise + labels_source, fontsize=14)

plt.xlabel("Number of microphones, $M$", fontsize=18)
plt.ylabel('RCA(%)', fontsize=18)
#plt.title("(a)", fontsize=18)
plt.ylim(0, 100)
#plt.text(7.8, 2, "(a)", fontsize=18)
plt.savefig(f"Results/Figures/RCA vs M.pdf", format='pdf', dpi=1000)
plt.show()


plt.figure()
plt.subplots_adjust(bottom=0.15, left=0.13)

# Create lists to store handles and labels for the legend
handles_noise = []
labels_noise = []
handles_source = []
labels_source = []

for j, source in enumerate(numSources):
    for i, snr in enumerate(noisePowers):
        label_noise = fr"$L_{{\text{{noise}}}} = {snr} $dB SPL"

        # Plot the data with lines and markers for the data points
        line, = plt.plot(micsPerNode, mean_rcpa[:, i, j],
                         linestyle='-' if source == 3 else '--', color=palette[i], marker='o')

        # Add a line for the noise level, but only add it once per noise level
        if label_noise not in labels_noise:
            # Add only the marker (not the line) for noise level to the legend
            marker = mlines.Line2D([], [], color=palette[i], marker='o', linestyle='None', label=label_noise)
            handles_noise.append(marker)
            labels_noise.append(label_noise)

# Add manual entries for source types with different line styles
handles_source.append(plt.Line2D([0], [0], color='black', linestyle='-'))  # For 3 sourcePool
handles_source.append(plt.Line2D([0], [0], color='black', linestyle='--'))  # For 5 sourcePool
labels_source = ['3 sourcePool', '5 sourcePool']

# Combine handles for the legend: first noise and then sourcePool
plt.legend(handles=handles_noise + handles_source, labels=labels_noise + labels_source, fontsize=14)

plt.xlabel(r"Number of microphones, $M$", fontsize=18)
plt.ylabel('RCPA(%)', fontsize=18)
#plt.title("(b)", fontsize=18)
plt.ylim(0, 100)
#plt.text(7.8, 2, "(b)", fontsize=18)
plt.savefig(f"Results/Figures/RCPA vs M.pdf", format='pdf', dpi=1000)

plt.show()









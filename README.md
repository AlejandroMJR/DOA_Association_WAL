# DOA_Association_WAL
Direction-of-Arrival Data Association for Wildlife Acoustic Localization - Implementation and Simulations.

## Environment Setup

This project supports two setup methods:

### Using Conda 

1. Create a new Conda environment:
   ```bash
   conda env create -f environment.ylm -n envname
   ```

2. Activate the environment:
   ```bash
   conda activate envname
   ```

### Using pip 

If you prefer not to use Conda, install the dependencies using `pip`:

```bash
pip install -r requirements.txt
```

---

## Quick Test Run

After setting up the environment, run the main script:

```bash
python3 main.py
```

- This will run a random simulation and print the ground truth associations and the estimated asssociations

---

## Usage

To view all available parameters and options:

```bash
python3 main.py -h
```

By default, the script will run a single random simulation using the default parameters.

---

## Output Files

The script will generate and save audio files to verify everything is working as expected:

- **`audio/micSignals/`**  
  Contains microphone signals with mixed sources — these are the raw mic recordings after simulation.

- **`audio/beamformed/`**
  - `#source.wav`: The raw source signals used for the simulation (where `#` is the source index).
  - `$node_to#source.wav`: The beamformed signals, where each file represents a beamformed signal from node `$node` to source `#source`.

    For example:
    - `1node_to1source.wav` should mainly contain the signal from `1source.wav` as received and processed by node 1.
    - Some combinations may not be present due to missed detections by the system.

---

## To replicate results and plots of the paper

Run the scripts `monte_carlo_doa_erros.py` and `monte_carlo_micsxnode.py`. This will save the results in the folder **`Results`**. Then you can run `plot_doaError_results.py` and `plot_micsXNode_results.py` to obtain figures 2 and 3 respectively. 

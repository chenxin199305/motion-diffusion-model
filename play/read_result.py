import os
import json
import pickle
import numpy as np

if __name__ == "__main__":
    # Paths
    HERE = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(
        HERE,
        "..",
        "save",
        "humanml_trans_enc_512",
        "samples_humanml_trans_enc_512_000200000_seed10_the_person_wave_hand_say_hello",
        "results.npy",
    )

    # Load the results.npz file
    results = np.load(result_path, allow_pickle=True).item()
    print(f"Loaded results from {result_path}")

    # Print keys in the results dictionary
    print(f"Keys in results: {list(results.keys())}")

    # Extract relevant data
    motion = results.get('motion', [])
    text = results.get('text', [])
    lengths = results.get('lengths', [])
    num_samples = results.get('num_samples', 0)
    num_repetitions = results.get('num_repetitions', 0)

    print(
        f"Number of samples: {num_samples}, Number of repetitions: {num_repetitions}, "
        f"Motion shape: {motion.shape if isinstance(motion, np.ndarray) else 'N/A'}, "
        f"Text length: {len(text) if isinstance(text, list) else 'N/A'}, "
        f"Lengths shape: {lengths.shape if isinstance(lengths, np.ndarray) else 'N/A'}"
    )

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

    # Load the results JSON file
    with open(result_path, 'r') as f:
        results = json.load(f)

    # Iterate through each result entry
    for idx, entry in enumerate(results):
        motion_data = np.array(entry['motion'])  # Assuming motion data is stored under 'motion' key
        text_prompt = entry.get('text', 'No prompt provided')  # Get text prompt if available

        # Save motion data as a .npy file
        motion_save_path = os.path.join(save_dir, f"motion_{idx:03d}.npy")
        np.save(motion_save_path, motion_data)

        # Save text prompt as a .txt file
        text_save_path = os.path.join(save_dir, f"prompt_{idx:03d}.txt")
        with open(text_save_path, 'w') as txt_file:
            txt_file.write(text_prompt)

        print(f"Saved motion to {motion_save_path} and prompt to {text_save_path}")

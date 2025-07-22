import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

LANGUAGE = os.getenv("LANGUAGE", "nheengatu")
SEGMENTER = os.getenv("SEGMENTER", "baseline")

def run_script(script_name):
    print(f"\n--- Running {script_name} ---")
    try:
        subprocess.run(["python", script_name], check=True, text=True, capture_output=False)
        print(f"--- Finished {script_name} ---\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        exit(1)

if __name__ == "__main__":
    print("Starting the BPE Segmentation and Training Pipeline...")

    # Step 1: Initial Data Extraction
    if LANGUAGE != "bororo5k":
        run_script("extract_tokens_tags.py")

    # Step 2: BPE Segmentation
    if SEGMENTER != "baseline":
        formatted_seg = SEGMENTER.lower()
        run_script(f"train_segment_{formatted_seg}.py")

        # Step 3: Propagate Tags for segmented tokens
        run_script("propagate_tags.py")

    # Step 4: Train the model using the prepared data
    run_script("train.py")

    print("\n✅ Extraction, Segmentation and Training Pipeline Complete!")
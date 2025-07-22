import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

LANGUAGE = os.getenv("LANGUAGE", "nheengatu")
MERGES = 1000  # number of merges
CODES_PATH = f"models/bpe/bpe_{MERGES}_{LANGUAGE}.codes"
BASELINE_FOLDER = "data/baseline"
BPE_FOLDER = "data/bpe"

os.makedirs("models/bpe", exist_ok=True)
os.makedirs(BPE_FOLDER, exist_ok=True)

def run_command(command):
    print(f"▶️ {command}")
    subprocess.run(command, shell=True, check=True)

def learn_bpe():
    cmd = f"subword-nmt learn-bpe -s {MERGES} < {BASELINE_FOLDER}/{LANGUAGE}_train_tokens.txt > {CODES_PATH}"
    run_command(cmd)

def apply_bpe(split):
    input_path = f"{BASELINE_FOLDER}/{LANGUAGE}_{split}_tokens.txt"
    output_path = f"{BPE_FOLDER}/{LANGUAGE}_{split}_bpe_tokens.txt"
    cmd = f"subword-nmt apply-bpe -c {CODES_PATH} < {input_path} > {output_path}"
    run_command(cmd)

def main():
    print(f"🚀 Learning BPE on {LANGUAGE} (merges={MERGES})")
    learn_bpe()
    print("✅ Done learning BPE!")

    for split in ["train", "dev"]:
        print(f"🚀 Applying BPE to {split} set")
        apply_bpe(split)
        print(f"✅ Finished {split}!")

if __name__ == "__main__":
    main()
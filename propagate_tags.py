from dotenv import load_dotenv
import os

load_dotenv()

LANGUAGE = os.getenv("LANGUAGE", "nheengatu")
SEGMENTER = os.getenv("SEGMENTER", "baseline")

print(f"Language selected: {LANGUAGE} | Segmenter selected: {SEGMENTER}")

BASELINE_FOLDER = 'data/baseline'

def propagate_tags(train_tokens_path, train_tags_path, segmenter_tokens_path, output_tags_path):
    with open(train_tokens_path, 'r', encoding='utf-8') as f_tokens, \
         open(train_tags_path, 'r', encoding='utf-8') as f_tags, \
         open(segmenter_tokens_path, 'r', encoding='utf-8') as f_segment, \
         open(output_tags_path, 'w', encoding='utf-8') as f_out:

        for orig_tokens, orig_tags, segmented_tokens in zip(f_tokens, f_tags, f_segment):
            orig_tokens = orig_tokens.strip().split()
            orig_tags = orig_tags.strip().split()
            segmented_tokens = segmented_tokens.strip().split()

            propagated_tags = []
            orig_idx = 0

            for segmented_token in segmented_tokens:
                propagated_tags.append(orig_tags[orig_idx])
                # Detect if this is the last subword of a word (no '@@')
                if not segmented_token.endswith('@@'):
                    orig_idx += 1

            assert orig_idx == len(orig_tags), f"Tagging mismatch in line: {orig_tokens}"

            f_out.write(' '.join(propagated_tags) + '\n')

    print(f"✅ {SEGMENTER} tags written to {output_tags_path}")

if SEGMENTER == "baseline":
    print("There's nothing to propagate! If baseline files are empty, please, run 'python extract_tokens_tags.py' again!")
    exit(0)

prefixes = ["train", "dev"]
segm_identifier = "_" if SEGMENTER == "baseline" else f"_{SEGMENTER}_"
for prefix in prefixes:
    propagate_tags(
        train_tokens_path=f'{BASELINE_FOLDER}/{LANGUAGE}_{prefix}_tokens.txt',
        train_tags_path=f'{BASELINE_FOLDER}/{LANGUAGE}_{prefix}_tags.txt',
        segmenter_tokens_path=f'data/{SEGMENTER}/{LANGUAGE}_{prefix}{segm_identifier}tokens.txt',
        output_tags_path=f'data/{SEGMENTER}/{LANGUAGE}_{prefix}{segm_identifier}tags.txt'
    )

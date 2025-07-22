from conllu import parse_incr

from dotenv import load_dotenv
import os

load_dotenv()

LANGUAGE = os.getenv("LANGUAGE", "nheengatu")
BASELINE_FOLDER = 'data/baseline'

print(f"Language selected: {LANGUAGE}")

def extract_sentences_and_pos_tags(conllu_path):
    sentences = []
    pos_tags = []
    raw_sentences = []

    with open(conllu_path, 'r', encoding='utf-8') as file:
        for tokenlist in parse_incr(file):
            words = []
            tags = []
            for token in tokenlist:
                if isinstance(token['id'], int):  # skip empty nodes
                    words.append(token['form'])
                    tags.append(token['upos'])
            sentences.append(words)
            pos_tags.append(tags)
            raw_sent = tokenlist.metadata.get("text", None)
            raw_sentences.append(raw_sent)

    return sentences, pos_tags, raw_sentences


def save_sentences_to_file(sentences, output_path):
    with open(f"{BASELINE_FOLDER}/{output_path}", 'w', encoding='utf-8') as f:
        for sent in sentences:
            f.write(' '.join(sent) + '\n')


def save_tags_to_file(tags, output_path):
    with open(f"{BASELINE_FOLDER}/{output_path}", 'w', encoding='utf-8') as f:
        for tag_seq in tags:
            f.write(' '.join(tag_seq) + '\n')


def extract_and_save(conllu_path, output_prefix, language=LANGUAGE):
    sentences, pos_tags, raw_sentences = extract_sentences_and_pos_tags(conllu_path)

    save_sentences_to_file(sentences, f"{language}_{output_prefix}_tokens.txt")
    save_tags_to_file(pos_tags, f"{language}_{output_prefix}_tags.txt")
    # save_sentences_to_file(raw_sentences, f"{output_prefix}_raw_sentences.txt")

    print(f"✅ Saved {language} tokens and tags to {BASELINE_FOLDER}/{language}_{output_prefix}_*.txt")

input_prefixes = ["dev", "train"]
for input_prefix in input_prefixes:
    train_file = f"data/corpora/{LANGUAGE}/{input_prefix}_corpus.conllu"
    extract_and_save(train_file, input_prefix, LANGUAGE)

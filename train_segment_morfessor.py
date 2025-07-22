import morfessor
import os
from dotenv import load_dotenv
load_dotenv()

LANGUAGE = os.getenv("LANGUAGE", "nheengatu")

MORFESSOR_MODEL_PATH = f'models/morfessor/morfessor_{LANGUAGE}.model'
BASELINE_FOLDER = 'data/baseline'
DATA_MORFESSOR_FOLDER = 'data/morfessor'

os.makedirs(DATA_MORFESSOR_FOLDER, exist_ok=True)

def train_morfessor_model(train_tokens_path, model_path):
    io = morfessor.MorfessorIO()
    model = morfessor.BaselineModel()

    # Morfessor espera um corpus em que cada linha tem uma palavra
    tmp_file = 'morfessor_wordlist.txt'
    with open(train_tokens_path, 'r', encoding='utf-8') as f_in, \
         open(tmp_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            words = line.strip().split()
            for word in words:
                f_out.write(word + '\n')

    data = io.read_corpus_file(tmp_file)
    model.load_data(data)
    model.train_batch(max_epochs=10, algorithm='recursive')
    io.write_binary_model_file(model_path, model)
    os.remove(tmp_file)
    print(f"✅ Morfessor model trained and saved to: {model_path}")
    return model

#segmenta em um formato compatível com o propagador
def segment_file_with_model(model, input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            words = line.strip().split()
            segmented = []
            for word in words:
                parts = model.viterbi_segment(word)[0]
                # Adiciona '@@' aos morfemas intermediários
                segmented.extend([p + "@@" for p in parts[:-1]] + [parts[-1]])
            f_out.write(' '.join(segmented) + '\n')
    print(f"✅ Segmented output saved to: {output_path}")

def main():
    prefixes = ['train', 'dev']
    
    train_tokens = f'{BASELINE_FOLDER}/{LANGUAGE}_train_tokens.txt'
    model = train_morfessor_model(train_tokens, MORFESSOR_MODEL_PATH)

    for prefix in prefixes:
        in_path = f'{BASELINE_FOLDER}/{LANGUAGE}_{prefix}_tokens.txt'
        out_path = f'{DATA_MORFESSOR_FOLDER}/{LANGUAGE}_{prefix}_morfessor_tokens.txt'
        segment_file_with_model(model, in_path, out_path)

if __name__ == '__main__':
    main()

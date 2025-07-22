import os
import flatcat
from flatcat.categorizationscheme import MorphUsageProperties

from dotenv import load_dotenv
load_dotenv()

LANGUAGE = os.getenv("LANGUAGE", "nheengatu")

FLATCAT_MODEL_PATH = f'models/flatcat/flatcat_{LANGUAGE}.model'
BASELINE_FOLDER = 'data/baseline'
DATA_FLATCAT_FOLDER = 'data/flatcat'

os.makedirs('models/flatcat', exist_ok=True)
os.makedirs(DATA_FLATCAT_FOLDER, exist_ok=True)

def create_segmentation_file(train_tokens_path, segmentation_file):
    """Create a segmentation file in the format expected by FlatCat"""
    with open(train_tokens_path, 'r', encoding='utf-8') as f_in, \
         open(segmentation_file, 'w', encoding='utf-8') as f_out:
        
        word_counts = {}
        
        # Count word frequencies
        for line in f_in:
            words = line.strip().split()
            for word in words:
                if word.strip():
                    word_counts[word.strip()] = word_counts.get(word.strip(), 0) + 1
        
        # Write in FlatCat segmentation format: count word
        for word, count in word_counts.items():
            f_out.write(f"{count} {word}\n")
    
    print(f"✅ Created segmentation file: {segmentation_file}")

def train_flatcat_model(train_tokens_path, model_path):
    """Train FlatCat model using the proper API"""
    
    # Create temporary segmentation file
    segmentation_file = 'temp_segmentation.txt'
    create_segmentation_file(train_tokens_path, segmentation_file)
    
    try:
        # Initialize FlatCat components
        io = flatcat.FlatcatIO()
        morph_usage = MorphUsageProperties()
        
        # Create model with proper parameters
        model = flatcat.FlatcatModel(morph_usage, corpusweight=1.0)
        
        print("📖 Loading corpus data...")
        # Add corpus data from segmentation file
        segmentation_data = io.read_segmentation_file(segmentation_file)
        model.add_corpus_data(segmentation_data)
        
        print("⏳ Initializing HMM...")
        # Initialize HMM parameters
        model.initialize_hmm()
        
        print("⏳ Training FlatCat model... This might take a while.")
        # Train the model
        model.train_batch(max_epochs=10)
        
        print("💾 Saving model...")
        # Save the trained model
        io.write_binary_model_file(model_path, model)
        print(f"✅ FlatCat model trained and saved to: {model_path}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise
    finally:
        # Clean up temporary file
        if os.path.exists(segmentation_file):
            os.remove(segmentation_file)
    
    return model

def segment_file_with_flatcat_model(model, input_path, output_path, output_categories=False):
    """Segment files using trained FlatCat model"""
    print(f"🔄 Segmenting file: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"⚠️  Input file not found: {input_path}")
        return
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            for line_num, line in enumerate(f_in, 1):
                line = line.strip()
                if not line:
                    f_out.write('\n')
                    continue
                
                words = line.split()
                segmented_line = []
                
                for word in words:
                    if not word.strip():
                        segmented_line.append(word)
                        continue
                    
                    # Skip punctuation - don't try to segment it
                    if word.strip() in [',', '.', ';', ':', '!', '?', '"', "'", '(', ')', '[', ']', '{', '}']:
                        segmented_line.append(word)
                        continue
                    
                    try:
                        if output_categories:
                            # Get analysis with categories
                            analysis = model.viterbi_analyze(word.strip())
                            # analysis is a list of CategorizedMorph objects
                            categorized_segments = [f"{morph.morph}/{morph.category}" for morph in analysis]
                            segmented_line.append(' '.join(categorized_segments))
                        else:
                            # Get just the segmentation - handle different return formats
                            result = model.viterbi_segment(word.strip())
                            
                            if len(result) == 3:
                                segments, categories, _ = result
                            elif len(result) == 2:
                                segments, categories = result
                            else:
                                # Fallback - assume first item is segments
                                segments = result[0] if isinstance(result, (list, tuple)) else result
                                categories = None
                            
                            # Format with @@ suffix except for last segment
                            if isinstance(segments, (list, tuple)) and len(segments) > 1:
                                formatted_segments = [s + "@@" for s in segments[:-1]] + [segments[-1]]
                            elif isinstance(segments, (list, tuple)):
                                formatted_segments = segments
                            else:
                                # Single segment case
                                formatted_segments = [segments]
                            
                            segmented_line.append(' '.join(formatted_segments))
                    
                    except Exception as e:
                        print(f"⚠️  Error segmenting word '{word}' on line {line_num}: {e}")
                        segmented_line.append(word)  # Use original word if segmentation fails
                
                f_out.write(' '.join(segmented_line) + '\n')
        
        print(f"✅ Segmented output (FlatCat) saved to: {output_path}")
    
    except Exception as e:
        print(f"❌ Error during segmentation: {e}")
        raise

def load_existing_model(model_path):
    """Load an existing FlatCat model"""
    io = flatcat.FlatcatIO()
    return io.read_binary_model_file(model_path)

def main():
    prefixes = ['train', 'dev']
    train_tokens = f'{BASELINE_FOLDER}/{LANGUAGE}_train_tokens.txt'
    
    # Check if training file exists
    if not os.path.exists(train_tokens):
        print(f"❌ Training file not found: {train_tokens}")
        return
    
    print(f"🚀 Starting FlatCat training for {LANGUAGE}")
    
    try:
        # Check if model already exists
        if os.path.exists(FLATCAT_MODEL_PATH):
            print(f"📂 Loading existing model from: {FLATCAT_MODEL_PATH}")
            flatcat_model = load_existing_model(FLATCAT_MODEL_PATH)
        else:
            # Train new model
            flatcat_model = train_flatcat_model(train_tokens, FLATCAT_MODEL_PATH)

        # Segment files
        for prefix in prefixes:
            in_path = f'{BASELINE_FOLDER}/{LANGUAGE}_{prefix}_tokens.txt'
            out_path = f'{DATA_FLATCAT_FOLDER}/{LANGUAGE}_{prefix}_flatcat_tokens.txt'
            
            if os.path.exists(in_path):
                segment_file_with_flatcat_model(flatcat_model, in_path, out_path)
            else:
                print(f"⚠️  Skipping missing file: {in_path}")
        
        print("🎉 FlatCat processing completed successfully!")
    
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        raise

if __name__ == '__main__':
    main()
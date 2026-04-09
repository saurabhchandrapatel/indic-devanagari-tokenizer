#Google SentencePiece based Tokenizers
import sentencepiece as spm
import os
import unicodedata
from  preprocess import panini_pre_tokenize, merge_compounds, panini_split, clean_text

def train_sentencepiece_tokenizer(
    corpus_paths: list or str,
    model_prefix: str = './models/devanagari_sp',
    vocab_size: int = 16000,
    model_type: str = "unigram",  # or "bpe"
    character_coverage: float = 0.9995  # High for Devanagari
):
    """
    Train SentencePiece tokenizer
    
    Args:
        corpus_path: Path to corpus file
        model_prefix: Output model name prefix
        vocab_size: Target vocabulary size
        model_type: "bpe" or "unigram"
        character_coverage: Coverage of characters (0.9995 for rare chars)
    """
      
    # Convert list → comma-separated string
    if isinstance(corpus_paths, list):
        input_files = ",".join(corpus_paths)
    else:
        input_files = corpus_paths


    # Training arguments
    train_args = {
        'input': input_files,
        'model_prefix': model_prefix,
        'vocab_size': vocab_size,
        'model_type': model_type,
        'character_coverage': character_coverage,
        'pad_id': 0,
        'unk_id': 1,
        'bos_id': 2,
        'eos_id': 3,
        'pad_piece': '[PAD]',
        'unk_piece': '[UNK]',
        'bos_piece': '[BOS]',
        'eos_piece': '[EOS]',
        'user_defined_symbols': ['[MASK]', '[CLS]', '[SEP]'],
        'num_threads': os.cpu_count(),
        'input_sentence_size': 1000000,  # Sample 1M sentences for training
        'shuffle_input_sentence': True,  # Shuffle for better training
        'train_extremely_large_corpus': False,  # For 3GB corpus
	'max_sentencepiece_length': 24,
        'split_by_whitespace': 1
    }
    
    # Train
    print(f"Training SentencePiece tokenizer...")
    print(f"Vocab size: {vocab_size}, Model type: {model_type}")
    
    spm.SentencePieceTrainer.train(**train_args)
    
    print(f"\n✓ Model saved: {model_prefix}.model")
    print(f"✓ Vocab saved: {model_prefix}.vocab")
    
    return f"{model_prefix}.model"

def test_sentencepiece(model_path, test_texts):
    """Test SentencePiece tokenizer"""
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    
    print("\n" + "="*60)
    print("SENTENCEPIECE TEST")
    print("="*60)
    print(f"Vocab size: {sp.get_piece_size()}")
    
    for text in test_texts:
        tokens = sp.encode_as_pieces(text)
        ids = sp.encode_as_ids(text)
        decoded = sp.decode_pieces(tokens)
        
        print(f"\nText: {text}")
        print(f"Tokens: {tokens}")
        print(f"IDs: {ids}")
        print(f"Decoded: {decoded}")

def preprocess_file(input_path, output_path):
    print(f"Preprocessing {input_path} → {output_path}")
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            # Remove nulls
            line = line.replace("\x00", "")
            line = line.encode("utf-8", "ignore").decode("utf-8")
            line = line.strip()
            line = clean_text(line)
            # Normalize
            line = unicodedata.normalize("NFKC", line)
            
            if not line:
                continue
            
            if len(line) > 2000:
                continue
            
            line = merge_compounds(line)
            f_out.write(line + "\n")

if __name__ == "__main__":
    
    # Option 1: Train on all files (RECOMMENDED for better coverage)
    ALL_FILES = [
        os.path.join(os.getcwd(), "../dataset/general/others", f"hi_part_{i}.txt") for i in range(1, 6)
    ]
    VOCAB_SIZE = 24000
    MODEL_PREFIX = "../models/devanagari_sp"
    # Files are already processed, use them directly  
    PROCESSED_FILES = []

    for path in ALL_FILES:
        out_path = path.replace(".txt", "_processed.txt")
        preprocess_file(path, out_path)
        PROCESSED_FILES.append(out_path)

    # Choose one:
    model_path = train_sentencepiece_tokenizer(
        corpus_paths=PROCESSED_FILES,
        vocab_size=VOCAB_SIZE,
        model_prefix=MODEL_PREFIX,
        model_type="unigram",
    )
    model_path = MODEL_PREFIX + ".model"
    # Test
    test_texts = [
        "नमस्ते दुनिया",
        "यह एक परीक्षण है",
        "Hello world in Devanagari: नमस्ते",
        "def train_model(): pass",
        "मशीन लर्निंग मॉडल ट्रेनिंग",
        "अविश्वसनीयता",
        "स्वतंत्रता",
        "भारत एक महान देश है",
        "AI भविष्य बदल देगा",
        "मशीन लर्निंग मॉडल क्या है?",
    ]
    test_sentencepiece(model_path, test_texts)
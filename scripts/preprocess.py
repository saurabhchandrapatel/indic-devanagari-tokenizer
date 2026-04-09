import os
import regex as re
import sentencepiece as spm
import unicodedata

INPUT_DIR = "raw_data"
OUTPUT_DIR = "dataset_clean"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVANAGARI_CLUSTER = r'\p{Devanagari}(?:\p{M}|\u094D\p{Devanagari})*'
pattern = re.compile(
    f"({DEVANAGARI_CLUSTER}+|[a-zA-Z]+|\\d+|[^\\s])"
)
model_prefix = "./devanagari_tokenizer/devanagari_sp"

def panini_pre_tokenize(text):
    return [m.group(0) for m in pattern.finditer(text)]

# Rule 1: Don’t split suffixes

#Rule 2: Handle common compounds
COMPOUNDS = ["मशीन लर्निंग", "कृत्रिम बुद्धिमत्ता"]

def merge_compounds(text):
    if not text:
        return text
        
    for comp in COMPOUNDS:
        text = text.replace(comp, comp.replace(" ", "_"))
    
    return text

#Rule 3: Add optional Panini rules
SUFFIXES = [ 
    "ता",   # क्षमता, स्वतंत्रता
    "पन",   # बचपन
    "त्व",   # मानवत्व
    "कार",  # शिक्षक + कार
    "वादी", # समाजवादी
    "करण",  # निर्माण
    "शील",  # विनम्रशील
]
def panini_split(text):
    if not text:   # 👈 critical fix
        return []
    text = merge_compounds(text)
    tokens = [m.group(0) for m in pattern.finditer(text)]

    # merge suffixes (basic rule)
    merged = []
    for tok in tokens:
        if merged and tok in SUFFIXES:
            merged[-1] += tok
        else:
            merged.append(tok)
    
    return merged

def clean_text(line):
    # Remove weird unicode junk (keep Hindi + English + basic punctuation)
    line = re.sub(r'[^\u0900-\u097Fa-zA-Z0-9\s.,!?()\-\[\]:"\'/]', '', line)
    # Collapse spaces
    line = re.sub(r'\s+', ' ', line)
    return line.strip()

def is_valid_line(line):
    if len(line) < 5:
        return False
    if len(line) > 1500:
        return False

    # Ensure enough alphabetic content
    alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
    if alpha_ratio < 0.5:
        return False

    return True
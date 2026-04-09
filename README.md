# Indic Devanagari Tokenizer

A production-ready SentencePiece tokenizer designed for Devanagari-based languages such as Hindi and Sanskrit. This project focuses on clean data processing, efficient subword segmentation, and compatibility with modern LLM pipelines.

---

## 🚀 Overview

This repository provides a complete pipeline to:

* Build a high-quality tokenizer for Devanagari text
* Clean and preprocess large-scale Hindi/Sanskrit corpora
* Train a SentencePiece unigram tokenizer
* Prepare datasets for LLM training (1B+ parameter scale)

The tokenizer is optimized for:

* Hindi-first language modeling
* Sanskrit + Hindi hybrid datasets
* Indian philosophy corpora (Vedas, Upanishads, Gita)

---

## ✨ Features

* ✅ SentencePiece Unigram tokenizer
* ✅ Optimized for Devanagari Unicode range
* ✅ Handles Hindi + Sanskrit text effectively
* ✅ Clean text preprocessing pipeline
* ✅ Configurable vocabulary size (16k / 24k / 32k)
* ✅ Supports long token formation (compound words)
* ✅ Ready for LLM training workflows

---

## 📁 Project Structure

```
indic-devanagari-tokenizer/
├── dataset/
│   ├── general/
│   │   ├── news/
│   │   ├── wiki/
│   │   ├── books/
│   │
│   ├── philosophy/
│       ├── gita/
│       ├── upanishads/
│       ├── vedanta/
│
├── scripts/
│   ├── preprocess.py
│   ├── train_tokenizer.py
│
├── models/
│   ├── devanagari_sp.model
│   ├── devanagari_sp.vocab
│
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/indic-devanagari-tokenizer.git
cd indic-devanagari-tokenizer

pip install -r requirements.txt
```

---

## 🧹 Data Preprocessing

Clean and normalize raw text before training:

```bash
python scripts/preprocess.py
```

Key steps:

* Unicode normalization (NFKC)
* Removal of corrupted characters
* Sentence-level formatting
* Length filtering

---

## 🧠 Training the Tokenizer

```bash
python scripts/train_tokenizer.py
```

### Example Configuration

* Model Type: `unigram`
* Vocabulary Size: `24000`
* Character Coverage: `0.9995`
* Max Token Length: `24`

---

## 🔍 Example Tokenization

Input:

```
मशीन लर्निंग मॉडल ट्रेनिंग
```

Output:

```
['▁मशीन', '▁लर्निंग', '▁मॉडल', '▁ट्रेनिंग']
```

---

## 📊 Recommended Data Strategy

For best results:

* 70–80% general Hindi corpus (news, wiki, books)
* 20–30% domain-specific data (philosophy, scriptures)

Avoid:

* Noisy scraped content
* Mixed encoding text
* Unverified sources

---

## 🧪 Evaluation Tips

* Check token splits for compound Hindi words
* Monitor average tokens per sentence
* Inspect vocabulary for noise or artifacts

---

## 🧭 Roadmap

* [ ] Add dataset cleaning pipeline (deduplication + filtering)
* [ ] Support additional Indic scripts
* [ ] Integration with LLM training frameworks
* [ ] Tokenizer benchmarking tools

---

## 🤝 Contributing

Contributions are welcome. Focus areas:

* Dataset quality improvements
* Better preprocessing techniques
* Tokenization evaluation tools

---

## 📜 License

MIT License

---

## ⚠️ Disclaimer

This project is intended for research and development in Indic NLP. Ensure compliance with data licensing when using external corpora.

---

## ⭐ Acknowledgements

* Google SentencePiece
* Indic NLP community
* Open-source linguistic datasets

---

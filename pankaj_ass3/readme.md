
---link https://api.wandb.ai/links/cs24m029-indian-institute-of-technology-madras/2r3onzpy


# 📝 Assignment 3 — Transliteration System using RNNs (DA6401)

## 🔍 Problem Statement

This assignment focuses on building a character-level sequence-to-sequence transliteration system using Recurrent Neural Networks. The goal is to map a romanized input string (e.g., `ghar`) to its equivalent native script representation (e.g., `घर`) using the [Dakshina dataset](https://github.com/google-research-datasets/dakshina).

---

## Project Structure

```
.
├── config.py               # Configuration for model and training
├── dataset.py             # Data preprocessing and custom dataset
├── model.py               # TranslitModel that wraps encoder-decoder
├── model_components.py    # Encoder, Decoder, Attention, Beam search
├── trainer.py             # Training and evaluation logic
├── sweep.py               # W&B sweep configuration and launch
├── train.py               # Main entrypoint to train and evaluate model
└── README.md              # This file
```

---

##  Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/da6401_assignment3/partA
   cd da6401_assignment3/partA
   ```



2. Enable GPU support (if using Google Colab or local CUDA-enabled device).

---

##  Dependencies

Key libraries used:

* `torch` – model implementation
* `wandb` – experiment tracking
* `nltk` – BLEU score computation
* `matplotlib` – visualization
* `tqdm` – progress bars

---

##  Model Architecture

You can configure the following components in the `ModelConfig`:

* Encoder/Decoder types: `RNN`, `GRU`, `LSTM`
* Number of layers
* Embedding size, Hidden size
* Dropout
* Attention mechanism (optional)
* Bidirectional encoder (optional)

Two decoders are supported:

* **Vanilla Decoder** (`DecoderRNN`)
* **Attention-based Decoder** (`DecoderAttnRNN`)

---

##  Running the Model

### Train and Evaluate

```bash
python train.py --language marathi --apply_attention --logging
```

### Important Flags:

| Flag                                | Description                              |
| ----------------------------------- | ---------------------------------------- |
| `--language`                        | Language code (e.g., `marathi`, `hindi`) |
| `--apply_attention`                 | Use attention mechanism                  |
| `--encoder_name` / `--decoder_name` | Choose between `RNN`, `GRU`, `LSTM`      |
| `--beam_size`                       | Beam size for decoding                   |
| `--logging`                         | Enable Weights & Biases logging          |

---

## 📊 Hyperparameter Sweeps (W\&B)

To run a W\&B sweep:

```bash
python sweep.py
```

**Parameters searched:**

* `embedding_size`, `hidden_size`
* `encoder/decoder_num_layers`
* `dropout_p`
* `learning_rate`, `teacher_forcing_p`
* `encoder_name`, `decoder_name`
* `apply_attention`

**Logged Metrics:**

* `train_loss`, `valid_loss`
* `train_acc`, `valid_acc`
* `test_bleu`

---



##  Visualizations

* 3×3 Attention heatmap grid: `attention_maps.png`
* W\&B sweep plots:

  * Accuracy vs. Run
  * Parallel Coordinates Plot
  * Correlation Matrix

---




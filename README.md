# DLRL_assignment

This repository contains a collection of **cleaned, modernized, and beginner-friendly DL/RL codes**.
The focus is on:
* Correct APIs and shapes
* Reproducibility
* Readability and reusability
* Minimal but meaningful improvements over original examples

## Contents

1. **AlexNet**
2. **Deep Reinforcement Learning**
3. **LSTM**
4. **RNN**
5. **Tic-Tac-Toe Reinforcement Learning (Self-Play)**

## 1. AlexNet
## Original Implementation
Basic AlexNet architecture with simple Conv2D layers
Manual configuration for ImageNet (1000 classes)
No batch normalization
No training pipeline or dataset handling

## Improvements made
**File:** `alexnet.ipynb`

A modernized AlexNet-style CNN implemented as a clean `tf.keras.Sequential` subclass.

### Key improvements vs. original AlexNet

* **Configurable first convolution stride**

  * `first_conv_stride` (default `2`)
  * Set to `4` to match the original AlexNet behavior.
* **Batch Normalization after every Conv2D**

  * Improves training stability and convergence.
* **Separate Activation layers**

  * `BatchNormalization()` → `Activation('relu')` (best practice).
* **He initialization & L2 regularization**

  * `kernel_initializer='he_normal'`
  * `kernel_regularizer=l2(weight_decay)` (default `5e-4`)
* **Removed redundant Conv biases**

  * `use_bias=False` when followed by BatchNorm.
* **Pooling with `padding="same"`**

  * Prevents accidental spatial collapse on odd-sized inputs.
* **Reusable class API**

  ```python
  AlexNet(
      input_shape=(224, 224, 3),
      num_classes=1000,
      first_conv_stride=2,
      weight_decay=5e-4
  )
  ```

## 2. Deep Reinforcement Learning

## Original Implementation
Traditional Q-learning with Q-matrix
Simple graph navigation problem
Basic visualization
**File:** `deep reinforcement learningh.ipynb`

## Improvements made

* **Bug fix:** removed incorrect global variable usage when sampling actions.
* **Modern NumPy usage:** replaced `np.matrix` with `np.ndarray`.
* **Reproducible layouts:** `networkx.spring_layout(seed=...)`.
* **Clear structure:** small, testable functions:

  * `build_graph`
  * `build_reward_matrix`
  * `available_actions`
  * `sample_next_action`
  * `train_q`
  * `extract_path`
  * `update_with_environment`
* **Environment-aware biasing**

  * Tracks “police” and “drug” traces to influence action choice.
* **Safety & UX**

  * Random seeds
  * Fallback logic
  * Informative prints and plots

## 3. LSTM 

**File:** `lstm.ipynb`

## Original Implementation
Single LSTM layer (10 units)
Basic sequence prediction
Hardcoded dataset path (Windows)
Simple visualization

### Key improvements made

* Consistent `tensorflow.keras` imports
* Correct LSTM input shape: `(samples, timesteps, features)`
* Helper functions:

  * `safe_load_series()`
  * `create_sequences()`
* CSV fallback: generates synthetic data if file is missing
* Reproducible seeds
* `EarlyStopping` to avoid overtraining
* Minimal dependencies

### How to run

With CSV:

```bash
python simple_lstm_passengers.py path/to/data.csv
```

Without CSV (synthetic data):

```bash
python simple_lstm_passengers.py
```

## 4. RNN

**File:** `rnn.ipynb`

A readable character-level RNN for next-character prediction.

## Original Implementation
SimpleRNN layer (50 units)
Small text corpus
Fixed sequence length (5)
Greedy decoding (argmax)

### Improvements made

* Uses `tensorflow.keras` only
* Clear data pipeline with `to_categorical`
* Small, focused helper functions:

  * `build_vocab`
  * `create_dataset`
  * `build_model`
  * `generate_text`
* Temperature-based sampling for generation
* Robust seed handling
* Reproducible defaults

## 5. Tic-Tac-Toe Reinforcement Learning (Improved)

**File:** `tic_tac_toe_rl.ipynb`

A simple RL demo where two agents learn Tic-Tac-Toe via self-play, then you can play against the trained agent.

## Original Implementation
Traditional Q-learning with state-value dictionary
Simple exploration strategy
Basic text-based interface

### Key fixes & improvements made

* Correct initial player symbol
* Robust CLI parsing (works in notebooks)
* Safe policy save/load using `pathlib`
* Reduced default training rounds (faster demos)
* Clear training progress output
* Cleaner code structure and type hints

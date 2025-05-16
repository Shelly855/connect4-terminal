# Connect 4 – Terminal Version

An **original terminal-based version** of a [Connect 4 game](https://github.com/Shelly855/connect4-ai), developed before transitioning to a GUI.

---

## Table of Contents

- [Connect 4 – Terminal Version](#connect-4--terminal-version)
- [Supported Modes](#supported-modes)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Files](#files)
- [Notes](#notes)
- [References](#references)

---

### Supported Modes
- Human vs Human
- Human vs AI (Random, Smart, Minimax, Basic ML, Minimax-Trained ML)
- AI vs AI

---

## Requirements

- Python 3
- Recommended: VS Code with the Python extension
- `scikit-learn` and `colorama`

> Install packages with:
> ```
> pip install scikit-learn colorama
> ```

## How to Run

1. Open a terminal in this folder.
2. Run the script with:
   ```bash
   python game.py

---

## Files

- `game.py` – The full CLI-based game logic  
- `ml_agent.pkl` – ML model trained using real game data (UCI dataset)
- `ml_agent_minimax.pkl` – ML model trained using data generated from the minimax algorithm
- `test_ml_prediction` - A simple script to test if ML agent predicts correctly (not required for running the game)

---

## Notes

- This version lacks a visual interface but demonstrates core functionality.  
- It's included as part of the project history, showing early development stages before the GUI was introduced.  

---

## References

- Keith Galli’s Connect 4 AI (GitHub)
  - https://github.com/KeithGalli/Connect4-Python/blob/master/connect4_with_ai.py
  - Used as a reference for structuring minimax, alpha-beta pruning, and evaluation heuristics
- Science Buddies – “Connect 4 AI Player using Minimax Algorithm with Alpha-Beta Pruning: Python Coding Tutorial” (YouTube):
  - https://www.youtube.com/watch?v=rbmk1qtVEmg
  - Used as a reference for alpha-beta pruning
- Connect 4 dataset from UCI Machine Learning Repository
  - https://archive.ics.uci.edu/dataset/26/connect+4
  - Used to train the basic ML agent

# Connect 4 – Terminal Version

This is the **original terminal-based version** of the Connect 4 game, developed before transitioning to a GUI.

It supports:
- Human vs Human
- Human vs AI (Random, Smart, Minimax, ML)
- AI vs AI

---

## Requirements

- Python 3.10 or later  
- Recommended: VS Code with the Python extension
- `scikit-learn` and `colorama` (typically preinstalled)

> If needed, install missing packages with:
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
- `ml_agent.pkl` – Trained ML model (basic)  
- `ml_agent_minimax.pkl` – Trained ML model using minimax-generated data  

---

## Notes

- This version lacks a visual interface but demonstrates core functionality.  
- It's included as part of the project history, showing early development stages before the GUI was introduced.  
- All AI agents and minimax logic work in this version.  

---

## References

- Keith Galli’s Connect 4 AI (GitHub)
  - https://github.com/KeithGalli/Connect4-Python/blob/master/connect4_with_ai.py
  - Used as a reference for structuring minimax, alpha-beta pruning, and evaluation heuristics
- Science Buddies – “Connect 4 AI Player using Minimax Algorithm with Alpha-Beta Pruning: Python Coding Tutorial” (YouTube):
  - https://www.youtube.com/watch?v=rbmk1qtVEmg
  - Used as a reference for alpha-beta pruning
- Connect 4 Dataset
  - https://archive.ics.uci.edu/dataset/26/connect+4
  - Used to train the basic ML agent

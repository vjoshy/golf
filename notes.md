Folder structure is as follows:

```
├── .git/                     # Git repository 
├── src/                      
│   ├── golf.py
│   ├── golf_auto.py
│   ├── golf_q.py           # Q-learning tabular train/test
│   ├── golf_q_mod.py
│   ├── golf_DQN_extended.py
│   ├── dqn_evil.py
│   ├── dqn_test.py         # DQN testing code
│   └── dqn_golf.py         # DQN training code
│
├── models/                   # this folder for saved models
│   ├── dqn_10k.pth       # Trained for 10k episodes
│   |── dqn_100k.pth      # Trained for 100k episodes (example)

│
├── data/                     # For any game data or statistics
│   └── ~.csv
│
├── notes.md                  # Your notes file
└── README.md                 # Add documentation about your project
```

DQN on golf game
- To train DQN model run `python src/dqn_golf.py`
- To test against smart (evil) player run `python src/dqn_test.py`

`dqn_golf.py` has the best the NN architecture. In this code, the trained model is being saved as `.pth` file in `models` folder, which is used in `dqn_test.py` file for testing. By compartmentalizing our procedures like this we can work on testing and training more efficiently.

We can also saved multiple versions fo trained models and compare them. For eg: Models trained on 10k episodes vs 100k or 1M episodes. Or perhaps comparing different combinations of hyperparameters. I recommend renaming the model files for different versions.


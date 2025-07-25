# Airplane Boarding Simulation

## Overview

This project simulates the process of boarding an airplane using a custom OpenAI Gymnasium environment. It models passengers, seats, and boarding lines, and supports reinforcement learning experiments to optimize boarding strategies. The environment is compatible with Stable Baselines3 and SB3-Contrib for RL training.

## Features
- Custom Gymnasium environment for airplane boarding
- Models passengers, seats, lobby, and aisle boarding line
- Supports random and RL-based boarding strategies
- Logging and error handling for robust experimentation
- Input validation for all critical operations
- Easily extensible for new boarding policies or passenger behaviors

## Setup Instructions

1. **Create and activate a virtual environment**
   ```sh
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Run a Random Boarding Simulation
```sh
python main.py
```
This will run a simulation using random actions and print the step-by-step boarding process in the terminal.

### Train a Reinforcement Learning Agent
```sh
python agent.py
```
This will start training a MaskablePPO agent using the custom environment. Training progress and models will be saved in the `logs/` and `models/` directories.

### Test a Trained Agent
Uncomment the `test("best_model")` line in `agent.py` and run:
```sh
python agent.py
```

## Dependencies
The main dependencies are pinned in `requirements.txt`. Key packages include:
- gymnasium
- numpy
- torch
- stable-baselines3
- sb3-contrib
- pandas
- matplotlib

(See `requirements.txt` for the full list and versions.)

## Project Structure
```
/airplane-boarding-simulation/
    airplane_boarding.py   # Main environment and logic
    main.py                # Random simulation runner
    agent.py               # RL training/testing script
    README.md
requirements.txt           # All dependencies
venv/                      # Virtual environment (not tracked)
```

## Contributing
- Fork the repo and create a feature branch
- Add tests for new features or bugfixes
- Ensure code passes linting and runs in a clean venv
- Open a pull request with a clear description

## License
MIT License

---

Feel free to update this README with more usage, research, or contribution details!
 

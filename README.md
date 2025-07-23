# ✈️ Airplane Boarding Simulation

![Python](https://img.shields.io/badge/Language-Python-blue.svg)
![Open Source](https://img.shields.io/badge/Open%20Source-Yes-brightgreen.svg)
![GSSoC](https://img.shields.io/badge/GSSoC-2025-orange)

## 📌 About the Project

This project simulates the real-world process of passengers boarding an airplane using various strategies (like back-to-front, random, window-middle-aisle, etc.).

The goal is to compare how these strategies affect **boarding time**, **efficiency**, and **crowd dynamics**.

🧠 This is a great tool for:
- Understanding **optimization**
- Simulating **real-life scenarios**
- Learning **Python OOPs** and **simulation modeling**

---

## ✨ Features

- Multiple boarding strategies (easily extendable)
- Visual feedback using terminal
- Modular and beginner-friendly code
- Gymnasium-style simulation environment

---

## 🚀 Getting Started

### 🔧 Prerequisites

- Python 3.7+
- pip (Python package installer)

### 🖥️ Installation

# Clone the repository
git clone https://github.com/18Prachi/airplane-boarding-simulation.git

cd airplane-boarding-simulation

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
If requirements.txt is missing, install manually:

bash
Copy
Edit
pip install numpy matplotlib gymnasium
🕹️ How to Run
bash
Copy
Edit
python main.py
You’ll see step-by-step boarding simulation printed in the terminal with rewards and actions.

🧩 Project Structure
bash
Copy
Edit
 airplane_boarding/         # Core logic & simulation environment

 env.py                 # Main environment (Gym)

 passenger.py           # Passenger logic & status

 row.py                 # Seat & row management

 main.py                    # Entry point to run the simulation

 README.md                  # You're here!

 CONTRIBUTING.md            # Contribution guidelines (optional but recommended)
 
🤝 Contributing

We welcome beginners and experts alike!

🍴 Fork this repo

👯 Clone your fork

🔧 Make your changes in a new branch

✅ Commit and push

📩 Create a Pull Request!

For detailed help, check CONTRIBUTING.md (or ask us on Discussions/Issues)

🧠 Good First Issues
Look for issues labeled:

good first issue

beginner friendly

documentation

Feel free to comment "I want to work on this" and maintainers will guide you 😊

🏆 GSSoC 2025
This project is part of GirlScript Summer of Code 2025.
We are committed to creating a helpful, inclusive, and beginner-friendly space.

Don't hesitate to ask questions. Every contribution counts!

👨‍💻 Maintainers

Mentor: SOHAM GHOSH 

📄 License
This project is licensed under the MIT License.

🙌 Acknowledgements
Inspired by real-world optimization problems

Built using Python and Gymnasium
 

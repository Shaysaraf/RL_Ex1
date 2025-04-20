# RL_Ex1
Reinforcment Learning Course - Ex1 

## Students
- Daniel Soref 
- Shay Saraf

---

## Description of Source/Input Files
1. **`gym_cart_pole.py`**
   - Implements the CartPole environment using OpenAI Gym.
   - Includes a random search algorithm to find the best weights for solving the CartPole problem.
   - Contains the `CustomAgent` class, `run_episode` function, and evaluation logic.

2. **`mnist.py`**
   - Implements MNIST classification using PyTorch.
   - Includes three models:
     - Logistic Regression with SGD
     - Logistic Regression with Adam
     - Deep Neural Network with Adam
   - Contains training and evaluation logic for each model.

3. **`bko_language.py`**
   - Implements the BKO language, a custom language for solving specific tasks.
   - Requires only NumPy for execution.
   - Contains functions and logic for parsing and executing BKO language commands.

4. **`data/`**
   - Contains any required datasets or preprocessed files for the MNIST classification task.

5. **`README.md`**
   - This file. Provides an overview of the project, file descriptions, commands, and output details.

---
## Commands to Run the Code
**Command:**
```bash
python RL_Ex1/gym_cart_pole.py
python RL_Ex1/bko_language.py
python RL_Ex1/mnist.py
```
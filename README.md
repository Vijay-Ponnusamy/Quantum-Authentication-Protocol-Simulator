# Quantum Authentication Protocol Simulator

***A Streamlit-based interactive simulator for visualizing and analyzing quantum authentication protocols.***

---

## Project Overview

Quantum communication and authentication are increasingly important in securing future networks against **quantum-enabled attacks**.  
The **Quantum Authentication Protocol Simulator** provides a **visual, interactive framework** to explore how **quantum challenge-response and entanglement-based protocols** can be used to authenticate parties securely.

This simulator demonstrates **quantum concepts** such as **qubit superposition, basis encoding, and entanglement** for authentication purposes.  
It provides an intuitive GUI to experiment with protocol parameters, observe authentication success rates, and visualize qubit states in real time.

---

## Key Features

 **BB84 Challenge-Response** – Simulates authentication using BB84 qubit encoding and sifting.    
 **Interactive Visualization** – Displays qubit bases, measurement outcomes, mismatch rates, and response tokens.  
 **User-Friendly Streamlit GUI** – Real-time interactive simulation of quantum authentication rounds.  
 **Report & Log Export** – Save simulation data, mismatch statistics, and response tokens.  
 **Educational Purpose** – Ideal for research, learning, and demonstration of quantum cryptography principles.  

---

## Tech Stack

| Component           | Technology / Library              |
| ------------------- | -------------------------------- |
| Frontend Interface  | Streamlit                        |
| Quantum Framework   | PennyLane / Qiskit                |
| Data Handling       | NumPy                             |
| Visualization       | Matplotlib, Plotly                |
| Documentation       | README (Markdown)                 |

---

## Installation & Setup

1. **Clone this repository**

   ```bash
   git clone (https://github.com/Vijay-Ponnusamy/Quantum-Authentication-Protocol-Simulator)
   cd QuantumAuthenticationSimulator
2. **Run the Simulator**

   ```bash
   streamlit run quantum_auth_simulator.py

3. **Install dependencies**

   ```bash
   pip install streamlit numpy matplotlib plotly
   pip install pennylane qiskit
   pip install seaborn

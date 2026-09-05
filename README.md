# DoublePendulumPred
This repository experiments in predicting the motion of a double pendulum.  
A double pendulum system is extremely chaotic to small changes in initial conditions making it difficult to predict in the long term.  
The model trained is compared to basic simulations.

## Research Question
To what extent can the motion of a double pendulum be predicted using a Transformer based model?

## Physics Model
The double pendulum system uses langrangian mechanis to derive equations.  
The system is non linear and uses second degree differential equations.

### Equation for first pendulum

$$
(m_1+m_2)l_1\ddot{\theta}_1
+
m_2l_2\ddot{\theta}_2\cos(\theta_1-\theta_2)
+
m_2l_2\dot{\theta}_2^2\sin(\theta_1-\theta_2)
+
(m_1+m_2)g\sin(\theta_1)
=0
$$

### Equation for second pendulum

$$
m_2l_2\ddot{\theta}_2
+
m_2l_1\ddot{\theta}_1\cos(\theta_1-\theta_2)
\-
m_2l_1\dot{\theta}_1^2\sin(\theta_1-\theta_2)+m_2g\sin(\theta_2)
=0
$$

## Physics Simulation
Upcoming...

## Model
Transformer based architecture.

## Dataset Generation
Upcoming...

## Training Setup
Upcoming...

## Results
Upcoming...

## Discussion
Upcoming...

## Limitations
Upcoming...

## How to Run the Project
pip install -r requirements.txt  
  
python data/generate_data.py  
python models/train.py  
python models/evaluate.py

## Project Structure
physics/  
models/  
data/  
results/  
main.py  
README.md

## Requirements
numpy  
torch
numba


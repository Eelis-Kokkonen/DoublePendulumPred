# DoublePendulumPred
This repository experiments in predicting the motion of a double pendulum.  
A double pendulum system is extremely chaotic to small changes in initial conditions making it difficult to predict in the long term.  
The model trained is compared to basic simulations.

## Research Question
To what extent can the motion of a double pendulum be predicted using a Transformer based model?

## Physics Model
The double pendulum system uses langrangian mechanis to derive equations.  
The system is non linear and uses second degree differential equations.  

(m1+m2)l1''1+m2l2''2cos(1-2)+m2l2'22sin(1-2)+(m1+m2)gsin1=0


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


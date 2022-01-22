# Machine Learning Sandbox
This repo contains various projects related to machine learning
including reinforcement, supervised, and unsupervised learning.  
To install packages required to run the programs in this sandbox, run 
```
pip install -r requirements.txt
```
## Reinforcement Learning (r_learning)
---
### 1_grid_world
Contains two algoritms, dynamic value iteration and Monte Carlo algorithm, to solve a grid world problem. The subfolder contains a README.md describing the game in further detail
Run dynamic.py or monte_carlo.py to view the iterations of the algorithm.

### 2_taxi_open_ai
Performs Q Learning to solve the taxi route problem. 
```
(West)
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
```
The following is the game rule written by the website at [Gym OpenAI](https://gym.openai.com/envs/Taxi-v2/)
>There are 4 locations (labeled by different letters) and your job is to pick up the passenger at one location and drop him off in another. You receive +20 points for a successful dropoff, and lose 1 point for every timestep it takes. There is also a 10 point penalty for illegal pick-up and drop-off actions.

To see the result, run in command
```
python replay.py [checkpoint path]
```
to use a pretrained checkpoint, use ```checkpoints\Taxi-v3_50000.npy```  
To train the policy netowrk, run
```
python taxi.py
```
### 3_walking_ARS
Performs augmented random search to walk a two legged bot across a field

<video width="320" height="240" controls>
  <source src="https://i.imgur.com/l0fjQCz.mp4" type="video/mp4">
</video>

To see the result/train, run in command
```
python ARS.py 
```

### pong_qlearning
Performs q learning to play pong against a simple ai.
To see the result/train, run in command
```
python qlearn.py 
```
## Supervised and Unsupervised Learning (supervised_unsupervised)
---
## TODO
# Grid World
Contains two algorithms to display value and policy lookup tables for a grid world.
```
---------------------------
  0  |  0  |  0  |  1  |
---------------------------
  0  |     |  0  |  -1 |
---------------------------
  0  |  0  |  0  |  0  |
```
## Game Rule
---
The grid above shows the reward to reach each square. The character cannot move to the square containing no value. If the character moves against the wall, it stays in the same square. For example, if the character in the top left corner moves up or left, it stays in the same square.
Game is terminated when the player reaches the squares with the value of 1 or -1.  
The goal is to determine the optimal directions so that regardless of where the character starts, it will take the path so it receives the highest reward.  

The game contains two parameters.
In dynamic.py and monte_carlo.py, there is a line in the last code block to change parameters of the game
```
g = standard_grid(obey_prob=0.8, step_cost=-0.0001)
```
obey_prob sets the probability to determine the probability that the character will obey the direction of the policy. For example, for obey_prob=0.8, the character will obey the instructions "U" (up) or "D" (down) 80% of the time, but will have 10% chance of moving left and 10% chance of moving right. Similarly, the character will obey the instructions "L" (left)) or "R" (right) 80% of the time, but will have 10% chance of moving up and 10% chance of moving down. (higher obey_prob leads the algorithm to find the safest path which may be longer)

step_cost sets the cost of each action. (higher step_cost leads the algorithm to find the shortest but riskier path)
## Details
---
The algorithms 
dynamic.py uses dynamic programming, and monte_carlo.py uses monte carlo reinforcement learning algorithm to solve the problem above.  
## Usage
---
Run using 
```
python (dynamic.py | monte_carlo.py)
```
to display the iteration results of its algorithms. dynamic.py also has the option of using value iteration by removing the ocmment on line 103 and commenting out line 104.
```
---------------------------
 0.78| 0.88| 1.00| 0.00|
---------------------------
 0.69| 0.00| 0.89| 0.00|
---------------------------
 0.61| 0.60| 0.73| 0.62|
---------------------------
  R  |  R  |  R  |     |
---------------------------
  U  |     |  U  |     |
---------------------------
  U  |  R  |  U  |  L  |
```
The two tables shows the resulting value and policy table after running monte_carlo.py algorithm.  
The top table shows the value of each square and the bottom table shows direction the player should move in each square. 

### TODO
Make it available to change the parameters of the game in a single command. Change the option between value iteration and policy iteration with a command parameter.

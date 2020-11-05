## CS 747 Programming Assignment 3: Windy Gridworld Problem

This is an implementation of Example 6.5, Exercise 6.9 and Exercise 6.10 from the book [Reinforcement Learning: An Introduction by Andrew Barto and Richard S. Sutton](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf "Reinforcement Learning: An Introduction"). This work also compares the performance of Sarsa, Q-Learning and Expected Sarsa with Full Bootstrapping.  

<!-- ![](images/gridworld.png) -->
<img src="images/gridworld.png" alt="gridworld" width="400"/>

The following tasks have been implemented:  
-  **Task 2**: Sarsa(0) with constant wind and 4 moves (baseline)  
-  **Task 3**: Sarsa(0) with constant wind and King's moves (8 moves total)  
-  **Task 4**: Sarsa(0) with stochastic wind and King's moves  
-  **Task 5**: Comparision of Sarsa, Q-Learning and Expected Sarsa with Full Bootstrapping  


To the run the program on your machine, navigate to this folder in bash/cmd and run:
```bash
python main.py --task i # i is the task number from {2,3,4,5}
# or
python main.py # runs task 5 by default
```
This will run the agent for 200 episodes and generate the plot of episodes vs cumulative time-steps.  
Hyper Parameters: 
```python
alpha = 0.5 # learning rate
epsilon = 0.1 # for epsilon-greedy choice
gamma = 1 # no discounting 
```
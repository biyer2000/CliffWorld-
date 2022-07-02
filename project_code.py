import numpy as np
import random
import matplotlib.pyplot as plt

from numpy.core.fromnumeric import argmax
#start_state = (0, 0, 0)
class GridWorld:
    def __init__(self, discount, alpha, num_steps, epsilon):
        self.discount = discount
        self.alpha = alpha
        self.num_steps = num_steps
        self.epsilon = epsilon
        self.rewards = [[-1, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
        self.rewards_2 = [[-1, -1, -1, -1, -1, -1, -1, -1,-1,-1,-1,-1,-1],[-1, -1, -1, -1, -1, -1, -1, -1,-1,-1,-1,-1,-1],[-1, -1, -1, -1, -1, -1, -1, -1,-1,-1,-1,-1,-1],[-1, -1, -1, -1, -1, -100, -100, -100,-100,-100,-100,-100,-1]
        ,[-1, -1, -1, -1, -1, -1, -1, -1,-1,-1,-1,-1,-1],[-1, -1, -1, -1, -1, -1, -1, -1,-1,-1,-1,-1,-1],[-1, -1, -1, -1, -1, -1, -1, -1,-1,-1,-1,-1,-1]]


   

def compute_Q(episodes, bonus, q_or_reward):
    q_values = np.ones((48, 4))
    if bonus:
        q_values = q_values * -1000

    grid = GridWorld(1, 0.85, 0.8, 0.1)
    rewards = []
    for i in range(episodes):


        finished = False
        current_state = (0,0)
        new_cord = (0,0)

        x = current_state[0]
        y = current_state[1]
        state = 12 * y + x
        #converts x and y cordinate to state from 0 to 47
        actions = [0,1, 2, 3]
        steps = 0
        total_reward = 0

        while not finished:
            #while we have not reached an end state
            '''if i == episodes - 1:
                print("y, x:", new_cord[1], new_cord[0])
                print("state: ", state)
                print("q value up: ", q_values[state][3])
                print("q value down: ", q_values[state][1])
                print("q value left: ", q_values[state][2])
                print("q value right: ", q_values[state][0])
                print("steps:", steps)'''
            if steps > 0:
                #not in the first step extract x and y cordinates
                x = new_cord[0]
                y = new_cord[1]
            if state == 11:
                finished = True
                rewards.append(total_reward)
                break
            '''
            if state >= 44 :
                actions = [0, 1,2]
            if state <= 3:
                actions = [0, 1, 3]
            if state % 3 == 0:
                actions = [1, 2, 3]
            if state % 4 == 0:
                actions = [0, 2, 3]
            '''
            if np.random.random() < grid.epsilon:
                action = random.choice(actions)
            else:
                action = np.argmax(q_values[state])
            if action == 0:
                if x == 11:
                    #check for edge region and if so keep the same x and y cordinates
                    x_n = x
                    y_n = y
                else:
                    y_n = y
                    x_n = x + 1
               
            if action == 1:
                if y == 0:
                    x_n = x
                    y_n = y
                else:
                    y_n = y - 1
                    x_n = x
               
            if action == 2:
                if x == 0:
                    x_n = x
                    y_n = y
                else:
                    y_n = y
                    x_n = x - 1
            if action == 3:
                if y == 3:
                    x_n = x
                    y_n = y
                else:
                    y_n = y + 1
                    x_n = x
            new_state = 12 * y_n + x_n
           
            reward = grid.rewards[y][x]
            q_values[state][action] += grid.alpha * (reward + grid.discount * np.max(q_values[new_state]) - q_values[state][action])
            steps += 1

            if new_state > 0 and new_state < 11:
                x_n = 0
                y_n = 0
                reward = reward - 100
           
            total_reward += reward

            if steps == 250:
                #if we have completed 250 steps
                rewards.append(total_reward)
                finished = True
           
            state = new_state
            new_cord = (x_n, y_n)
    #print(rewards)
    if q_or_reward == 'q':
        return(q_values)
    else:
        return(rewards)
    #print(q_values[13][0], q_values[13][3])

def sarsa(episodes, bonus, q_or_reward):
    q_values = np.ones((48, 4))
    if bonus:
        q_values = q_values * -1000
    grid = GridWorld(1, 0.8, 0.8, 0.1)
    rewards = []
    for i in range(episodes):


        finished = False
        current_state = (0,0)
        new_cord = (0,0)

        x = current_state[0]
        y = current_state[1]
        state = 12 * y + x
        #converts x and y cordinate to state from 0 to 47
        actions = [0,1, 2, 3]
        steps = 0
        total_reward = 0

        while not finished:
            #while we have not reached an end state
            '''if i == episodes - 1:
                print("y, x:", new_cord[1], new_cord[0])
                print("state: ", state)
                print("q value up: ", q_values[state][3])
                print("q value down: ", q_values[state][1])
                print("q value left: ", q_values[state][2])
                print("q value right: ", q_values[state][0])
                print("steps:", steps)'''
            if steps > 0:
                #not in the first step extract x and y cordinates
                x = new_cord[0]
                y = new_cord[1]
            if state == 11:
                finished = True
                rewards.append(total_reward)
                break
            '''
            if state >= 44 :
                actions = [0, 1,2]
            if state <= 3:
                actions = [0, 1, 3]
            if state % 3 == 0:
                actions = [1, 2, 3]
            if state % 4 == 0:
                actions = [0, 2, 3]
            '''
            if np.random.random() < grid.epsilon:
                action = random.choice(actions)
            else:
                action = np.argmax(q_values[state])
            if action == 0:
                if x == 11:
                    #check for edge region and if so keep the same x and y cordinates
                    x_n = x
                    y_n = y
                else:
                    y_n = y
                    x_n = x + 1
           
            if action == 1:
                if y == 0:
                    x_n = x
                    y_n = y
                else:
                    y_n = y - 1
                    x_n = x
           
            if action == 2:
                if x == 0:
                    x_n = x
                    y_n = y
                else:
                    y_n = y
                    x_n = x - 1
            if action == 3:
                if y == 3:
                    x_n = x
                    y_n = y
                else:
                    y_n = y + 1
                    x_n = x
            new_state = 12 * y_n + x_n
           
            reward = grid.rewards[y][x]
            if np.random.random() < grid.epsilon:
                a = random.choice(actions);
            else:
                a = np.argmax(q_values[new_state])
            q_values[state][action] += grid.alpha * (reward + grid.discount * q_values[new_state][a] - q_values[state][action])
            steps += 1

            if new_state > 0 and new_state < 11:
                x_n = 0
                y_n = 0
                reward = reward - 100
           
            total_reward += reward

            if steps == 250:
                #if we have completed 250 steps
                rewards.append(total_reward)
                finished = True
       
            state = new_state
            new_cord = (x_n, y_n)
    #print(rewards)
    if q_or_reward == 'q':
        return(q_values)
    else:
        return(rewards)
    #print(q_values[13][0], q_values[13][3])
def compute_Q_2(episodes):
    q_values = np.zeros((91, 4))
    grid = GridWorld(1, 0.85, 0.8, 0.1)
    #print(grid.rewards_2[3][12])
    rewards = []
    for i in range(episodes):


        finished = False
        current_state = (0,3)
        new_cord = (0,0)

        x = current_state[0]
        y = current_state[1]
        state = 13 * y + x
        #converts x and y cordinate to state from 0 to 47
        actions = [0,1, 2, 3]
        steps = 0
        total_reward = 0

        while not finished:
            #while we have not reached an end state
            '''if i == episodes - 1:
                print("y, x:", new_cord[1], new_cord[0])
                print("state: ", state)
                print("q value up: ", q_values[state][3])
                print("q value down: ", q_values[state][1])
                print("q value left: ", q_values[state][2])
                print("q value right: ", q_values[state][0])
                print("steps:", steps)'''
            if steps > 0:
                #not in the first step extract x and y cordinates
                x = new_cord[0]
                y = new_cord[1]
            if state == 12 or state == 90:
                finished = True
                rewards.append(total_reward)
                break
            '''
            if state >= 44 :
                actions = [0, 1,2]
            if state <= 3:
                actions = [0, 1, 3]
            if state % 3 == 0:
                actions = [1, 2, 3]
            if state % 4 == 0:
                actions = [0, 2, 3]
            '''
            if np.random.random() < grid.epsilon:
                action = random.choice(actions)
            else:
                action = np.argmax(q_values[state])
            if action == 0:
                if x == 12:
                    #check for edge region and if so keep the same x and y cordinates
                    x_n = x
                    y_n = y
                else:
                    y_n = y
                    x_n = x + 1
               
            if action == 1:
                if y == 0:
                    x_n = x
                    y_n = y
                else:
                    y_n = y - 1
                    x_n = x
               
            if action == 2:
                if x == 0:
                    x_n = x
                    y_n = y
                else:
                    y_n = y
                    x_n = x - 1
            if action == 3:
                if y == 6:
                    x_n = x
                    y_n = y
                else:
                    y_n = y + 1
                    x_n = x
            new_state = 13 * y_n + x_n
            #print(y)
            #print(x)
            #print("reward")
           
            reward = grid.rewards_2[y][x]
            q_values[state][action] += grid.alpha * (reward + grid.discount * np.max(q_values[new_state]) - q_values[state][action])
            steps += 1

            if new_state > 43 and new_state < 51:
                x_n = 0
                y_n = 0
                reward = reward - 100
           
            total_reward += reward

            if steps == 250:
                #if we have completed 250 steps
                rewards.append(total_reward)
                finished = True
           
            state = new_state
            new_cord = (x_n, y_n)
    #print(rewards)
   
    return(rewards)
def sarsa2(episodes):
    q_values = np.zeros((91, 4))
    grid = GridWorld(1, 0.8, 0.8, 0.1)
    rewards = []
    for i in range(episodes):


        finished = False
        current_state = (0,3)
        new_cord = (0,0)

        x = current_state[0]
        y = current_state[1]
        state = 13 * y + x
        #converts x and y cordinate to state from 0 to 47
        actions = [0,1, 2, 3]
        steps = 0
        total_reward = 0

        while not finished:
            #while we have not reached an end state
            '''if i == episodes - 1:
                print("y, x:", new_cord[1], new_cord[0])
                print("state: ", state)
                print("q value up: ", q_values[state][3])
                print("q value down: ", q_values[state][1])
                print("q value left: ", q_values[state][2])
                print("q value right: ", q_values[state][0])
                print("steps:", steps)'''
            if steps > 0:
                #not in the first step extract x and y cordinates
                x = new_cord[0]
                y = new_cord[1]
            if state == 12 or state == 90:
                finished = True
                rewards.append(total_reward)
                break
            '''
            if state >= 44 :
                actions = [0, 1,2]
            if state <= 3:
                actions = [0, 1, 3]
            if state % 3 == 0:
                actions = [1, 2, 3]
            if state % 4 == 0:
                actions = [0, 2, 3]
            '''
            if np.random.random() < grid.epsilon:
                action = random.choice(actions)
            else:
                action = np.argmax(q_values[state])
            if action == 0:
                if x == 12:
                    #check for edge region and if so keep the same x and y cordinates
                    x_n = x
                    y_n = y
                else:
                    y_n = y
                    x_n = x + 1
           
            if action == 1:
                if y == 0:
                    x_n = x
                    y_n = y
                else:
                    y_n = y - 1
                    x_n = x
           
            if action == 2:
                if x == 0:
                    x_n = x
                    y_n = y
                else:
                    y_n = y
                    x_n = x - 1
            if action == 3:
                if y == 6:
                    x_n = x
                    y_n = y
                else:
                    y_n = y + 1
                    x_n = x
            new_state = 13 * y_n + x_n
           
            reward = grid.rewards_2[y][x]
            if np.random.random() < grid.epsilon:
                a = random.choice(actions);
            else:
                a = np.argmax(q_values[new_state])
            q_values[state][action] += grid.alpha * (reward + grid.discount * q_values[new_state][a] - q_values[state][action])
            steps += 1

            if new_state > 43 and new_state < 51:
                x_n = 0
                y_n = 0
                reward = reward - 100
           
            total_reward += reward

            if steps == 250:
                #if we have completed 250 stepss
                rewards.append(total_reward)
                finished = True
       
            state = new_state
            new_cord = (x_n, y_n)
    #print(rewards)
    return(q_values)
    #print(q_values[13][0], q_values[13][3])

arr = []
avg = []
for i in range(10):
    arr.append(compute_Q(500, False, 'reward'))
#print(arr)
for i in range(500):
    sum = 0
    for j in range(10):
        sum += arr[j][i]
    avg.append(sum/10)

arr2 = []
avg2 = []
for i in range(10):
    arr2.append(sarsa(500, False, 'reward'))
#print(arr2)
for i in range(500):
    sum = 0
    for j in range(10):
        sum += arr2[j][i]
    avg2.append(sum/10)

#print(avg)
plt.plot(avg)
plt.plot(avg2)
plt.title('Fixed Epsilon')
plt.xlabel('Episode')
plt.ylabel('Average score')
plt.ylim([-100, 0])
plt.savefig('foo.png')

q_vals = compute_Q(500, False, 'q')
matrix = []
arr = []

for i in range(48):
    val = argmax(q_vals[i])
    if val == 0:
        arr.append('R')
    elif val == 1:
        arr.append('D')
    elif val == 2:
        arr.append('L')
    elif val == 3:
        arr.append('U')
    if (i + 1) % 12 == 0 and i != 0:
        matrix.append(arr)
        arr = []

for row in range(4):
    print(matrix[3 - row])




#plt.figure(figsize=(4,12))
#plt.imshow(matrix)
import gym
import pandas as pd
import numpy as np
from collections import deque
import random
import skimage
from skimage import transform
from skimage.color import rgb2gray
import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Bidirectional, Activation, Input

env = gym.make('SpaceInvaders-v0')
state_space = env.observation_space
action_size = env.action_space.n
state_space = (110, 84)
possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())
stack_size=4
learning_rate = 0.00025
decay_rate = 0.00001
exploration_start = 1
exploration_stop = 0.01

def pre_process_frame(frame): #convery frame to gray-scale
    gray = rgb2gray(frame) #crop the frame to be a set width and height
    cropped_frame = gray[8:-12, 4:-12]
    #normalize the frame
    normalized_frame = cropped_frame / 255.0
    #resize the frame and return it
    preprocessed_frame = transform.resize(normalized_frame, [110, 84])
    return preprocessed_frame

def stack_frames(stacked_frames, new_frame, new_episode):
    # A couple of variables so that we can store data and return it later.
    stacked_frames = stacked_frames
    stacked_state = None
    # pre-process a new frame passed to the function
    frame = pre_process_frame(new_frame)
    #if the episode is just starting, do the following
    if new_episode:
        #create a stack of frames
        stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)])
        # for as many frames as you want in the stack, fill the stacked_frames stack with the first frame in the game
        for _ in range(stack_size):
            stacked_frames.append(frame)
            stacked_state = np.stack(stacked_frames, axis=2)
    #Not starting a new game? Awesome! Do this
    else:
         # add the pre-processed frame to your image stack
         stacked_frames.append(frame)
         # convert the updated stacked_frames stack to a stacked_state
         stacked_state = np.stack(stacked_frames, axis=2)
    # return the whole thing so we can use it.
    return stacked_state, stacked_frames

def predict_action(nn, decay_step, state):
    exp_exp_tradeoff = np.random.rand()
    explore_proba = exploration_stop + (exploration_start - exploration_stop) * np.exp(-decay_rate * decay_step)
    if (explore_proba > exp_exp_tradeoff):
        choice = random.randint(1, len(possible_actions)) - 1
        action = possible_actions[choice]
    else:
        choice = np.argmax(nn.predict(np.array(stacked_frames).reshape(1, *state_size)))
        action = possible_actions[choice]
    return action

def sampleMemory(buffered_list, batch_size):
    buffer_size = len(buffered_list)
    index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
    return [buffered_list[i] for i in index]

model=Sequential()
model.add(Input(shape=(110,84,4)))
model.add(Conv2D(200, (60,60), data_format="channels_last"))
model.add(Conv2D(100, (20,20), data_format="channels_last"))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(env.action_space.n, activation='softmax'))
model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate, beta_1=exploration_stop, beta_2=exploration_start, decay=decay_rate), loss='categorical_crossentropy')


#create a qeue that will hold all the moments that
# the model will learn from
memory = deque(maxlen=1000)
#and we'll get our stacked frames started
# with an empty list. We'll fill it in via our function.
stacked_frames = []

#And we'll store JUST our final scores here.
rewards_list = []
#At each step in the training process, we're going to be
# increasing what we'll call the decay step. Remember from
# the Q-Table lesson how we used the decay_step to dictate
# what the threshold was for if we'd pick a random action or
# not? It'll play the same role here.
decay_step = 0

batch_size=20
#This next part is arbitrary, but vitally important. We need
# to choose how many games we want to run through. For
# each game, we set it up the usual way, by re-setting the
# game environment, and setting the total_reward for the
# game to zero. But we start this off a little differently by
# creating a stacked_frames and stacked_state right after
# and setting the new_game parameter in stack_frames to
# true.
for episode in range(9001):
    state = env.reset()
    total_reward = 0
    state, stacked_frames = stack_frames(stacked_frames, state, True)
    #Awesome! Now we'll get started with what needs
    # to happed for each step in the game. Step one is
    # deciding how many steps to take . . . all pun
    # intended. I set my algo to 1001.
    for step in range(1001):
        #So if you want to see the game play out at each step,
        # uncomment the following line. This renders the game
        # environment, so at each step this will play the game.
        #env.render()
        #Otherwise, we start off by increasing our decay_step
        # by one for this turn.
        decay_step += 1
        #Next, we'll predict an action using our pre-fabricated
        # prediction function. Since it spits out a 1-hot array,
        # and Gym needs an int for an action, we'll take the argmax
        # of the output from the function, hence why the function
        # is embedded in the np.argmax()
        act = np.argmax(predict_action(model, decay_step, decay_step))
        #Now, we'll update the game environment like we have
        # ever single other time we've played with this outside
        # of the box.
        obs, reward, done, info = env.step(act)
        #With the reward, we'll add it to our existing, total_reward
        total_reward += reward
        #So, if the game is done, we'll make sure the system knows
        # it's game over, append the reward to the total_reward list,
        # and add the stacked_state, action taken, reward, the new
        # state (obs), and whether the game finished.
        if done == True:
           obs = np.zeros((110, 84))
           obs, stacked_frames = stack_frames(stacked_frames, obs, False)
           rewards_list.append(total_reward)
           memory.append((state, act, reward, obs, done))
        #And if the game isn't finished, basically don't add total_reward
        # to reward_list and you're golden.
        else:
           obs, stacked_frames = stack_frames(stacked_frames, obs, False)
           memory.append((state, act, reward, obs, done))
           #After all of this, update state to BE the stacked version of obs, and
           # keep on keeping on.
        state = obs
    #So here's where the learning kicks in. To start, we need to specify
    # when we want to even start learning. How many memories is a
    # good starting point? For me, I'm arbitrarily starting at 100. Once
    # we have 100 memories in memory, then get rolling.
    if len(memory) > 100:

        # Pulling out useful info from a batch of memories and organizing
        # them before we use each bit of info to structure training data
        batch = sampleMemory(memory, batch_size=batch_size)
        actions = [item[1] for item in batch]
        states = np.array([item[0] for item in batch], ndmin=3)
        rewards = [item[2] for item in batch]
        next_states = np.array([item[0] for item in batch], ndmin=3)
        # Creates the rewards that the net will receive for the actions
        # taken in the batch.
        targets = [learning_rate * np.max(item) for item in model.predict(next_states)]
        targets = [targets[i] + rewards[i] for i in range(len(targets))]
        # Creates the outputs to fit to
        target_f = [item for item in model.predict(states)]
        for i in range(len(target_f)):
           target_f[i][actions[i]] = targets[i]
        # train on whole batch
        model.train_on_batch(x=np.array(states).reshape(-1, * state_space), y=np.array(target_f).reshape(-1, action_size))
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
import numpy as np
import gym

def getStateSize():
   state=env.reset()
   action = env.action_space.sample()
   obs, _, _, _ = env.step(action)
   return len(obs)

env = gym.make('CartPole-v0')
classes = env.action_space.n
len_data = getStateSize()

model = Sequential()
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(25))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

def initial_data(number_of_games, game_turns, acceptable_score):
   #The point of this function is to return some data that the computer can
   # use to learn. The function will play a number of games that the student
   # sets, for a number of possible turns in that game (also set by the
   # student), and then be divied up into good and bad games according
   # to an acceptable score. We then take the moves from the good games
   # and use those and the observations of what the game environment
   # looked like prior to that move in order to train the computer to do
   # the best possible move in a given situation.
   #We'll save all those good moves we're talking about here to train off of.
   x=[]
   y=[]
   l_1hot = [0 for i in range(classes)]
   #So let's start playing some games at random. The computer will play through
   # as many games as we tell it to here.
   for i in range(number_of_games):
      #Every time it plays a game, the environment needs to be reset.
      env.reset()
      #We need, too, to have some memory of what's happening in the game.
      # This is where we can store that info.
      game_memory = []
      #Each choice is dependent on the last observation made in the game.
      # You can't exactly predict the future, right? So we'll create a list variable
      # to hold onto that previous observation.
      prev_observation = []
      #And finally, we'll keep track of the score here. In game, the score is
      # called the reward, and it's either a 0 or 1 value for if things go well
      # or poorly. We can keep track of the total game score by adding the
      # reward to the int variable, "score”
      score = 0
      #So the game will start making moves up until the number of turns
      # we're allowing it to make.
      for turn in range(game_turns):
         #Each move is taking an action . . .
         action = env.action_space.sample()
         #Which we then pull the end result out of here as a series of variables
         observation, reward, done, info = env.step(action)
         #And then we update the score with how well the last turn went.
         score += int(reward)
         #Here's how we'll save this. If the turn is after the first move (turn 0),
         # then we'll append to game memory the previous observation and
         # the action the system took.
         if turn > 0:
            game_memory.append([prev_observation, int(action)])
         #And we'll set the variable for the previous observation to the data
         # from the most recent turn.
         prev_observation = observation
         #And if the game is over, it breaks the loop.
         if done == True:
            break
      #From here, we'll check to see what the score for this game was, and if it’s
      # greater than the lowest score we're willing to accept, we'll add the moves
      # from this game to our training data to play off of.
      if score >= acceptable_score:
         for data in game_memory:
            x.append(np.array(data[0]).reshape(1, len_data))
            #Perhaps problematically, the Keras Model only accepts data that
            # is structured as a 1-hot vector. We thus convert the action
            # generated in the action space to a 1-hot vector, with either a 1
            # at the index indicating a choice made. Think of it like this:
            """
                     MOVE      DON'T MOVE
            Yes, move  1        0
            No, don't    0          1

            """
            label=list(l_1hot)
            label[data[1]]=1
            y.append(np.array(label).reshape(1, classes))
   #Return training data to use for the NN
   print('{} examples were made.'.format(len(x)))
   return np.array(x).reshape(-1, 1, len_data), np.array(y).reshape(-1, 1, classes)

def play_game(n_games, n_moves, model=None):
   for i in range(n_games):
      state = env.reset()
      score = 0
      for step in range(n_moves):
         action = None
         if model==None:
            action = env.action_space.sample()
         else:
            action = np.argmax(model.predict(prev_observation))
         observation, reward, done, info = env.step(action)
         score += int(reward)
         state = observation
         if done == True:
            break
            print(score)

x, y = initial_data(1000, 500, 50)
model.fit(x=x, y=y, epochs=10, verbose=2)



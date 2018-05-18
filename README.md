# dqn_atari
The repository is a learning project; essentially me trying to figure out how Deep Q-learning works. 

It's a re-implementation of the DQN algorithm by DeepMind [1]. I've tried to keep the code simple, so it's easy to understand, and also fast to keep debugging cycles short.

Here's the agent playing Breakout after about 30M observations. This took a bit less than two days (wall time) on my desktop PC (1080 Ti) @ roughly 180 frames per second. It does so well, then falls in a heap after losing its first life (?!).
<img src="assets/breakout.gif" width="480">

I've learned a lot from the implementations of others. Notably from:

https://github.com/yilundu/DQN-DDQN-on-Space-Invaders

https://github.com/keras-rl/keras-rl

https://github.com/devsisters/DQN-tensorflow

and from the following blogs:

https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26

https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html

[1] Mnih et al., Human-level control through deep reinforcement learning, Nature (2015)

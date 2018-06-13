# dqn_atari
The repository is a learning project; essentially me trying to figure out how Deep Q-learning works. 

It's a re-implementation of the DeepMind DQN algorithm by Mnih et al. [1], with the Double Q-Learning modification by van Hasselt et al. [2]. I've tried to keep the code simple, so it's easy to understand, and also fast to keep debugging cycles short... debugging this thing took forever (!).

Here's the agent playing Breakout after about 10M observations. This took about 20 hours (wall time) on my desktop PC (1080 Ti) @ roughly 150 frames per second. 

<img src="assets/breakout_399_decimated.gif" width="480">

I've learned a lot from the implementations of others. Notably from:

https://github.com/yilundu/DQN-DDQN-on-Space-Invaders

https://github.com/keras-rl/keras-rl

https://github.com/devsisters/DQN-tensorflow

and from the following blogs:

https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26

https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html

[1] Mnih et al., Human-level control through deep reinforcement learning, Nature (2015)

[2] van Hasselt et al., Deep Reinforcement Learning with Double Q-learning, arXiv:1509.06461v3

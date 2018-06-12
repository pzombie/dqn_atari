import numpy as np

class replay_buffer:
    """Circular buffer for storing experiences for experience replay"""
    def __init__(self, size):
        self._observation_buffer = []
        self._experience_buffer = []
        self._max_size = size
        self._next_idx = 0
    
    def __len__(self):
        return(len(self._experience_buffer))
    
    def append(self, observation_t0, action, reward, done):
        experience = (action, reward, done)
        
        # haven't yet filled the buffer
        if self._next_idx >= len(self._experience_buffer):
            self._experience_buffer.append(experience)
            self._observation_buffer.append(observation_t0.copy())
        # replace old elements
        else:
            self._experience_buffer[self._next_idx] = experience
            self._observation_buffer[self._next_idx] = observation_t0.copy()
            
        # increment the next index, wrapping if necessary
        self._next_idx = (self._next_idx + 1) % self._max_size
        
    def get_batch(self, batch_size):
        # Pretty sure this function is leading to the creation of unnecessary copies of arrays... but where? Help.
        observations_t0, actions, rewards, dones, observations_t1 = [], [], [], [], []
        
        while len(actions) < batch_size:
            
            # Don't choose the elements near the start of the buffer so we don't have to 
            # deal with indexing a list slice across the boundary. That means there's essentially
            # some "dead" elements in the experience buffer, but we've got plenty so this should
            # be fine.
            index = np.random.randint(4, len(self._experience_buffer) - 2)
            
            # check if any (except the last) frame is terminal
            if True in [done for action, reward, done in self._experience_buffer[index-4:index+0]]:
                continue
            else:
                experience = self._experience_buffer[index]
                action, reward, done = experience
                observations_t0.append(self._observation_buffer[index-4:index+0])
                observations_t1.append(self._observation_buffer[index-3:index+1])

                actions.append(action)
                rewards.append(reward)
                dones.append(done)
            
        obs0_    = np.transpose(np.array(observations_t0, copy = False), (0, 2, 3, 1))
        actions_ = np.array(actions)
        rewards_ = np.array(rewards)
        dones_   = np.array(dones)
        obs1_    = np.transpose(np.array(observations_t1, copy = False), (0, 2, 3, 1))                
            
        return obs0_.copy(), actions_.astype('uint8'), rewards_, dones_, obs1_.copy()
    
    def get_last_state(self):
        if self._next_idx >= 4:
            return np.dstack(self._observation_buffer[self._next_idx-4:self._next_idx]).copy()
        # wrapping the container
        else:
            return np.dstack(self._observation_buffer[self._max_size - 4 + self._next_idx:] + self._observation_buffer[0:self._next_idx]).copy()
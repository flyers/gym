import gym
import numpy

env = gym.make('VREP-v0')
# env = gym.make('Breakout-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = numpy.random.normal(0, 0.1, (4,))
        # action = env.action_space.sample()
        # print 'action:', action
        observation, reward, done, info = env.step(action)
        print observation
        if done:
            print "Episode %d finished after %d timesteps" % (i_episode, t+1)
            break
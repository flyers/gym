import gym
import numpy
from gym.envs.vrep.vrep_hierarchy_target_env import VREPHierarchyTargetEnv

# env = gym.make('VREP-v0')
env = VREPHierarchyTargetEnv(
    vrep_path='/home/sliay/V-REP_PRO_EDU_V3_3_2_64_Linux',
    scene_path='/home/sliay/Documents/vrep-uav/scenes/quadcopter_hierarchy_64x64.ttt',
    headless=False, log=False, obs_type='state',
    state_type='body', reward_func=6, random_start=True,
    action_type='discrete',
)
# env = gym.make('Breakout-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        # action = numpy.random.normal(0, 0.1, (4,))
        action = env.action_space.sample()
        # print 'action:', action
        observation, reward, done, info = env.step(action)
        print observation
        if done:
            print "Episode %d finished after %d timesteps" % (i_episode, t + 1)
            break

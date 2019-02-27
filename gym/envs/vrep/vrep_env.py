import logging
from collections import deque
import numpy
from gym import error, spaces
from .vrep_base import VREPBaseEnv
from . import config
from .utils import goal_func_3, aux_func

try:
    from . import vrep
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to perform the setup instructions here: http://www.coppeliarobotics.com/helpFiles/en/remoteApiClientSide.htm.)".format(
            e))

logger = logging.getLogger(__name__)


class VREPEnv(VREPBaseEnv):
    def __init__(self,
                 scene_path='/home/sliay/Documents/vrep-uav/scenes/quadcopter_hierarchy_64x64.ttt',
                 reward_func=0, log=False, action_type='continuous', discrete_type=0,
                 action_bound=0.025, state_history_length=1, action_history_length=0,
                 future_action_length=1,
                 **kwargs):
        self._reward_func = reward_func
        self.scene_path = scene_path
        self._log = log
        self._action_type = action_type
        self._discrete_type = discrete_type
        self._goal_height = 1.5
        self._goal_target = numpy.array([0., 0., 0.4])

        super(VREPEnv, self).__init__(**kwargs)

        # set action bound
        if self._action_type == 'continuous':
            self.action_space = spaces.Box(low=-1., high=1., shape=(4,))
            self._action_lb = -1 * action_bound
            self._action_ub = action_bound
        else:
            self._action_granularity = 0.02
            if self._discrete_type == 0:
                self.AVAILABLE_ACTION = config.AVAILABLE_ACTION_2 * self._action_granularity
            else:
                self.AVAILABLE_ACTION = config.AVAILABLE_ACTION_1 * self._action_granularity
            self.action_space = spaces.Discrete(self.AVAILABLE_ACTION.shape[0])
        state_bound = numpy.array([4, 4, 4] + [4, 4, 4] +  # v, w
                                  [numpy.inf, ] +  # quadcopter height
                                  [1, 1, 1, 1] +  # quaternion
                                  # normalized coordinates on camera plane
                                  [0.5, 0.5, 1]
                                  )
        self.state_space = spaces.Box(low=-state_bound, high=state_bound)
        assert self._obs_type == 'state'

        self._state_history_length = state_history_length
        self._action_history_length = action_history_length
        self._future_action_length = future_action_length
        self.states = deque([], maxlen=self._state_history_length)
        self.actions = deque([], maxlen=self._action_history_length)
        state_ub = self.state_space.high.tolist()
        action_ub = self.action_space.high.tolist()
        observation_bound = state_ub * self._state_history_length + \
            action_ub * self._action_history_length
        observation_bound = numpy.array(observation_bound)
        self.observation_space = spaces.Box(
            low=-observation_bound, high=observation_bound)

    def _concat_state(self):
        state = numpy.concatenate(self.states)
        if self._action_history_length > 0:
            action = numpy.concatenate(self.actions)
            return numpy.concatenate([state, action])
        else:
            return state

    def _get_state(self):
        if self._state_type == 'world':
            return numpy.array(self.linear_velocity_g +
                               self.angular_velocity_g +
                               [self.quadcopter_pos[2], ] +
                               self.quadcopter_quaternion +
                               self.target_coordinates.tolist(),
                               dtype='float32')
        else:
            return numpy.array(self.linear_velocity_b.tolist() +
                               self.angular_velocity_b +
                               [self.quadcopter_pos[2], ] +
                               self.quadcopter_quaternion +
                               self.target_coordinates.tolist(),
                               dtype='float32')

    def _act(self, action):
        # send control signal to server side
        vrep.simxSetStringSignal(self.client_id, 'control_signal',
                                 vrep.simxPackFloats(action), vrep.simx_opmode_oneshot)
        # trigger next simulation step
        vrep.simxSynchronousTrigger(self.client_id)
        # read sensor
        self._read_sensor()

        eval_str = 'self.eval_reward_' + str(self._reward_func) + '(action)'
        return eval(eval_str)

    def eval_reward_0(self, action):
        if self._game_over():
            return -10.0
        r_coord = goal_func_3(
            self.target_coordinates[:2] - self._goal_target[:2], [0.1, 0.25])
        r_scale = goal_func_3(
            self.target_coordinates[2] - self._goal_target[2], [0.05, 0.2])
        r_orientation = aux_func(numpy.array(
            self.quadcopter_orientation[0:2]) / numpy.pi - 0, 0.03) * 1.0
        r_height = aux_func(
            self.quadcopter_pos[2] - self._goal_height, 0.5) * 0.5
        reward = r_coord + r_scale + r_orientation + r_height
        if self._log:
            logger.info(
                'TargetLoc_x Reward:%.4f\nTargetLoc_y Reward:%.4f\nTargetLoc_h Reward:%.4f\nTargetPos Reward:%.4f\nW Reward:%f\nTotal Reward:%.4f\n' %
                (
                    r_coord,
                    r_coord,
                    r_scale,
                    r_height,
                    r_orientation,
                    reward,
                )
            )
        return reward

    def eval_reward_1(self, action):
        """
        Add penalty for large actions 
        """
        if self._game_over():
            return -10.0
        r_coord = goal_func_3(
            self.target_coordinates[:2] - self._goal_target[:2], [0.1, 0.25])
        r_scale = goal_func_3(
            self.target_coordinates[2] - self._goal_target[2], [0.05, 0.2])
        r_orientation = aux_func(numpy.array(
            self.quadcopter_orientation[0:2]) / numpy.pi - 0, 0.03) * 1.0
        r_height = aux_func(
            self.quadcopter_pos[2] - self._goal_height, 0.5) * 0.5
        r_action = aux_func(action, 0.5)
        reward = r_coord + r_scale + r_orientation + r_height + r_action
        if self._log:
            logger.info(
                'TargetLoc_x Reward:%.4f\nTargetLoc_y Reward:%.4f\nTargetLoc_h Reward:%.4f\nTargetPos Reward:%.4f\nW Reward:%f\nTotal Reward:%.4f\n' %
                (
                    r_coord,
                    r_coord,
                    r_scale,
                    r_height,
                    r_orientation,
                    reward,
                )
            )
        return reward

    def eval_reward_2(self, action):
        """
        Add penalty for large actions 
        """
        if self._game_over():
            return -10.0
        r_coord = goal_func_3(
            self.target_coordinates[:2] - self._goal_target[:2], [0.1, 0.25])
        r_scale = goal_func_3(
            self.target_coordinates[2] - self._goal_target[2], [0.05, 0.2])
        r_orientation = aux_func(numpy.array(
            self.quadcopter_orientation[0:2]) / numpy.pi - 0, 0.03) * 1.0
        r_height = aux_func(
            self.quadcopter_pos[2] - self._goal_height, 0.5) * 0.5
        r_action = aux_func(action, 0.3)
        reward = r_coord + r_scale + r_orientation + r_height + r_action
        if self._log:
            logger.info(
                'TargetLoc_x Reward:%.4f\nTargetLoc_y Reward:%.4f\nTargetLoc_h Reward:%.4f\nTargetPos Reward:%.4f\nW Reward:%f\nTotal Reward:%.4f\n' %
                (
                    r_coord,
                    r_coord,
                    r_scale,
                    r_height,
                    r_orientation,
                    reward,
                )
            )
        return reward

    def eval_reward_3(self, action):
        """
        Add penalty for large actions 
        """
        if self._game_over():
            return -10.0
        r_coord = goal_func_3(
            self.target_coordinates[:2] - self._goal_target[:2], [0.1, 0.25])
        r_scale = goal_func_3(
            self.target_coordinates[2] - self._goal_target[2], [0.05, 0.2])
        r_orientation = aux_func(numpy.array(
            self.quadcopter_orientation[0:2]) / numpy.pi - 0, 0.03) * 1.0
        r_height = aux_func(
            self.quadcopter_pos[2] - self._goal_height, 0.5) * 0.5
        r_action = aux_func(action, 0.1)
        reward = r_coord + r_scale + r_orientation + r_height + r_action
        if self._log:
            logger.info(
                'TargetLoc_x Reward:%.4f\nTargetLoc_y Reward:%.4f\nTargetLoc_h Reward:%.4f\nTargetPos Reward:%.4f\nW Reward:%f\nTotal Reward:%.4f\n' %
                (
                    r_coord,
                    r_coord,
                    r_scale,
                    r_height,
                    r_orientation,
                    reward,
                )
            )
        return reward

    def eval_reward_4(self, action):
        """
        Add penalty for large action variations 
        """
        if self._game_over():
            return -10.0
        r_coord = goal_func_3(
            self.target_coordinates[:2] - self._goal_target[:2], [0.1, 0.25])
        r_scale = goal_func_3(
            self.target_coordinates[2] - self._goal_target[2], [0.05, 0.2])
        r_orientation = aux_func(numpy.array(
            self.quadcopter_orientation[0:2]) / numpy.pi - 0, 0.03) * 1.0
        r_height = aux_func(
            self.quadcopter_pos[2] - self._goal_height, 0.5) * 0.5
        past_actions = numpy.array(self.actions)
        delta_actions = past_actions[1::] - past_actions[0:-1]
        r_action = aux_func(delta_actions, 0.5) * 0.5
        reward = r_coord + r_scale + r_orientation + r_height + r_action
        if self._log:
            logger.info(
                'TargetLoc_x Reward:%.4f\nTargetLoc_y Reward:%.4f\nTargetLoc_h Reward:%.4f\nTargetPos Reward:%.4f\nW Reward:%f\nTotal Reward:%.4f\n' %
                (
                    r_coord,
                    r_coord,
                    r_scale,
                    r_height,
                    r_orientation,
                    reward,
                )
            )
        return reward

    def eval_reward_5(self, action):
        """
        Add penalty for large action variations 
        """
        if self._game_over():
            return -10.0
        r_coord = goal_func_3(
            self.target_coordinates[:2] - self._goal_target[:2], [0.1, 0.25])
        r_scale = goal_func_3(
            self.target_coordinates[2] - self._goal_target[2], [0.05, 0.2])
        r_orientation = aux_func(numpy.array(
            self.quadcopter_orientation[0:2]) / numpy.pi - 0, 0.03) * 1.0
        r_height = aux_func(
            self.quadcopter_pos[2] - self._goal_height, 0.5) * 0.5
        past_actions = numpy.array(self.actions)
        delta_actions = past_actions[1::] - past_actions[0:-1]
        r_action = aux_func(delta_actions, 0.3) * 0.5
        reward = r_coord + r_scale + r_orientation + r_height + r_action
        if self._log:
            logger.info(
                'TargetLoc_x Reward:%.4f\nTargetLoc_y Reward:%.4f\nTargetLoc_h Reward:%.4f\nTargetPos Reward:%.4f\nW Reward:%f\nTotal Reward:%.4f\n' %
                (
                    r_coord,
                    r_coord,
                    r_scale,
                    r_height,
                    r_orientation,
                    reward,
                )
            )
        return reward

    def eval_reward_6(self, action):
        """
        Add penalty for large action variations 
        """
        if self._game_over():
            return -10.0
        r_coord = goal_func_3(
            self.target_coordinates[:2] - self._goal_target[:2], [0.1, 0.25])
        r_scale = goal_func_3(
            self.target_coordinates[2] - self._goal_target[2], [0.05, 0.2])
        r_orientation = aux_func(numpy.array(
            self.quadcopter_orientation[0:2]) / numpy.pi - 0, 0.03) * 1.0
        r_height = aux_func(
            self.quadcopter_pos[2] - self._goal_height, 0.5) * 0.5
        past_actions = numpy.array(self.actions)
        delta_actions = past_actions[1::] - past_actions[0:-1]
        r_action = aux_func(delta_actions, 0.1) * 0.5
        reward = r_coord + r_scale + r_orientation + r_height + r_action
        if self._log:
            logger.info(
                'TargetLoc_x Reward:%.4f\nTargetLoc_y Reward:%.4f\nTargetLoc_h Reward:%.4f\nTargetPos Reward:%.4f\nW Reward:%f\nTotal Reward:%.4f\n' %
                (
                    r_coord,
                    r_coord,
                    r_scale,
                    r_height,
                    r_orientation,
                    reward,
                )
            )
        return reward

    def eval_reward_7(self, action):
        """
        Add penalty for large action accelerations
        """
        if self._game_over():
            return -10.0
        r_coord = goal_func_3(
            self.target_coordinates[:2] - self._goal_target[:2], [0.1, 0.25])
        r_scale = goal_func_3(
            self.target_coordinates[2] - self._goal_target[2], [0.05, 0.2])
        r_orientation = aux_func(numpy.array(
            self.quadcopter_orientation[0:2]) / numpy.pi - 0, 0.03) * 1.0
        r_height = aux_func(
            self.quadcopter_pos[2] - self._goal_height, 0.5) * 0.5
        past_actions = numpy.array(self.actions)
        delta_actions = past_actions[1::] - past_actions[0:-1]
        acce_actions = delta_actions[1::] - delta_actions[0:-1]
        r_action = aux_func(acce_actions, 0.1) * 0.5
        reward = r_coord + r_scale + r_orientation + r_height + r_action
        if self._log:
            logger.info(
                'TargetLoc_x Reward:%.4f\nTargetLoc_y Reward:%.4f\nTargetLoc_h Reward:%.4f\nTargetPos Reward:%.4f\nW Reward:%f\nTotal Reward:%.4f\n' %
                (
                    r_coord,
                    r_coord,
                    r_scale,
                    r_height,
                    r_orientation,
                    reward,
                )
            )
        return reward

    def eval_reward_8(self, action):
        """
        Add penalty for large action accelerations
        """
        if self._game_over():
            return -10.0
        r_coord = goal_func_3(
            self.target_coordinates[:2] - self._goal_target[:2], [0.1, 0.25])
        r_scale = goal_func_3(
            self.target_coordinates[2] - self._goal_target[2], [0.05, 0.2])
        r_orientation = aux_func(numpy.array(
            self.quadcopter_orientation[0:2]) / numpy.pi - 0, 0.03) * 1.0
        r_height = aux_func(
            self.quadcopter_pos[2] - self._goal_height, 0.5) * 0.5
        r_action = aux_func(action, 0.1)        
        past_actions = numpy.array(self.actions)
        delta_actions = past_actions[1::] - past_actions[0:-1]
        r_action_delta = aux_func(delta_actions, 0.1) * 0.5
        reward = r_coord + r_scale + r_orientation + r_height + r_action + r_action_delta
        if self._log:
            logger.info(
                'TargetLoc_x Reward:%.4f\nTargetLoc_y Reward:%.4f\nTargetLoc_h Reward:%.4f\nTargetPos Reward:%.4f\nW Reward:%f\nTotal Reward:%.4f\n' %
                (
                    r_coord,
                    r_coord,
                    r_scale,
                    r_height,
                    r_orientation,
                    reward,
                )
            )
        return reward

    def _game_over(self):
        done = (not self.state_space.contains(self._get_state())) \
            or self.quadcopter_pos[2] <= 0 or self.quadcopter_pos[2] >= 5 \
            or abs(self.quadcopter_orientation[0]) >= numpy.pi / 3 \
            or abs(self.quadcopter_orientation[1]) >= numpy.pi / 3
        return done

    def _reset(self):
        obs = super(VREPEnv, self)._reset()
        for _ in range(self._state_history_length):
            self.states.append(obs)
        for _ in range(self._action_history_length):
            self.actions.append(numpy.zeros(self.action_space.shape))
        return self._concat_state()

    def _step(self, a):
        reward = 0.0
        if self._log:
            logger.info(
                'OriginalAction:%.4f\t%.4f\t%.4f\t%.4f' % (
                    a[0], a[1], a[2], a[3])
            )
        self.actions.append(a)
        if self._action_type == 'continuous':
            a = self._normalize_action(a)
        else:
            a = self.AVAILABLE_ACTION[a]
        for _ in range(self.frame_skip):
            reward += self._act(a)
        ob = self._get_obs()

        self.states.append(ob)

        if self._log:
            logger.info(
                'TargetPos:%.4f\t%.4f\t%.4f' % (self.target_pos[0], self.target_pos[1], self.target_pos[2]))
            logger.info(
                'TargetCameraLoc:%.4f\t%.4f\t%.4f' % (
                    self.target_coordinates[0], self.target_coordinates[1], self.target_coordinates[2]))
            logger.info(
                'QuadPos:%.4f\t%.4f\t%.4f' % (self.quadcopter_pos[0], self.quadcopter_pos[1], self.quadcopter_pos[2]))
            logger.info(
                'QuadOrient:%.4f\t%.4f\t%.4f' % (
                    self.quadcopter_orientation[0], self.quadcopter_orientation[1], self.quadcopter_orientation[2])
            )
            logger.info(
                'QuadQuaternion:%.4f\t%.4f\t%.4f\t%.4f' % (
                    self.quadcopter_quaternion[0], self.quadcopter_quaternion[1], self.quadcopter_quaternion[2],
                    self.quadcopter_quaternion[3])
            )
            logger.info(
                'QuadLinearVel:%.4f\t%.4f\t%.4f' % (
                    self.linear_velocity_g[0], self.linear_velocity_g[1], self.linear_velocity_g[2]))
            logger.info(
                'QuadAngVel:%.4f\t%.4f\t%.4f' % (
                    self.angular_velocity_g[0], self.angular_velocity_g[1], self.angular_velocity_g[2]))
            logger.info(
                'QuadLinearBodyVel:%.4f\t%.4f\t%.4f' % (
                    self.linear_velocity_b[0], self.linear_velocity_b[1], self.linear_velocity_b[2]))
            logger.info(
                'QuadAngBodyVel:%.4f\t%.4f\t%.4f' % (
                    self.angular_velocity_b[0], self.angular_velocity_b[1], self.angular_velocity_b[2]))
            logger.info(
                'QuadMotor:%.4f\t%.4f\t%.4f\t%.4f' % (a[0], a[1], a[2], a[3]))
            logger.info('Reward:%f' % reward)

        terminal = self._game_over()
        info = {}
        return self._concat_state(), reward, terminal, info

    # transform normalized action back
    def _normalize_action(self, action):
        lb = self._action_lb
        ub = self._action_ub
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = numpy.clip(scaled_action, lb, ub)
        return scaled_action

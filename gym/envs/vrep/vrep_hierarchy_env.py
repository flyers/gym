import logging
import numpy
from gym import error, spaces
from .vrep_base import VREPBaseEnv
from . import config

try:
    from . import vrep
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to perform the setup instructions here: http://www.coppeliarobotics.com/helpFiles/en/remoteApiClientSide.htm.)".format(
            e))

logger = logging.getLogger(__name__)


class VREPHierarchyEnv(VREPBaseEnv):

    def _get_image(self):
        # return image shape is 3 by height by width
        return self.image.transpose(2, 0, 1)

    def _get_state(self):
        if self._state_type == 'world':
            return numpy.array(self.linear_velocity_g +
                               self.angular_velocity_g +
                               self.quadcopter_pos +
                               self.quadcopter_quaternion, dtype='float32')
        else:
            return numpy.array(self.linear_velocity_b.tolist() +
                               self.angular_velocity_b +
                               self.quadcopter_pos +
                               self.quadcopter_quaternion, dtype='float32')

    def _act(self, action):
        # send control signal to server side
        vrep.simxSetStringSignal(self.client_id, 'control_signal',
                                 vrep.simxPackFloats(action), vrep.simx_opmode_oneshot)
        # trigger next simulation step
        vrep.simxSynchronousTrigger(self.client_id)
        # read sensor
        self._read_sensor()

        return self.eval_reward_dy(action)

    def eval_reward_dy(self, action):
        if self._state_type == 'world':
            v = self.linear_velocity_g
            w = self.angular_velocity_g
        else:
            v = self.linear_velocity_b
            w = self.angular_velocity_b
        reward_p = \
            numpy.exp(-numpy.square(
                self.quadcopter_pos[2] - self._goal_height))
        # p_baseline = numpy.exp(-numpy.square([0.2]))
        reward_q = \
            numpy.exp(-numpy.square(self.quadcopter_quaternion -
                                    numpy.array([0, 0, 1, 0])).sum())
        # q_baseline = numpy.exp(-numpy.square([0.3, 0.3, 0.3, 0.3]).sum())
        reward_v = \
            numpy.exp(-numpy.square(numpy.array(v) - [1, 0, 0]).sum())
        # v_baseline = numpy.exp(-numpy.square([0.2, 0.2, 0.2]).sum())
        reward_w = \
            numpy.exp(-numpy.square(numpy.array(w) - [0, 0, 0]).sum())
        # w_baseline = numpy.exp(-numpy.square([0.2, 0.2, 0.2]).sum())
        R_c = self.R_c
        reward = \
            reward_p + reward_q + reward_v + reward_w - R_c
        if self._log:
            logger.info('Baseline Reward Constant:%.2f' % R_c)
            logger.info(
                'Position Reward:%.4f\nQuaternion Reward:%.4f\nV Reward:%.4f\nW Reward:%.4f\nTotal Reward:%.2f' %
                (
                    reward_p,
                    reward_q,
                    reward_v,
                    reward_w,
                    # reward_u, 0, reward_u,
                    reward,
                )
            )
        return numpy.asscalar(reward)

    def _game_over(self):
        done = (not self.state_space.contains(self._get_state())) \
            or self.quadcopter_pos[2] <= 0 or self.quadcopter_pos[2] >= 5 \
            or abs(self.quadcopter_orientation[0]) >= numpy.pi/3 \
            or abs(self.quadcopter_orientation[1]) >= numpy.pi/3
        return done

    def __init__(self,
                 scene_path='/home/sliay/Documents/vrep-uav/scenes/quadcopter_hierarchy.ttt',
                 reward_func=1, log=False, action_type='continuous', discrete_type=0,
                 reward_baseline=2.5, terminal_penalty=0,
                 **kwargs):
        self._reward_func = reward_func
        self.scene_path = scene_path
        self._log = log
        self.R_c = reward_baseline
        self.R_terminal = terminal_penalty
        self._action_type = action_type
        self._discrete_type = discrete_type
        self._goal_height = 1.5
        super(VREPHierarchyEnv, self).__init__(**kwargs)

        # set action bound
        if self._action_type == 'continuous':
            self.action_space = spaces.Box(low=-1., high=1., shape=(4,))
            self._action_lb = -0.1
            self._action_ub = 0.1
        else:
            # self._action_granularity = numpy.array([0.01, 0.01, 0.01, 0.01])
            self._action_granularity = 0.02
            if self._discrete_type == 0:
                self.AVAILABLE_ACTION = config.AVAILABLE_ACTION_2 * self._action_granularity
            else:
                self.AVAILABLE_ACTION = config.AVAILABLE_ACTION_1 * self._action_granularity
            self.action_space = spaces.Discrete(self.AVAILABLE_ACTION.shape[0])
        state_bound = numpy.array([4, 4, 4] + [4, 4, 4] +  # v, w
                                  # [4, 4, 4] + [4, 4, 4] + # a_v, a_w
                                  [numpy.inf, numpy.inf, numpy.inf] +  # position
                                  [1, 1, 1, 1]  # quaternion
                                  )
        self.state_space = spaces.Box(low=-state_bound, high=state_bound)
        if self._obs_type == 'image':
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(3, 128, 128))
        elif self._obs_type == 'state':
            self.observation_space = spaces.Box(
                low=-state_bound, high=state_bound)
        else:
            raise error.Error(
                'Unrecognized observation type: {}'.format(self._obs_type))

    def _step(self, a):
        reward = 0.0
        if self._action_type == 'continuous':
            a = self._normalize_action(a)
        else:
            a = self.AVAILABLE_ACTION[a]
        for _ in range(self.frame_skip):
            reward += self._act(a)
        ob = self._get_obs()

        if self._log:
            logger.info(
                'QuadPos:%.4f\t%.4f\t%.4f' % (self.quadcopter_pos[0], self.quadcopter_pos[1], self.quadcopter_pos[2]))
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
        if terminal:
            reward -= self.R_terminal
        return ob, reward, terminal, {}

    # transform normalized action back
    def _normalize_action(self, action):
        lb = self._action_lb
        ub = self._action_ub
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = numpy.clip(scaled_action, lb, ub)
        return scaled_action

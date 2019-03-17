import logging
import numpy
from gym import error, spaces
from .vrep_base import VREPBaseEnv

try:
    from . import vrep
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to perform the setup instructions here: http://www.coppeliarobotics.com/helpFiles/en/remoteApiClientSide.htm.)".format(
            e))

logger = logging.getLogger(__name__)


class VREPThrustEnv(VREPBaseEnv):

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
        vrep.simxSetStringSignal(self.client_id, 'thrust',
                                 vrep.simxPackFloats(action), vrep.simx_opmode_oneshot)
        # trigger next simulation step
        vrep.simxSynchronousTrigger(self.client_id)
        # read sensor
        self._read_sensor()

        motors = numpy.zeros(4)
        motors[0] = action[0] * (1 - action[1] + action[2] + action[3])
        motors[1] = action[0] * (1 - action[1] - action[2] - action[3])
        motors[2] = action[0] * (1 + action[1] - action[2] + action[3])
        motors[3] = action[0] * (1 + action[1] + action[2] - action[3])
        return self.eval_reward_dy(motors)

    def eval_reward_dy(self, action):
        if self._state_type == 'world':
            v = self.linear_velocity_g
            w = self.angular_velocity_g
        else:
            v = self.linear_velocity_b
            w = self.angular_velocity_b
        reward_p = \
            numpy.exp(-numpy.square(self.quadcopter_pos[2] - self._goal_height))
        # p_baseline = numpy.exp(-numpy.square([0.2]))
        reward_q = \
            numpy.exp(-numpy.square(self.quadcopter_quaternion - numpy.array([0, 0, 1, 0])).sum())
        # q_baseline = numpy.exp(-numpy.square([0.3, 0.3, 0.3, 0.3]).sum())
        reward_v = \
            numpy.exp(-numpy.square(numpy.array(v) - [1, 0, 0]).sum())
        # v_baseline = numpy.exp(-numpy.square([0.2, 0.2, 0.2]).sum())
        reward_w = \
            numpy.exp(-numpy.square(numpy.array(w) - [0, 0, 0]).sum())
        # w_baseline = numpy.exp(-numpy.square([0.2, 0.2, 0.2]).sum())
        reward_u = \
            numpy.exp(-numpy.square(action - 5.335).sum())
        # u_baseline = numpy.exp(-numpy.square([1, 1, 1, 1]).sum())
        # R_c = p_baseline + q_baseline + v_baseline + w_baseline + u_baseline + 0.5
        R_c = self.R_c
        reward = \
            reward_p + reward_q + reward_v + reward_w + reward_u - R_c
        if self._log:
            logger.info(action)
            logger.info('Baseline Reward Constant:%.2f' % R_c)
            logger.info(
                'Position Reward:%.4f\nQuaternion Reward:%.4f\nV Reward:%.4f\nW Reward:%.4f\nMotor Reward:%.4f\nTotal Reward:%.2f' %
                (
                    reward_p,
                    reward_q,
                    reward_v,
                    reward_w,
                    reward_u,
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
                 scene_path='/home/sliay/Documents/vrep-uav/scenes/quadcopter_thrust.ttt',
                 reward_func=1, log=False,
                 reward_baseline=2.95, terminal_penalty=0,
                 **kwargs):
        self._reward_func = reward_func
        self.R_c = reward_baseline
        self.R_terminal = terminal_penalty
        self.scene_path = scene_path
        self._log = log
        self._goal_height = 1.5
        super(VREPThrustEnv, self).__init__(**kwargs)

        # set action bound
        self.action_space = spaces.Box(low=-1., high=1., shape=(4,))
        self._action_lb = numpy.array([4.97, -0.05, -0.05, -0.05])
        self._action_ub = numpy.array([5.7, 0.05, 0.05, 0.05])
        state_bound = numpy.array([4, 4, 4] + [4, 4, 4] + # v, w
                                  # [4, 4, 4] + [4, 4, 4] + # a_v, a_w
                                  [numpy.inf, numpy.inf, numpy.inf] +  # position
                                  [1, 1, 1, 1]  # quaternion
                                  )
        self.state_space = spaces.Box(low=-state_bound, high=state_bound)
        if self._obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(3, 128, 128))
        elif self._obs_type == 'state':
            self.observation_space = spaces.Box(low=-state_bound, high=state_bound)
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

    def _step(self, a):
        reward = 0.0
        a = self._normalize_action(a)
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
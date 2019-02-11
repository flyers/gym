import logging
import numpy
from gym import error, spaces
from gym.envs.vrep.vrep_base import VREPBaseEnv

try:
    from gym.envs.vrep import vrep
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to perform the setup instructions here: http://www.coppeliarobotics.com/helpFiles/en/remoteApiClientSide.htm.)".format(
            e))

logger = logging.getLogger(__name__)

AVAILABLE_ACTION_1 = numpy.array([
    [1, 1, 1, 1],
    [-1, 1, 1, 1],
    [1, -1, 1, 1],
    [-1, -1, 1, 1],
    [1, 1, -1, 1],
    [-1, 1, -1, 1],
    [1, -1, -1, 1],
    [-1, -1, -1, 1],
    [1, 1, 1, -1],
    [-1, 1, 1, -1],
    [1, -1, 1, -1],
    [-1, -1, 1, -1],
    [1, 1, -1, -1],
    [-1, 1, -1, -1],
    [1, -1, -1, -1],
    [-1, -1, -1, -1],
])
AVAILABLE_ACTION_2 = numpy.array([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, -1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 1, -1],
    [0, 0, -1, 0],
    [0, 0, -1, 1],
    [0, 0, -1, -1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 0, -1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [0, 1, 1, -1],
    [0, 1, -1, 0],
    [0, 1, -1, 1],
    [0, 1, -1, -1],
    [0, -1, 0, 0],
    [0, -1, 0, 1],
    [0, -1, 0, -1],
    [0, -1, 1, 0],
    [0, -1, 1, 1],
    [0, -1, 1, -1],
    [0, -1, -1, 0],
    [0, -1, -1, 1],
    [0, -1, -1, -1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 0, -1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 0, 1, -1],
    [1, 0, -1, 0],
    [1, 0, -1, 1],
    [1, 0, -1, -1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 0, -1],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
    [1, 1, 1, -1],
    [1, 1, -1, 0],
    [1, 1, -1, 1],
    [1, 1, -1, -1],
    [1, -1, 0, 0],
    [1, -1, 0, 1],
    [1, -1, 0, -1],
    [1, -1, 1, 0],
    [1, -1, 1, 1],
    [1, -1, 1, -1],
    [1, -1, -1, 0],
    [1, -1, -1, 1],
    [1, -1, -1, -1],
    [-1, 0, 0, 0],
    [-1, 0, 0, 1],
    [-1, 0, 0, -1],
    [-1, 0, 1, 0],
    [-1, 0, 1, 1],
    [-1, 0, 1, -1],
    [-1, 0, -1, 0],
    [-1, 0, -1, 1],
    [-1, 0, -1, -1],
    [-1, 1, 0, 0],
    [-1, 1, 0, 1],
    [-1, 1, 0, -1],
    [-1, 1, 1, 0],
    [-1, 1, 1, 1],
    [-1, 1, 1, -1],
    [-1, 1, -1, 0],
    [-1, 1, -1, 1],
    [-1, 1, -1, -1],
    [-1, -1, 0, 0],
    [-1, -1, 0, 1],
    [-1, -1, 0, -1],
    [-1, -1, 1, 0],
    [-1, -1, 1, 1],
    [-1, -1, 1, -1],
    [-1, -1, -1, 0],
    [-1, -1, -1, 1],
    [-1, -1, -1, -1],
    ])


def goal_func_3(delta, thres):
    norm = numpy.linalg.norm(delta)
    if norm <= thres[0]:
        return numpy.exp(-norm)
    elif norm <= thres[1]:
        return 0
    else:
        return -1 * (norm - thres[1])


def goal_func_2(delta, thres):
    norm = numpy.linalg.norm(delta)
    return numpy.exp(-norm) if norm <= thres else 0.0


def aux_func(delta, thres):
    norm = numpy.linalg.norm(delta)
    return (numpy.exp(-norm) - 1) if norm <= thres else -1.0


class VREPHierarchyTargetEnv(VREPBaseEnv):

    def _init_sensor(self):
        super(VREPHierarchyTargetEnv, self)._init_sensor()
        if self._obs_type == 'image':
            _, self.resolution, self.image = vrep.simxGetVisionSensorImage(
                self.client_id, self.camera_handle, 0, vrep.simx_opmode_streaming)

    def _read_sensor(self):
        super(VREPHierarchyTargetEnv, self)._read_sensor()
        if self._obs_type == 'image':
            self._read_camera_image()

    def _get_image(self):
        # return image shape is 3 by height by width
        return self.image.transpose(2, 0, 1)

    def _get_state(self):
        if self._state_type == 'world':
            return numpy.array(self.linear_velocity_g +
                               self.angular_velocity_g +
                               # self.quadcopter_pos +
                               [self.quadcopter_pos[2],] +
                               self.quadcopter_quaternion +
                               # self.quadcopter_orientation +
                               # self.target_pos +
                               self.target_coordinates.tolist(),
                               dtype='float32')
        else:
            return numpy.array(self.linear_velocity_b.tolist() +
                               self.angular_velocity_b +
                               # self.quadcopter_pos +
                               [self.quadcopter_pos[2],] +
                               self.quadcopter_quaternion +
                               # self.quadcopter_quaternion.tolist() +
                               # self.quadcopter_orientation +
                               # self.target_pos +
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

        if self._reward_func == 6:
            return self.eval_reward_cz_5(action)
        elif self._reward_func == 10:
            return self.eval_reward_cz_9(action)
        elif self._reward_func == 11:
            return self.eval_reward_cz_10(action)
        elif self._reward_func == 12:
            return self.eval_reward_cz_11(action)
        elif self._reward_func == 13:
            return self.eval_reward_cz_12(action)
        elif self._reward_func == 14:
            return self.eval_reward_cz_13(action)
        elif self._reward_func == 21:
            return self.eval_reward_cz_20(action)
        else:
            raise error.Error('Unrecognized reward function type: {}'.format(self._reward_func))

    def eval_reward_cz_5(self, action):
        if self._game_over():
            reward = -10.0
        else:
            r_coord = goal_func_2(self.target_coordinates[:2] - self._goal_target[:2], 0.1)
            r_scale = goal_func_2(self.target_coordinates[2] - self._goal_target[2], 0.05)
            r_orientation_0 = aux_func(self.quadcopter_orientation[0]/numpy.pi - 0, 0.05) * 0.5
            r_orientation_1 = aux_func(self.quadcopter_orientation[1]/numpy.pi - 0, 0.05) * 0.5
            r_height = aux_func(self.quadcopter_pos[2] - self._goal_height, 0.5) * 0.5
            reward = r_coord + r_scale + r_orientation_0 + r_orientation_1 + r_height
            if self._log:
                logger.info(
                    'TargetLoc_x Reward:%.4f\nTargetLoc_y Reward:%.4f\nTargetLoc_h Reward:%.4f\nTargetPos Reward:%.4f\nW Reward:%f\nTotal Reward:%.4f\n' %
                    (
                        r_coord,
                        r_coord,
                        r_scale,
                        r_height,
                        r_orientation_0 + r_orientation_1,
                        reward,
                    )
                )

        return reward

    def eval_reward_cz_9(self, action):
        if self._game_over():
            reward = -10.0
        else:
            r_coord = goal_func_3(self.target_coordinates[:2] - self._goal_target[:2], [0.1, 0.25])
            r_scale = goal_func_3(self.target_coordinates[2] - self._goal_target[2], [0.05, 0.2])
            r_orientation_0 = aux_func(self.quadcopter_orientation[0] / numpy.pi - 0, 0.03) * 0.5
            r_orientation_1 = aux_func(self.quadcopter_orientation[1] / numpy.pi - 0, 0.03) * 0.5
            r_height = aux_func(self.quadcopter_pos[2] - self._goal_height, 0.5) * 0.5
            reward = r_coord + r_scale + r_orientation_0 + r_orientation_1 + r_height
            if self._log:
                logger.info(
                    'TargetLoc_x Reward:%.4f\nTargetLoc_y Reward:%.4f\nTargetLoc_h Reward:%.4f\nTargetPos Reward:%.4f\nW Reward:%f\nTotal Reward:%.4f\n' %
                    (
                        r_coord,
                        r_coord,
                        r_scale,
                        r_height,
                        r_orientation_0 + r_orientation_1,
                        reward,
                    )
                )

        return reward

    def eval_reward_cz_10(self, action):
        if self._game_over():
            reward = -10.0
        else:
            r_coord = goal_func_3(self.target_coordinates[:2] - self._goal_target[:2], [0.1, 0.25])
            r_scale = goal_func_3(self.target_coordinates[2] - self._goal_target[2], [0.05, 0.2])
            r_orientation_0 = aux_func(self.quadcopter_orientation[0]/numpy.pi - 0, 0.05) * 0.5
            r_orientation_1 = aux_func(self.quadcopter_orientation[1]/numpy.pi - 0, 0.05) * 0.5
            r_height = aux_func(self.quadcopter_pos[2] - self._goal_height, 0.5) * 0.5
            reward = r_coord + r_scale + r_orientation_0 + r_orientation_1 + r_height
            if self._log:
                logger.info(
                    'TargetLoc_x Reward:%.4f\nTargetLoc_y Reward:%.4f\nTargetLoc_h Reward:%.4f\nTargetPos Reward:%.4f\nW Reward:%f\nTotal Reward:%.4f\n' %
                    (
                        r_coord,
                        r_coord,
                        r_scale,
                        r_height,
                        r_orientation_0 + r_orientation_1,
                        reward,
                    )
                )

        return reward

    def eval_reward_cz_11(self, action):
        if self._game_over():
            reward = -10.0
        else:
            r_coord = goal_func_3(self.target_coordinates[:2] - self._goal_target[:2], [0.1, 0.25])
            r_scale = goal_func_3(self.target_coordinates[2] - self._goal_target[2], [0.05, 0.2])
            r_orientation_0 = aux_func(self.quadcopter_orientation[0] / numpy.pi - 0, 0.05) * 0.5
            r_orientation_1 = aux_func(self.quadcopter_orientation[1] / numpy.pi - 0, 0.05) * 0.5
            r_height = aux_func(self.quadcopter_pos[2] - self._goal_height, 0.5) * 0.5
            r_action = aux_func(action, 0.5) * 0.5
            reward = r_coord + r_scale + r_orientation_0 + r_orientation_1 + r_height + r_action
            if self._log:
                logger.info(
                    'TargetLoc_x Reward:%.4f\nTargetLoc_y Reward:%.4f\nTargetLoc_h Reward:%.4f\nTargetPos Reward:%.4f\nW Reward:%f\nTotal Reward:%.4f\n' %
                    (
                        r_coord,
                        r_coord,
                        r_scale,
                        r_height,
                        r_orientation_0 + r_orientation_1,
                        reward,
                    )
                )

        return reward

    def eval_reward_cz_12(self, action):
        if self._game_over():
            reward = -10.0
        else:
            r_coord = goal_func_3(self.target_coordinates[:2] - self._goal_target[:2], [0.1, 0.25])
            r_scale = goal_func_3(self.target_coordinates[2] - self._goal_target[2], [0.05, 0.2])
            r_orientation_0 = aux_func(self.quadcopter_orientation[0] / numpy.pi - 0, 0.05) * 0.5
            r_orientation_1 = aux_func(self.quadcopter_orientation[1] / numpy.pi - 0, 0.05) * 0.5
            r_height = aux_func(self.quadcopter_pos[2] - self._goal_height, 0.5) * 0.5
            r_action = aux_func(action, 0.3) * 0.5
            reward = r_coord + r_scale + r_orientation_0 + r_orientation_1 + r_height + r_action
            if self._log:
                logger.info(
                    'TargetLoc_x Reward:%.4f\nTargetLoc_y Reward:%.4f\nTargetLoc_h Reward:%.4f\nTargetPos Reward:%.4f\nW Reward:%f\nTotal Reward:%.4f\n' %
                    (
                        r_coord,
                        r_coord,
                        r_scale,
                        r_height,
                        r_orientation_0 + r_orientation_1,
                        reward,
                    )
                )

        return reward

    def eval_reward_cz_13(self, action):
        if self._game_over():
            reward = -10.0
        else:
            r_coord = goal_func_3(self.target_coordinates[:2] - self._goal_target[:2], [0.1, 0.25])
            r_scale = goal_func_3(self.target_coordinates[2] - self._goal_target[2], [0.05, 0.2])
            r_orientation_0 = aux_func(self.quadcopter_orientation[0] / numpy.pi - 0, 0.05) * 0.5
            r_orientation_1 = aux_func(self.quadcopter_orientation[1] / numpy.pi - 0, 0.05) * 0.5
            r_height = aux_func(self.quadcopter_pos[2] - self._goal_height, 0.5) * 0.5
            r_action = aux_func(action, 0.1) * 0.5
            reward = r_coord + r_scale + r_orientation_0 + r_orientation_1 + r_height + r_action
            if self._log:
                logger.info(
                    'TargetLoc_x Reward:%.4f\nTargetLoc_y Reward:%.4f\nTargetLoc_h Reward:%.4f\nTargetPos Reward:%.4f\nW Reward:%f\nTotal Reward:%.4f\n' %
                    (
                        r_coord,
                        r_coord,
                        r_scale,
                        r_height,
                        r_orientation_0 + r_orientation_1,
                        reward,
                    )
                )

        return reward

    def eval_reward_cz_20(self, action):
        if self._game_over():
            reward = -10.0
        else:
            r_coord = goal_func_3(self.target_coordinates[:2] - self._goal_target[:2], [0.1, 0.25])
            r_scale = goal_func_3(self.target_coordinates[2] - self._goal_target[2], [0.05, 0.2])
            r_orientation = aux_func(numpy.array(self.quadcopter_orientation[0:2]) / numpy.pi - 0, 0.03) * 1.0
            # r_orientation_0 = aux_func(self.quadcopter_orientation[0] / numpy.pi - 0, 0.03) * 0.5
            # r_orientation_1 = aux_func(self.quadcopter_orientation[1] / numpy.pi - 0, 0.03) * 0.5
            r_height = aux_func(self.quadcopter_pos[2] - self._goal_height, 0.5) * 0.5
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

    def _game_over(self):
        done = (not self.state_space.contains(self._get_state())) \
               or self.quadcopter_pos[2] <= 0 or self.quadcopter_pos[2] >= 5 \
               or abs(self.quadcopter_orientation[0]) >= numpy.pi/3 \
               or abs(self.quadcopter_orientation[1]) >= numpy.pi/3
        return done

    def __init__(self,
                 scene_path='/home/sliay/Documents/vrep-uav/scenes/quadcopter_hierarchy.ttt',
                 reward_func=0, log=False, action_type='continuous', discrete_type=0,
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
        self._goal_target = numpy.array([0., 0., 0.4])
        super(VREPHierarchyTargetEnv, self).__init__(**kwargs)

        # set action bound
        if self._action_type == 'continuous':
            self.action_space = spaces.Box(low=-1., high=1., shape=(4,))
            self._action_lb = -0.025
            self._action_ub = 0.025
        else:
            # self._action_granularity = numpy.array([0.01, 0.01, 0.01, 0.01])
            self._action_granularity = 0.02
            if self._discrete_type == 0:
                self.AVAILABLE_ACTION = AVAILABLE_ACTION_2 * self._action_granularity
            else:
                self.AVAILABLE_ACTION = AVAILABLE_ACTION_1 * self._action_granularity
            self.action_space = spaces.Discrete(self.AVAILABLE_ACTION.shape[0])
        state_bound = numpy.array([4, 4, 4] + [4, 4, 4] + # v, w
                                  # [4, 4, 4] + [4, 4, 4] + # a_v, a_w
                                  # [numpy.inf, numpy.inf, numpy.inf] +  # position
                                  [numpy.inf, ] +  # quadcopter height
                                  [1, 1, 1, 1] +  # quaternion
                                  # [numpy.pi/3, numpy.pi/3, numpy.inf] +  # quadcopter orientation
                                  # [numpy.inf, numpy.inf, numpy.inf] +  # target position
                                  [0.5, 0.5, 1] # normalized coordinates on camera plane
                                  )
        self.state_space = spaces.Box(low=-state_bound, high=state_bound)
        if self._obs_type == 'image':
            # self.observation_space = spaces.Box(low=0, high=255, shape=(3, 128, 128))
            self.observation_space = spaces.Box(low=0, high=255, shape=(3, 64, 64))
            self.observation_state_space = spaces.Box(low=-state_bound[0:-3], high=-state_bound[0:-3])
        elif self._obs_type == 'state':
            self.observation_space = spaces.Box(low=-state_bound, high=state_bound)
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

    def _reset(self):
        obs = super(VREPHierarchyTargetEnv, self)._reset()
        # logger.info('TargetLoc Height:%.2f' % self.target_coordinates[2])
        # self._goal_target = numpy.array([0., 0., self.target_coordinates[2]])
        if self._obs_type == 'image':
            state = self._get_state()
            return {'obs_img': obs, 'obs_state': state[0:-3]}
        else:
            return obs

    def _step(self, a):
        reward = 0.0
        if self._log:
            logger.info(
                'OriginalAction:%.4f\t%.4f\t%.4f\t%.4f' % (a[0], a[1], a[2], a[3])
            )
        if self._action_type == 'continuous':
            a = self._normalize_action(a)
        else:
            a = self.AVAILABLE_ACTION[a]
        for _ in range(self.frame_skip):
            reward += self._act(a)
        ob = self._get_obs()

        if self._log:
            logger.info(
                'TargetPos:%.4f\t%.4f\t%.4f' % (self.target_pos[0], self.target_pos[1], self.target_pos[2]))
            logger.info(
                'TargetCameraLoc:%.4f\t%.4f\t%.4f' % (
                self.target_coordinates[0], self.target_coordinates[1], self.target_coordinates[2]))
            logger.info(
                'QuadPos:%.4f\t%.4f\t%.4f' % (self.quadcopter_pos[0], self.quadcopter_pos[1], self.quadcopter_pos[2]))
            logger.info(
                'QuadOrient:%.4f\t%.4f\t%.4f' % (self.quadcopter_orientation[0], self.quadcopter_orientation[1], self.quadcopter_orientation[2])
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
        if terminal:
            reward -= self.R_terminal
        info = {}
        if self._obs_type == 'image':
            state = self._get_state()
            info = {'obs_state': state[0:-3]}
        return ob, reward, terminal, info

    # transform normalized action back
    def _normalize_action(self, action):
        lb = self._action_lb
        ub = self._action_ub
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = numpy.clip(scaled_action, lb, ub)
        return scaled_action

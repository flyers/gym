import logging
import math
import gym
from gym import spaces, error
import numpy
import subprocess
import os
import signal
import time

try:
    from gym.envs.vrep import vrep
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to perform the setup instructions here: http://www.coppeliarobotics.com/helpFiles/en/remoteApiClientSide.htm.)".format(e))


logger = logging.getLogger(__name__)

def quad2mat(q):
    mat = numpy.zeros((3, 3), dtype='float32')
    q = numpy.array(q)
    sq = q * q
    mat[0, 0] = numpy.array([1, -1, -1, 1]).dot(sq)
    mat[1, 1] = numpy.array([-1, 1, -1, 1]).dot(sq)
    mat[2, 2] = numpy.array([-1, -1, 1, 1]).dot(sq)

    xy = q[0] * q[1]
    zw = q[2] * q[3]
    mat[1, 0] = 2 * (xy + zw)
    mat[0, 1] = 2 * (xy - zw)

    xz = q[0] * q[2]
    yw = q[1] * q[3]
    mat[2, 0] = 2 * (xz - yw)
    mat[0, 2] = 2 * (xz + yw)

    yz = q[1] * q[2]
    xw = q[0] * q[3]
    mat[2, 1] = 2 * (yz + xw)
    mat[1, 2] = 2 * (yz - xw)

    return mat

# TODO, here there is a bug that the vrep server will crash with the progress of the env
class VREPEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }
    def _init_server(self):
        vrep_cmd = os.path.join(self.vrep_path, 'vrep')
        if self.headless:
            vrep_cmd += ' -h'
        vrep_arg = ' -gREMOTEAPISERVERSERVICE_'+str(self.remote_port)+'_FALSE_TRUE '
        execute_cmd = vrep_cmd + vrep_arg + self.scene_path + '&'
        logger.info('vrep launching command:%s' % execute_cmd)
        self.server_process = subprocess.Popen(execute_cmd, shell=True,
                                               # stdout=subprocess.PIPE,
                                               # stderr=subprocess.PIPE
                                               )
        self.server_process.wait()
        logger.info(self.server_process.pid)
        logger.info('server launch return code:%s' % self.server_process.poll())
        if self.server_process.poll() != 0:
            raise ValueError('vrep server launching failed')

    def _init_handle(self):
        # get object handles
        _, self.quadcopter_handle = vrep.simxGetObjectHandle(
            self.client_id, 'Quadricopter_base',vrep.simx_opmode_oneshot_wait)
        _, self.target_handle = vrep.simxGetObjectHandle(
            self.client_id, 'Bill', vrep.simx_opmode_oneshot_wait)
        _, self.camera_handle = vrep.simxGetObjectHandle(
            self.client_id, 'Quadricopter_frontSensor', vrep.simx_opmode_oneshot_wait)
        _, self.target_neck = vrep.simxGetObjectHandle(
            self.client_id, 'Mark_Head', vrep.simx_opmode_oneshot_wait)
        _, self.target_back = vrep.simxGetObjectHandle(
            self.client_id, 'Mark_Back', vrep.simx_opmode_oneshot_wait)
        _, self.target_leftfoot = vrep.simxGetObjectHandle(
            self.client_id, 'Mark_LeftFoot', vrep.simx_opmode_oneshot_wait)
        _, self.target_rightfoot = vrep.simxGetObjectHandle(
            self.client_id, 'Mark_RightFoot', vrep.simx_opmode_oneshot_wait)

    def _init_sensor(self):
        # enable streaming of state values and the observation image
        _, self.resolution, self.image = vrep.simxGetVisionSensorImage(
            self.client_id, self.camera_handle, 0, vrep.simx_opmode_streaming)
        _, self.linear_velocity_g, self.angular_velocity_g = vrep.simxGetObjectVelocity(
            self.client_id, self.quadcopter_handle, vrep.simx_opmode_streaming)
        _, self.quadcopter_pos = vrep.simxGetObjectPosition(
            self.client_id, self.quadcopter_handle, -1, vrep.simx_opmode_streaming)
        _, self.quadcopter_orientation = vrep.simxGetObjectOrientation(
            self.client_id, self.quadcopter_handle, -1, vrep.simx_opmode_streaming)
        _, self.target_pos = vrep.simxGetObjectPosition(
            self.client_id, self.target_handle, -1, vrep.simx_opmode_streaming)

        _, self.quadcopter_angular_variation = vrep.simxGetStringSignal(
            self.client_id, 'angular_variations', vrep.simx_opmode_streaming)
        self.quadcopter_angular_variation = vrep.simxUnpackFloats(
            self.quadcopter_angular_variation)
        _, self.quadcopter_quaternion = vrep.simxGetStringSignal(self.client_id, 'quaternion', vrep.simx_opmode_streaming)
        self.quadcopter_quaternion = vrep.simxUnpackFloats(
            self.quadcopter_quaternion)

        _, self.target_neck_pos = vrep.simxGetObjectPosition(
            self.client_id, self.target_neck, self.camera_handle, vrep.simx_opmode_streaming)
        _, self.target_back_pos = vrep.simxGetObjectPosition(
            self.client_id, self.target_back, self.camera_handle, vrep.simx_opmode_streaming)
        _, self.target_leftfoot_pos = vrep.simxGetObjectPosition(
            self.client_id, self.target_leftfoot, self.camera_handle, vrep.simx_opmode_streaming)
        _, self.target_rightfoot_pos = vrep.simxGetObjectPosition(
            self.client_id, self.target_rightfoot, self.camera_handle, vrep.simx_opmode_streaming)

    def _read_sensor(self):
        self._read_camera_image()

        _, self.linear_velocity_g, self.angular_velocity_g = vrep.simxGetObjectVelocity(
            self.client_id, self.quadcopter_handle, vrep.simx_opmode_buffer)
        _, self.quadcopter_pos = vrep.simxGetObjectPosition(
            self.client_id, self.quadcopter_handle, -1, vrep.simx_opmode_buffer)
        _, self.quadcopter_orientation = vrep.simxGetObjectOrientation(
            self.client_id, self.quadcopter_handle, -1, vrep.simx_opmode_buffer)
        _, self.target_pos = vrep.simxGetObjectPosition(
            self.client_id, self.target_handle, -1, vrep.simx_opmode_buffer)

        _, self.quadcopter_angular_variation = vrep.simxGetStringSignal(
            self.client_id, 'angular_variations', vrep.simx_opmode_buffer)
        self.quadcopter_angular_variation = vrep.simxUnpackFloats(
            self.quadcopter_angular_variation)
        _, self.quadcopter_quaternion = vrep.simxGetStringSignal(self.client_id, 'quaternion', vrep.simx_opmode_buffer)
        self.quadcopter_quaternion = vrep.simxUnpackFloats(
            self.quadcopter_quaternion)

        self.angular_velocity_b = self.quadcopter_angular_variation
        mat = quad2mat(self.quadcopter_quaternion)
        self.linear_velocity_b = mat.transpose().dot(self.linear_velocity_g)

        _, self.target_neck_pos = vrep.simxGetObjectPosition(
            self.client_id, self.target_neck, self.camera_handle, vrep.simx_opmode_buffer)
        _, self.target_back_pos = vrep.simxGetObjectPosition(
            self.client_id, self.target_back, self.camera_handle, vrep.simx_opmode_buffer)
        _, self.target_leftfoot_pos = vrep.simxGetObjectPosition(
            self.client_id, self.target_leftfoot, self.camera_handle, vrep.simx_opmode_buffer)
        _, self.target_rightfoot_pos = vrep.simxGetObjectPosition(
            self.client_id, self.target_rightfoot, self.camera_handle, vrep.simx_opmode_buffer)

        self._get_track_coordinates()

    def _get_obs(self):
        if self._obs_type == 'image':
            return self._get_image()
        elif self._obs_type == 'state':
            return self._get_state()
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

    def _get_image(self):
        # return image shape is 3 by height by width
        return self.image.transpose(2, 0, 1)

    def _get_state(self):
        # return the state representation
        return numpy.array(self.linear_velocity_b.tolist() +
                           self.angular_velocity_b +
                           self.target_coordinates.tolist(), dtype='float32')

    def _act(self, action):
        # send control signal to server side
        vrep.simxSetStringSignal(self.client_id, 'thrust',
                                 vrep.simxPackFloats(action), vrep.simx_opmode_oneshot)
        # trigger next simulation step
        vrep.simxSynchronousTrigger(self.client_id)
        # read sensor
        self._read_sensor()
        # get reward
        # TODO, adjust reward setting
        reward = 0.0
        reward += numpy.exp(-numpy.sum((self.target_coordinates - self._goal)**2))
        return reward

    def _game_over(self):
        done = (not self.state_space.contains(self._get_state())) \
               or self.quadcopter_pos[2] <= 0 or self.quadcopter_pos[2] >= 5 \
               or abs(self.quadcopter_orientation[0]) >= 1 \
               or abs(self.quadcopter_orientation[1]) >= 1
        return done

    def __init__(self, frame_skip=1, obs_type='state',
                 vrep_path='/home/sliay/Documents/V-REP_PRO_EDU_V3_3_1_64_Linux',
                 scene_path='/home/sliay/Dropbox/Documents/Phd/UAVControl/simulator/quadcopter_control_arena.ttt',
                 headless=True):
        self.frame_skip = frame_skip
        self._obs_type = obs_type
        self.vrep_path = vrep_path
        self.scene_path = scene_path
        self.headless = headless
        self.server_process = None

        # find an unoccupied port
        self.remote_port = 20000
        while vrep.simxStart('127.0.0.1', self.remote_port, True, True, 5000, 5) != -1:
            self.remote_port += 1
        # start a remote vrep server on this port
        self._init_server()
        # wait for the server initialization
        time.sleep(6)
        # now try to connect the server
        self.client_id = vrep.simxStart('127.0.0.1', self.remote_port, True, True, 5000, 5)
        self._goal = numpy.array([0., 0., 0.4])

        self.viewer = None

        if self.client_id == -1:
            raise error.Error('Failed connecting to remote API server')

        # set action bound
        self.action_space = spaces.Box(low=-1., high=1., shape=(4,))
        self._action_lb = 4.583759
        self._action_ub = 6.501942
        # set upper bound on linear velocity, angular velocity and target coordinates
        state_bound = numpy.array([2, 2, 2] + [1, 1, 1] + [0.5, 0.5, 0.5])
        self.state_space = spaces.Box(low=-state_bound, high=state_bound)
        if self._obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(3, 128, 128))
        elif self._obs_type == 'state':
            self.observation_space = spaces.Box(low=-state_bound, high=state_bound)
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

    def _reset(self):
        assert self.client_id != -1
        # stop the current simulation
        vrep.simxStopSimulation(self.client_id, vrep.simx_opmode_oneshot_wait)

        start_time = time.time()
        self._init_handle()
        end_time = time.time()
        logger.info('init handle time:%f' % (end_time-start_time))

        # init sensor reading
        start_time = time.time()
        self._init_sensor()
        end_time = time.time()
        logger.info('init read buffer time:%f' % (end_time-start_time))

        # enable the synchronous mode on the client
        vrep.simxSynchronous(self.client_id, True)
        # start the simulation, in blocking mode
        vrep.simxStartSimulation(self.client_id, vrep.simx_opmode_oneshot_wait)

        # trigger several simulation steps for api initialization
        for i in range(2):
            vrep.simxSynchronousTrigger(self.client_id)
        # read sensor data from server side
        self._read_sensor()

        return self._get_obs()

    def _step(self, a):
        reward = 0.0
        a = self._normalize_action(a)
        for _ in range(self.frame_skip):
            reward += self._act(a)
        ob = self._get_obs()

        return ob, reward, self._game_over(), {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self.image
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer == None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(numpy.uint8(img))

    def _close(self):
        # close the vrep server process, whose pid is the parent pid plus 1
        try:
            os.kill(self.server_process.pid+1, signal.SIGKILL)
        except OSError:
            logger.info('Process does not exist')

    # helper function to get the rgb image from vrep simulator
    def _read_camera_image(self):
        _, self.resolution, self.image = vrep.simxGetVisionSensorImage(
            self.client_id, self.camera_handle, 0, vrep.simx_opmode_buffer)
        # image shape is height by width by 3
        self.image = numpy.array(self.image).reshape((self.resolution[1], self.resolution[0], 3))
        self.image = numpy.flipud(self.image)
        index = numpy.zeros(self.image.shape, dtype=self.image.dtype)
        index[self.image < 0] = 1
        self.image += 256 * index
        self.image = numpy.uint8(self.image)

    # get the target coordinates on the camera image plane
    def _get_track_coordinates(self):
        # use built in APIs in V-REP to get the target position on the camera image
        # for scale, we only consider the height information and ignore the width
        cx = self.target_back_pos[0] / self.target_back_pos[2]
        y_top = self.target_neck_pos[1] / self.target_neck_pos[2]
        y_bottom = (self.target_leftfoot_pos[1]/self.target_leftfoot_pos[2] +
                    self.target_rightfoot_pos[1]/self.target_rightfoot_pos[2]) / 2.0
        h = abs(y_bottom - y_top)
        cy = (y_bottom + y_top) / 2.0
        self.target_coordinates = numpy.array([cx, cy, h])

    # transform normalized action back
    def _normalize_action(self, action):
        lb = self._action_lb
        ub = self._action_ub
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = numpy.clip(scaled_action, lb, ub)
        return scaled_action


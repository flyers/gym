import logging
import numpy
import time
import socket
import struct
from gym import error, spaces
from gym.envs.vrep.vrep_base import VREPBaseEnv

try:
    from gym.envs.vrep import vrep
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to perform the setup instructions here: http://www.coppeliarobotics.com/helpFiles/en/remoteApiClientSide.htm.)".format(
            e))

logger = logging.getLogger(__name__)


def serve_socket(port):
    '''
    Serves a blocking socket on the specified port.  Returns a new socket object
    representing the client, on which send() and recv() can be invoked.
    '''
    print('Server: running on port %d' % port)
    sock = None
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('', port))
            break
        except socket.error as err:
            print('Server: bind failed: ' + str(err))
            exit(1)
        time.sleep(1)
    sock.listen(1)

    return sock


def serve_client(sock):
    print('Server: waiting for client to connect ...')
    try:
        client, _ = sock.accept()
        print('Server: accepted connection')
    except:
        print('Failed')
        exit(1)

    return client


def sendFloats(client, data):
    client.send(struct.pack('%sf' % len(data), *data))


def unpackFloats(msg, nfloats):
    return struct.unpack('f' * nfloats, msg)


def receiveFloats(client, nfloats):
    # We use 32-bit floats
    msgsize = 4 * nfloats

    # Implement timeout
    start_sec = time.time()
    remaining = msgsize
    msg = ''
    while remaining > 0:
        msg += client.recv(remaining)
        remaining -= len(msg)
        if (time.time() - start_sec) > 1.0:
            return None

    return unpackFloats(msg, nfloats)


def receiveString(client):
    return client.recv(int(receiveFloats(client, 1)[0]))

class VREPMixEnv(VREPBaseEnv):

    def _init_sensor(self):
        super(VREPMixEnv, self)._init_sensor()

    def _read_sensor(self):
        super(VREPMixEnv, self)._read_sensor()

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

    def _choose_action(self, prob, pid_action, policy_action):
        vrep.simxSetStringSignal(self.client_id, 'control_signal',
                                 vrep.simxPackFloats(pid_action), vrep.simx_opmode_oneshot)
        # trigger next simulation step
        vrep.simxSynchronousTrigger(self.client_id)
        motor_action = receiveFloats(self.socket_client, 4)
        do_exploration = (numpy.random.rand() < prob)
        if do_exploration:
            motor_action = self._raw_action2norm(motor_action)
            # logger.info(
            #     'QuadMotor: %.4f\t%.4f\t%.4f\t%.4f\n' % (motor_action[0], motor_action[1], motor_action[2], motor_action[3]))
        else:
            motor_action = policy_action
        return motor_action

    def _act(self, motor_action):
        sendFloats(self.socket_client, motor_action)
        # read sensor
        self._read_sensor()

        motor_action = numpy.array(motor_action)
        if self._reward_func == 1:
            return self.eval_reward1(motor_action)
        elif self._reward_func == 2:
            return self.eval_reward2(motor_action)
        elif self._reward_func == 3:
            return self.eval_reward3(motor_action)
        elif self._reward_func == 0:
            return self.eval_reward_dy(motor_action)
        else:
            raise error.Error('Unrecognized reward function type: {}'.format(self._reward_func))

    def eval_reward1(self, action):
        # reward setting
        # refer to paper: Learning Deep Control Policies for Autonomous Aerial Vehicles with MPC-Guided Policy Search
        if self._state_type == 'world':
            v = self.linear_velocity_g
            w = self.angular_velocity_g
        else:
            v = self.linear_velocity_b
            w = self.angular_velocity_b
        cost = \
            1e3 * numpy.square(self.quadcopter_pos[2] - self._goal_height) + \
            1e3 * numpy.square(self.quadcopter_quaternion - numpy.array([0, 0, 1, 0])).sum() + \
            5e2 * numpy.square(numpy.array(v) - [1, 0, 0]).sum() + \
            5e2 * numpy.square(numpy.array(w) - [0., 0., 0.]).sum() + \
            1e1 * numpy.square(action - 5.335).sum()
        reward = numpy.exp(-cost / 1e3)
        return numpy.asscalar(reward)


    def eval_reward2(self, action):
        if self._state_type == 'world':
            v = self.linear_velocity_g
            w = self.angular_velocity_g
        else:
            v = self.linear_velocity_b
            w = self.angular_velocity_b
        reward = \
            1e1 * numpy.exp(-numpy.square(self.quadcopter_pos[2] - self._goal_height)) + \
            1e1 * numpy.exp(-numpy.square(self.quadcopter_quaternion - numpy.array([0, 0, 1, 0])).sum()) + \
            5e0 * numpy.exp(-numpy.square(numpy.array(v) - [1, 0, 0]).sum()) + \
            5e0 * numpy.exp(-numpy.square(numpy.array(w) - [0, 0, 0]).sum()) + \
            1e0 * numpy.exp(-numpy.square(action - 5.335).sum())
        return numpy.asscalar(reward)

    def eval_reward3(self, action):
        if self._state_type == 'world':
            v = self.linear_velocity_g
            w = self.angular_velocity_g
        else:
            v = self.linear_velocity_b
            w = self.angular_velocity_b
        reward = \
            numpy.exp(-numpy.square(self.quadcopter_pos[2] - self._goal_height)) + \
            numpy.exp(-numpy.square(self.quadcopter_quaternion - numpy.array([0, 0, 1, 0])).sum()) + \
            numpy.exp(-numpy.square(numpy.array(v) - [1, 0, 0]).sum()) + \
            numpy.exp(-numpy.square(numpy.array(w) - [0, 0, 0]).sum()) + \
            0.5 * numpy.exp(-numpy.square(action - 5.335).sum())
        if abs(self.quadcopter_orientation[0]) > numpy.pi / 6 or abs(self.quadcopter_orientation[1]) > numpy.pi / 6:
            reward -= 1
        return numpy.asscalar(reward)

    def eval_reward_dy(self, action):
        if self._state_type == 'world':
            v = self.linear_velocity_g
            w = self.angular_velocity_g
        else:
            v = self.linear_velocity_b
            w = self.angular_velocity_b
        reward_p = \
            numpy.exp(-numpy.square(self.quadcopter_pos[2] - self._goal_height))
        p_baseline = numpy.exp(-numpy.square([0.2]))
        reward_q = \
            numpy.exp(-numpy.square(self.quadcopter_quaternion - numpy.array([0, 0, 1, 0])).sum())
        q_baseline = numpy.exp(-numpy.square([0.3, 0.3, 0.3, 0.3]).sum())
        reward_v = \
            numpy.exp(-numpy.square(numpy.array(v) - [1, 0, 0]).sum())
        v_baseline = numpy.exp(-numpy.square([0.2, 0.2, 0.2]).sum())
        reward_w = \
            numpy.exp(-numpy.square(numpy.array(w) - [0, 0, 0]).sum())
        w_baseline = numpy.exp(-numpy.square([0.2, 0.2, 0.2]).sum())
        reward_u = \
            numpy.exp(-numpy.square(action - 5.335).sum())
        u_baseline = numpy.exp(-numpy.square([1, 1, 1, 1]).sum())
        reward = \
            reward_p + reward_q + reward_v + reward_w + reward_u - \
            (p_baseline + q_baseline + v_baseline + w_baseline + u_baseline) + 0.5
        # logger.info(
        #     'Position Reward:%.4f-%.4f=%.4f\nQuaternion Reward:%.4f-%.4f=%.4f\nV Reward:%.4f-%.4f=%.4f\nW Reward:%.4f-%.4f=%.4f\nMotor Reward:%.4f-%.4f=%.4f\nTotal Reward:%.2f' %
        #     (
        #         reward_p, p_baseline, reward_p - p_baseline,
        #         reward_q, q_baseline, reward_q - q_baseline,
        #         reward_v, v_baseline, reward_v - v_baseline,
        #         reward_w, w_baseline, reward_w - w_baseline,
        #         reward_u, u_baseline, reward_u - u_baseline,
        #         reward,
        #     )
        # )
        return numpy.asscalar(reward)

    def _game_over(self):
        done = (not self.state_space.contains(self._get_state())) \
               or self.quadcopter_pos[2] <= 0 or self.quadcopter_pos[2] >= 5 \
               or abs(self.quadcopter_orientation[0]) >= numpy.pi/3 \
               or abs(self.quadcopter_orientation[1]) >= numpy.pi/3
        return done

    def __init__(self,
                 socket_port=21000,
                 scene_path='/home/sliay/Documents/vrep-uav/scenes/quadcopter_mix.ttt',
                 reward_func=1, **kwargs):
        self._reward_func = reward_func
        self.scene_path = scene_path
        self._goal_height = 1.5
        self.socket_port = socket_port

        super(VREPMixEnv, self).__init__(**kwargs)

        self.socket = serve_socket(self.socket_port)
        self.socket_client = None

        # set action bound
        self.action_space = spaces.Box(low=-1., high=1., shape=(4,))
        self._action_lb = 4.52
        self._action_ub = 6.15
        state_bound = numpy.array([4, 4, 4] + [4, 4, 4] +  # v, w
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

    def _reset(self):
        assert self.client_id != -1
        # stop the current simulation
        vrep.simxStopSimulation(self.client_id, vrep.simx_opmode_oneshot_wait)

        if self.socket_client is not None:
            self.socket_client.close()

        start_time = time.time()
        self._init_handle()
        end_time = time.time()
        logger.info('init handle time:%f' % (end_time - start_time))

        # init sensor reading
        start_time = time.time()
        self._init_sensor()
        end_time = time.time()
        logger.info('init read buffer time:%f' % (end_time - start_time))

        vrep.simxSetIntegerSignal(self.client_id, 'socket_port', self.socket_port, vrep.simx_opmode_oneshot)

        # enable the synchronous mode on the client
        vrep.simxSynchronous(self.client_id, True)
        # start the simulation, in blocking mode
        vrep.simxStartSimulation(self.client_id, vrep.simx_opmode_oneshot_wait)

        # random initialization
        if self.random_start:
            a = numpy.random.uniform(low=3, high=6)
            b = numpy.random.uniform(low=-1, high=1)
            vrep.simxSetObjectPosition(self.client_id, self.target_handle, self.quadcopter_handle, [a, b, -1.5],
                                       vrep.simx_opmode_oneshot_wait)

        # trigger several simulation steps for api initialization
        for i in range(2):
            vrep.simxSynchronousTrigger(self.client_id)

        self.socket_client = serve_client(self.socket)
        tmp = receiveString(self.socket_client)
        print 'String: ', tmp

        # read sensor data from server side
        self._read_sensor()

        return self._get_obs()

    def _step(self, a):
        reward = 0.0
        scaled_action = self._normalize_action(a)
        for _ in range(self.frame_skip):
            reward += self._act(scaled_action)
        ob = self._get_obs()

        # logger.info(
        #     'QuadPos: %.4f\t%.4f\t%.4f\n' % (self.quadcopter_pos[0], self.quadcopter_pos[1], self.quadcopter_pos[2]))
        # logger.info(
        #     'QuadLinearVel: %.4f\t%.4f\t%.4f\n' % (
        #     self.linear_velocity_g[0], self.linear_velocity_g[1], self.linear_velocity_g[2]))
        # logger.info(
        #     'QuadAngVel: %.4f\t%.4f\t%.4f\n' % (
        #     self.angular_velocity_g[0], self.angular_velocity_g[1], self.angular_velocity_g[2]))
        # logger.info(
        #     'QuadLinearBodyVel: %.4f\t%.4f\t%.4f\n' % (
        #     self.linear_velocity_b[0], self.linear_velocity_b[1], self.linear_velocity_b[2]))
        # logger.info(
        #     'QuadAngBodyVel: %.4f\t%.4f\t%.4f\n' % (
        #     self.angular_velocity_b[0], self.angular_velocity_b[1], self.angular_velocity_b[2]))
        # logger.info('Reward:%f\n' % reward)

        return ob, reward, self._game_over(), {'raw_action': scaled_action, 'norm_action': a}

    # transform normalized action back
    def _normalize_action(self, action):
        lb = self._action_lb
        ub = self._action_ub
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = numpy.clip(scaled_action, lb, ub)
        return scaled_action

    def _raw_action2norm(self, action):
        lb = self._action_lb
        ub = self._action_ub
        action = (numpy.array(action) - lb) * 2. / (ub - lb) - 1
        action = numpy.maximum(numpy.minimum(action, 1), -1)
        return action

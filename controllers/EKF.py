#!/usr/bin/env python
import numpy as np
import scipy as sp
import scipy.signal as sig

import pyfmi as fmi

import helpers


class EKF(object):
    def __init__(self, simulator, settings):
        self.nonlin = simulator
        self.Ts = settings['Ts']

        self.states = self.nonlin.sim.get_states_list()
        self.inputs = self.nonlin.sim.get_input_list()

        def sub_dict(dict, remove):
            original = dict.keys()
            # Ignore selected states
            for key in remove:
                # Save the index
                remove[key] = original.index(key)
                # Remove
                dict.pop(key)

            return dict

        self.ignored_states = settings['ignored_states']
        self.not_outputs = settings['not_measured_states']

        self.states = sub_dict(self.states.copy(), self.ignored_states)
        self.outputs = sub_dict(self.states.copy(), self.not_outputs)

        self.n_states = len(self.states)
        self.n_outputs = len(self.outputs)

        self.Ak = np.matrix([])

        self.Ck = np.diag(np.ones(self.n_states))
        self.Ck = np.delete(self.Ck, (self.not_outputs.values()), axis=0)
        self.Ck = np.asmatrix(self.Ck)

        self.P = np.diag(np.power(settings['P_std'], 2))
        self.Q = np.diag(np.power(settings['Q_std'], 2))
        self.R = np.diag(np.power(settings['R_std'], 2))

        self.x_hat = np.array([])
        self.y_hat = np.array([])

    def time_update(self, k, x_hat, u_k):
        res = self.nonlin.simulate(
            state=x_hat,
            start_time=k*self.Ts,
            final_time=(k+1)*self.Ts,
            input=u_k,
            append=True
        )

        # Linearize
        Ak, Bk, Ck, Dk = self.nonlin.sim.get_state_space_representation()
        # Discretize
        (Ak, Bk, Ck, Dk, dt) = sig.cont2discrete((
            Ak.toarray(), Bk.toarray(), Ck.toarray(), Dk.toarray()
        ), self.Ts)

        self.Ak = Ak

        # Remove ignored states
        self.Ak = np.delete(self.Ak, (self.ignored_states.values()), axis=0)
        self.Ak = np.delete(self.Ak, (self.ignored_states.values()), axis=1)
        # Turn into numpy matrix, for matrix multiplication with *
        self.Ak = np.asmatrix(self.Ak)

        # Obtain state values
        x_hat = []
        for state in self.states:
            x_hat.append(res[state][-1])

        # Transpose, to obtain a state column vector
        self.x_hat = np.array([x_hat]).T

        # Calculate measurement column vector
        self.y_hat = self.Ck * self.x_hat

        # Obtain new error covariance
        self.P = self.Ak * self.P * self.Ak.T + self.Q

    def measurement_update(self, y):
        # Calculate Kalman gain
        inv = np.linalg.inv
        K = self.P * self.Ck.T * inv(self.Ck * self.P * self.Ck.T + self.R)

        # Update state estimate
        self.x_hat = self.x_hat + K * (y - self.y_hat)

        # Update error covariance
        I = np.eye(self.n_states)
        self.P = (I - K*self.Ck) * self.P * (I - K*self.Ck).T

    def get_x_hat(self):
        # Prep state for use in simulation (return dictionary w. start values)
        x_hat = {}
        for i, state in enumerate(self.states):
            x_hat['_start_' + state] = self.x_hat.item(i)

        return x_hat

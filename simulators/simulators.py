#!/usr/bin/env python
import os
import pickle

import numpy as np
import scipy as sp
import pandas as pd

import pyjmi as jmi
import pyfmi as fmi
import pymodelica as pym

import helpers

from pyfmi.common.io import ResultDymolaTextual
from pyfmi.common.algorithm_drivers import JMResultBase

# MONKEY PATCHING PYFMI!

# Append simulation data (res: ResultDymolaTextual), without
# shifting time as the original method.
def append_dym(self, res): self.data[1] = np.vstack((self.data[1],res.data[1]))
ResultDymolaTextual.append = append_dym

# Add an append method to JMResultBase, utilizing append method of
# ResultDymolaTextual.
def append_jm(self, res):
    self._result_data.append(res._result_data)
JMResultBase.append = append_jm


class FMUExtension(object):
    def __init__(self):
        self.res = {}

    def append(self, sim_res):
        # If first simulation, use as basis for appending, else append
        if not self.res:
            self.res = sim_res
        else:
            self.res.append(sim_res)

    def aggregated_result(self):
        return helpers.aggregated_result(self.res)

    def simulate(self, state=False, append=False, *args, **kwargs):
        if state:
            self.reset()
            self.set(state.keys(), state.values())

        sim_res = self.sim.simulate(*args, **kwargs)
        
        if append:
            self.append(sim_res)

        return sim_res

    def reset(self, *args, **kwargs):
        return self.sim.reset(*args, **kwargs)

    def get(self, *args, **kwargs):
        return self.sim.get(*args, **kwargs)

    def set(self, *args, **kwargs):
        return self.sim.set(*args, **kwargs)

class SimulatorC2(FMUExtension):
    def __init__(self, models):
        super(SimulatorC2, self).__init__()
        self.name = self.__class__.__name__

        # Modelica package dependencies
        self.deps = ['FluidJM', 'DHP']

        # Path to dependencies and problem
        self.paths = helpers.path(self.deps, models['file'])

        # Compiler options
        self.options = {
            "state_initial_equations": True,
            "enable_variable_scaling": True
        }

        self.simulator_model = models['sim']
        self.optimization_problem = models['opt']

        self.load()

    def load(self):
        self.SIM_FMU = self.compile_FMU(self.simulator_model)
        self.sim = fmi.load_fmu(self.SIM_FMU)
        self.problem = self.compile_OPT(self.optimization_problem)

        self.x_ss = self.get_SS()
        self.nominal_trajectory = self.get_nominal_trajectory()
        self.initial_trajectory = self.get_initial_trajectory()

        # HACK:
        # Reset cost (can't be done before nominal trajectory), as
        # it messes with scaling and solver fails...
        self.x_ss['_start_MPC.cost'] = 0

    def compile_FMU(self, model):
        FMU_dir = 'FMUs'
        FMU_file = '{0}/{1}.fmu'.format(FMU_dir, self.name)
        
        # If FMU is compiled, use that
        if os.path.isfile(FMU_file):
            print "Using cached FMU..."
            return FMU_file

        FMU = pym.compile_fmu(
            model, self.paths,
            compiler_options=self.options,
            version='2.0',
            compiler_log_level='error',
            compile_to=FMU_file
        )

        return FMU

    def compile_OPT(self, model):
        problem = jmi.transfer_optimization_problem(
            model, self.paths,
            compiler_options=self.options,
            compiler_log_level='error'
        )
        return problem

    def get_SS(self):
        res = self.simulate(start_time=0, final_time=3600000)
        self.reset()
        # Get state names and their end value
        x_ss = helpers.get_x(self.sim, res, -1)
        return x_ss

    def get_nominal_trajectory(self):
        # Set start values
        self.set(self.x_ss.keys(), self.x_ss.values())

        # Nominal input: use max for all inputs
        limits = []
        names = []
        
        for inp in self.problem.getVariables(self.problem.REAL_INPUT):
            u_max = self.get('MPC.par.' + inp.getName() + '_max')[0]
            names.append(inp.getName())
            limits.append(u_max)

        def u(t): return np.array(limits)

        # Simulate
        res = self.simulate(start_time=0., final_time=3600, input=(names, u))
        self.reset()
        return res

    def get_initial_trajectory(self):
        # Set start values
        self.set(self.x_ss.keys(), self.x_ss.values())
        # Simulate
        res = self.simulate(start_time=0., final_time=3600)
        self.reset()
        return res

    def get_state(self, sim_res=False):
        if sim_res:
            return helpers.get_x(self.sim, sim_res, -1)

        return helpers.get_x(self.sim, self.res, -1)


class LinearSimulatorC2(SimulatorC2):
    def __init__(self, models):
        # Injected linearization (check if cached to file first):
        LINSYS_dir = 'LINSYS'
        LINSYS_file = '{0}/LINSYS.pickle'.format(LINSYS_dir)
        
        # If LINSYS available, use that
        if os.path.isfile(LINSYS_file):
            print "Using cached LINSYS..."
            with open(LINSYS_file, 'rb') as file:
                self.linsys = pickle.load(file)
        else:
            # Linearize and save the state-space system to file (pickle it)
            self.linsys = self.obtain_linsys(
                models['nonlin'], models['u_lin'], models['t_lin']
            )
            with open(LINSYS_file, 'wb') as file:
                pickle.dump(self.linsys, file)

        # Build map between nonlin states and lin states (look-up table)
        self.build_state_map()
        

        # Call constructor for super-class
        super(LinearSimulatorC2, self).__init__(models)

    def obtain_linsys(self, model, u_lin, t_lin):
        def u(t): return np.array(u_lin.values())
        linsys = helpers.linearize(model, (u_lin.keys(), u), t_lin)
        # NB: In Modelica, D is used for the consumer dissatisfaction
        linsys['Dx'] = linsys['D']
        return linsys

    def load_state_space_system(self, model):
        model_str = 'dhloop.loopLINBase.{0}'
        system_matrices = ['A', 'B', 'C', 'Dx']
        operating_point = ['x0', 'u0', 'y0']
        
        for matrix in system_matrices:
            helpers.set_matrix_parameter(
                model, model_str.format(matrix), self.linsys[matrix]
            )
        
        for vector in operating_point:
            helpers.set_vector_parameter(
                model, model_str.format(vector), self.linsys[vector]
            )


    def load(self):
        self.SIM_FMU = self.compile_FMU(self.simulator_model)
        self.sim = fmi.load_fmu(self.SIM_FMU)
        self.problem = self.compile_OPT(self.optimization_problem)

        # Load state-space system in opt. problem
        self.load_state_space_system(self.sim)
        self.load_state_space_system(self.problem)

        self.x_ss = self.get_SS()
        self.nominal_trajectory = self.get_nominal_trajectory()
        self.initial_trajectory = self.get_initial_trajectory()

        # HACK:
        # Reset cost (can't be done before nominal trajectory), as
        # it messes with scaling and solver fails...
        self.x_ss['_start_MPC.cost'] = 0

    def simulate(self, state=False, append=False, *args, **kwargs):
        if state:
            self.reset()
            self.set(state.keys(), state.values())

        # Hot-load state-space system in simulation FMU
        # WHY: When .reset is called on FMU, parameters return to default
        self.load_state_space_system(self.sim)
        sim_res = self.sim.simulate(*args, **kwargs)
    
        if append:
            self.append(sim_res)

        return sim_res

    def build_state_map(self):
        # The order in 'xn' correspond to the naming scheme for linear states
        # E.g.: first entry in 'xn' => dhsys.x[1]
        names = self.linsys['xn']
        x0 = self.linsys['x0']
        nonlin_str = "_start_{0}"
        lin_str = "_start_dhloop.loopLINBase.x[{0}]"
        state_map = {}
        x0_map = {}
        for n in range(len(names)):
            state_map[nonlin_str.format(names[n])] = lin_str.format(n + 1)
            x0_map[nonlin_str.format(names[n])] = x0[n][0]

        self.state_map = state_map
        self.x0_map = x0_map

    def map_state(self, x):
        # Map between nonlin and lin states
        x_mapped = {}
        
        for x_i in x.keys():
            # If some states don't exist in map, then we don't transform them
            try:
                x_mapped[self.state_map[x_i]] = x[x_i] - self.x0_map[x_i]
            except:
                x_mapped[x_i] = x[x_i]
        
        return x_mapped


class Simulator(FMUExtension):
    def __init__(self, models):
        super(Simulator, self).__init__()

        FMU_dir = 'FMUs'
        FMU_file = '{0}/{1}'.format(FMU_dir, models['FMU'])
        
        # If FMU is compiled, use that
        if os.path.isfile(FMU_file):
            print "Using cached FMU..."
        else:
            # Compile sim FMU using Dymola script
            print "Compiling new sim FMU (calling Dymola)..."
            os.system('dymola -nowindow gen_dhp_fmu.mos')
            print "... Done! ({0})".format(FMU_file)

        self.sim = fmi.load_fmu(FMU_file)
        self.state_ss = self.get_SS(models['stoch_params'])

    def get_SS(self, stochastic_consumer_parameters=False):
        self.sim.simulate(final_time=1)

        if stochastic_consumer_parameters:
            # Initialize simulation w. stochastic params for each group
            houses = range(1, self.sim.get('dhloop.house.N_types')[0])
            for house in houses:
                for par, std in stochastic_consumer_parameters.iteritems():
                    par = par.format(house)
                    self.sim.set(par, self.sim.get(par) + std * np.random.randn())

        start = self.get_fmu_state()
        self.simulate(state=start, start_time=0, final_time=3.6e6)
        self.set('MPC.cost', 0)
        # Return state of FMU in steady-state
        return self.get_fmu_state()

    def simulate(self, state=False, append=False, *args, **kwargs):
        if state:
            self.sim.setup_experiment()
            if 'options' in kwargs:
                kwargs['options']['initialize'] = False
            else:
                kwargs['options'] = {'initialize': False}
            self.set_fmu_state(state)
        
        sim_res = self.sim.simulate(*args, **kwargs)
        
        if append:
            self.append(sim_res)

        return sim_res

    def get_fmu_state(self, *args, **kwargs):
        return self.sim.get_fmu_state(*args, **kwargs)

    def set_fmu_state(self, *args, **kwargs):
        return self.sim.set_fmu_state(*args, **kwargs)

    def build_state_map(self, C2_states):
        self.state_map = {}
        sim_vars = self.sim.get_model_variables()

        n_pipeT = len('pipe.T')
        n_start = len('_start_')

        for C2_state in C2_states:
            sim_state = C2_state[n_start:]
            if sim_state in sim_vars:
                self.state_map[C2_state] = sim_state
            else:
                # Try to swap pipe.T for pipe.state_b.T (DynamicPipe)
                sim_state = C2_state[n_start:-n_pipeT] + 'pipe.mediums[1].T'
                if sim_state in sim_vars:
                    self.state_map[C2_state] = sim_state

    def get_state(self, sim_res = False):
        state = {}
        # Loop through map, find values corresponding to C2 state
        for C2_state, sim_state in self.state_map.iteritems():
            if sim_res:
                state[C2_state] = sim_res[sim_state][-1]
            else:
                state[C2_state] = self.res[sim_state][-1]

        return state

    def get_measurement(self, outputs, noise_std = False, sim_res = False):
        y = []
        # If simulation data not supplied, use aggregated data
        if not sim_res:
            sim_res = self.res

        # Loop through outputs and construct vector
        for i, measurement in enumerate(outputs):
            # Sample noise
            if noise_std:
                noise = noise_std[i] * np.random.randn()
            else:
                noise = 0.05 * np.random.randn()

            real = sim_res[self.state_map['_start_' + measurement]][-1]
            meas = real + noise

            # HACK:
            # If accu E measurement and value less than zero, then set to zero
            # to avoid running out of constraints
            if measurement == 'dhloop.accumulator.accumulator.E' and meas < 0:
                meas = 0
            
            y.append(meas)
        
        return np.array([y]).T
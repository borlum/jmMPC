#!/usr/bin/env python
import numpy as np


class MPCFail(Exception):
    pass


class MPC(object):
    def __init__(self, problem, external, init_traj, nominal_traj, settings):
        self.problem = problem
        # Sample time
        self.Ts = settings['Ts']
        # Horizon size (in # samples)
        self.N_predict = settings['N_predict']
        # Number of collocation points (points pr. sample)
        self.N_coll_points = settings['N_c']
        # Update scaling? (Scaling values as parameters in NLP)
        self.mutable_scaling = settings['mutable_scaling']
        # External data
        self.external = external
        # Accepted return status
        self.accepted = ['Solve_Succeeded', 'Solved_To_Acceptable_Level']
        # Fallback inputs (last working inputs)
        self.u_k = ()
        # Solution statistics
        self.stats = []

        self._set_options(init_traj, nominal_traj)
        self._transcribe_to_NLP()

    def _create_external_data(self):
        from pyjmi.optimization.casadi_collocation import ExternalData
        return ExternalData(eliminated=self.external)

    def _set_options(self, init_traj, nominal_traj):
        from pyjmi.optimization.casadi_collocation import BlockingFactors
        defaults = self.problem.optimize_options()

        # Discretization options
        defaults['n_e'] = self.N_predict
        defaults['n_cp'] = self.N_coll_points

        # Init + scaling
        defaults['init_traj'] = init_traj
        defaults['nominal_traj'] = nominal_traj
        defaults['equation_scaling'] = True
        # To allow updating nominal traj at every sample
        defaults['variable_scaling_allow_update'] = self.mutable_scaling

        # Create the external data (eliminated variables)
        defaults['external_data'] = self._create_external_data()

        # Blocking Factors (obtain constant input through sample)
        # Assuming 1 el./sample (class does not allow for anything else atm)
        # Look at Axelssons thesis for description of blocking factors
        factors = {}
        for inp in self.problem.getVariables(self.problem.REAL_INPUT):
            if not inp.getName() in self.external.keys():
                factors[inp.getName()] = [1]*defaults['n_e']
        defaults['blocking_factors'] = BlockingFactors(factors=factors)

        # IPOPT
        defaults['IPOPT_options']['max_iter'] = 300
        defaults['IPOPT_options']['tol'] = 1e-8
        defaults['IPOPT_options']['print_level'] = 5

        self.optimization_options = defaults

    def _transcribe_to_NLP(self):
        solver = self.problem.prepare_optimization(
            options=self.optimization_options
        )

        # Warm start options
        solver.set_solver_option('IPOPT', 'warm_start_init_point', 'yes')
        solver.set_solver_option('IPOPT', 'mu_init', 1e-1)

        solver.set_solver_option('IPOPT', 'warm_start_bound_push', 1e-4)
        solver.set_solver_option('IPOPT', 'warm_start_mult_bound_push', 1e-4)
        solver.set_solver_option('IPOPT', 'warm_start_bound_frac', 1e-4)
        solver.set_solver_option('IPOPT', 'warm_start_slack_bound_frac', 1e-4)
        solver.set_solver_option('IPOPT', 'warm_start_slack_bound_push', 1e-4)

        self.solver = solver

    def _shift_external_data(self, k):
        for variable in self.external:
            # Data length
            length = self.external[variable].shape[1]
            # Shifting matrix (shifts time vector by -k*Ts)
            shift = np.vstack([
                np.ones(length) * k * -self.Ts, np.zeros(length)
            ])
            # Shifted data
            shifted = np.add(self.external[variable], shift)
            # Update it in NLP
            self.solver.set_external_variable_data(variable, shifted)

    def _accepted_solution(self, k):
        if k < 0:
            return False
        return self.stats[k][1][0] in self.accepted

    def update_scaling(self, nominal_traj):
        # Update scaling factors
        if self.mutable_scaling:
            self.solver.set_nominal_traj(nominal_traj)

    def update(self, k, x):
        solver = self.solver
        solver.set(x.keys(), x.values())
        # Shift time
        solver.set('startTime', k*self.Ts)
        solver.set('finalTime', (self.N_predict+k)*self.Ts)
        # Shift external data
        self._shift_external_data(k)

    def sample(self, k):
        names = []
        inputs = []
        prob = self.problem
        solver = self.solver

        # Solve optimization problem
        res = solver.optimize()

        # Check constraints of new solution
        self.check_constraints(res)
        # Save solver statistics
        self.stats.append([k, res.get_solver_statistics()])
        # Extract solution time (will return it to loop)
        solution_time = self.stats[k][1][3]

        # If first run; enable warm start of solver
        if k == 0:
            solver.set_warm_start(True)

        # Check if solution was OK
        if not self._accepted_solution(k):
            # Check previous solution
            if self._accepted_solution(k-1):
                # Use old input
                return solution_time, self.u_k
            else:
                raise MPCFail("MPC could not recover!")

        # Loop through inputs and construct inputs for simulator
        for i, inp in enumerate(prob.getVariables(prob.REAL_INPUT)):
            names.append(inp.getName())
            inputs.append(res[names[i]][0])

        # u(t) just returns a constant (constant input through each sample)
        def u(t):
            return np.array(inputs)

        # Update last accepted solution/inputs and return
        self.u_k = (names, u)
        return solution_time, self.u_k

    def print_solver_stats(self):
        for e in self.stats:
            print "{0}: {1} in {2} iterations".format(e[0], e[1][0], e[1][1])

import sys
sys.path.append('../')

import numpy as np
import casadi as ca
from acados_template.acados_ocp import AcadosOcp
from acados_template.acados_ocp_solver import AcadosOcpSolver
from acados_template.acados_sim_solver import AcadosSimSolver
from models.robot_model import export_robot_ode_model
from utils.obstacle_generator import generate_random_obstacles
from utils.visualization import VisStaticRobotEnv
from models.world_specification import N_SOLV, TF, R_ROBOT, MARGIN, TOL

class RobotOcpProblem():
    def __init__(self, robot_init, robot_end, seed=None):
        self.robot_init = robot_init
        self.robot_end = robot_end
        self.subgoal = self.robot_end
        self.seed = seed
        
        if self.seed is not None:
            self.obstacles = generate_random_obstacles(self.seed)
        else:
            self.obstacles = generate_random_obstacles()
        
        
        self.model = export_robot_ode_model()
        self.nx = self.model.x.size()[0]
        self.nu = self.model.u.size()[0]
        self.R = 0.4 * np.eye(self.nu)
        self.E_pos = 10 * np.eye(2)   # penalty on end position
        self.E_dot = 5 * np.eye(2)   # penalty on final speed (angular + translation)
        
        # keeping track of robot position and trajectory
        self.simX = np.ndarray((0, self.nx))
        self.simU = np.ndarray((0, self.nu))
        self.x0 = robot_init
        self.simX = np.append(self.simX, self.x0.reshape((1, self.nx)), axis=0)
        self.reached_goal = False
        self.min_margin_traj = np.inf
        
        # initialize ocp and the solver
        self.init_ocp()
        self.init_ocp_solver()
    
    def init_ocp(self):
        self.ocp = AcadosOcp()
        self.ocp.model = self.model
        self.ocp.dims.N = N_SOLV
        
        ## setup ocp cost
        self.ocp.cost.cost_type = "LINEAR_LS"
        self.ocp.cost.cost_type_e = "LINEAR_LS"
        self.ocp.cost.Vx = np.zeros((self.nu, self.nx))
        self.ocp.cost.Vu = np.eye(self.nu)
        self.ocp.cost.yref = np.zeros(self.nu)
        self.ocp.cost.Vx_0 = self.ocp.cost.Vx
        self.ocp.cost.Vu_0 = self.ocp.cost.Vu
        self.ocp.cost.yref_0 = self.ocp.cost.yref
        self.ocp.cost.Vx_e = np.eye(self.nx)
        self.ocp.cost.yref_e = np.hstack((self.robot_end.reshape(2), np.zeros(self.nx-2)))
        self.ocp.cost.W = 2 * self.R
        self.ocp.cost.W_0 = self.ocp.cost.W
        # build the matrix for the terminal costs
        W_e_1 = np.hstack((self.E_pos, np.zeros((2, self.nx - 2))))
        W_e_2 = np.zeros((1, self.nx))
        W_e_3 = np.hstack((np.zeros((2, self.nx - 2)), self.E_dot))
        self.ocp.cost.W_e = 2* np.vstack((W_e_1, W_e_2, W_e_3))
        
        ## setup ocp constraints
        # fix initial position
        self.ocp.constraints.x0 = self.robot_init
        
        # no driving backwards
        # & staying within designated rectangle defined by edges (-2, -2) , (2, 2)
        self.ocp.constraints.lbx = np.array([-7, -7, 0, -6])
        self.ocp.constraints.ubx = np.array([7, 7, 6, 6])
        self.ocp.constraints.idxbx = np.array([0, 1, 3, 4])
        
        # avoid obstacles
        h = []
        h_lb = []
        h_ub = []
        for o in self.obstacles:
            h += [(self.model.x[0] - o.x)**2 + (self.model.x[1] - o.y)**2]
            h_lb += [(o.r + R_ROBOT + MARGIN)**2]
            h_ub += [1e15]
        
        if len(h) > 0:
            self.model.con_h_expr = ca.vertcat(*h)
            self.ocp.constraints.lh = np.array(h_lb)
            self.ocp.constraints.uh = np.array(h_ub)
    
    def init_ocp_solver(self):
        # configure solver
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'IRK'
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # 'SQP_RTI'
        self.ocp.solver_options.tf = TF
        
        # specify solver and integrator, using same specification
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')
        self.ocp_integrator = AcadosSimSolver(self.ocp, json_file='acados_ocp.json')
        
        # initialize control trajectory to all 0
        for i in range(N_SOLV):
            self.ocp_solver.set(i, 'u', np.array([0, 0]))
    
    def step(self, max_iter, visualize=False):
        """
        Run RTI solver until reaching the goal or but for at most max_iter steps.
        If vis is true show trajectory completed by the robot until now.
        Returns:
            x0: robot position after the iterations
            hit: whether robot hit an obstacle
            reached_goal: whether robot reached the tolerance area around the goal
        """
        reached_subgoal = False
        i = 0
        while i < max_iter:
            # set constraint on starting position (x0_bar)
            self.ocp_solver.set(0, 'ubx', self.x0)
            self.ocp_solver.set(0, 'lbx', self.x0)
            
            # solve the ocp for the new starting position and get control for next step
            print(f"\n\nRUNNING STAGE {i+1}")
            self.ocp_solver.solve()
            self.ocp_solver.print_statistics()
            
            
            # simulate model from current position and computed control
            u = self.ocp_solver.get(0, 'u')
            self.ocp_integrator.set('x', self.x0)
            self.ocp_integrator.set('u', u)
            self.ocp_integrator.solve()
            
            # get the next starting position after simulation
            self.x0 = self.ocp_integrator.get('x')
            
            # also get (x,y) coordinate of predicted terminal state xN from the solved OCP
            # will be later needed for training the RL agend
            xN = self.ocp_solver.get(N_SOLV, 'x')[0:2]
            
            # check wether robot did hit an obstacle:
            min_margin = np.inf
            for o in self.obstacles:
                margin = np.sqrt((self.x0[0] - o.x)**2 + (self.x0[1] - o.y)**2) - (o.r + R_ROBOT)
                if margin < min_margin:
                    min_margin = margin
            if min_margin < self.min_margin_traj:
                self.min_margin_traj = min_margin
            
            self.simX = np.append(self.simX, self.x0.reshape((1, self.nx)), axis=0)
            self.simU = np.append(self.simU, u.reshape((1, self.nu)), axis=0)
            
            # compute the norm of the state vector (excluding the orientation which we do not consider)
            # in case we are within the tolerance end the closed loop simulation
            if np.linalg.norm(np.take(self.x0, [0, 1, 3, 4]) - np.append(self.subgoal, [0, 0])) <= TOL:
                reached_subgoal = True
                break
            
            # shift initialization for initial guess
            for j in range(N_SOLV-1):
                self.ocp_solver.set(j, 'x', self.ocp_solver.get(j+1, 'x'))
                self.ocp_solver.set(j, 'u', self.ocp_solver.get(j+1, 'u'))
            self.ocp_solver.set(N_SOLV-1, 'x', self.ocp_solver.get(N_SOLV, 'x'))
            # keep end values for state trajectory (assuming they are similar) but initialize the controls to 0
            self.ocp_solver.set(N_SOLV-1, 'u', np.array([0, 0]))
                
            i += 1
            
            if i == max_iter // 2:
                # ocp_solver.cost_set(N-1, 'yref', np.array([0, 0]))
                print("set new y_ref")
            
            # # some statistics
            # print(ocp_solver.get_stats('time_tot'))
        
        print(f"Final difference to sub goal state: {self.simX[-1][0:2] - self.subgoal}")
        
        if visualize:
            vis = VisStaticRobotEnv((-7.5, 7.5), (-7.5, 7.5), (0, 0), R_ROBOT, self.obstacles)
            vis.set_trajectory(self.simX[:,:2].T)
            vis.run_animation()
        
        return self.simX[-1], (self.min_margin_traj <= 0), reached_subgoal
    
    def set_subgoal(self, x, y):
        """
        Sets a new subgoal that the RTI controller will try to reach.
        """
        self.subgoal = np.array([x, y])
        self.ocp_solver.cost_set(N_SOLV, 'yref', np.array([x, y, 0, 0, 0]))

    
if __name__ == "__main__":
    ocp_problem = RobotOcpProblem(np.array([-3, -6, np.pi / 4, 0, 0]), np.array([6, 6]), 0)
    for i in range(1):
        ocp_problem.step(20, True)
    ocp_problem.set_subgoal(3, 6)
    for i in range(1):
        ocp_problem.step(20, True)
        
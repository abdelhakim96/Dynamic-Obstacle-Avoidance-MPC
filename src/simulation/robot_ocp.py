from acados_template.acados_model import AcadosModel
from acados_template.acados_ocp import AcadosOcp
from acados_template.acados_ocp_solver import AcadosOcpSolver
from acados_template.acados_sim_solver import AcadosSimSolver

sys.path.append('../')
from typing import List
from models.robot_model import export_robot_ode_model
from utils.visualization import Obstacle, VisStaticRobotEnv
from utils.obstacle_generator import generate_random_obstacles, generate_regular_obstacle_grid
from models.world_specification import MARGIN, N_SOLV, R_ROBOT, TF, TOL
import numpy as np
import casadi as ca
from matplotlib import pyplot as plt

def solve_robot_ocp_closed_loop(robot_init, robot_end, obstacles: List[Obstacle], max_iter):
    ocp = AcadosOcp()
    robot_radius = 0.2
    
    model = export_robot_ode_model()
    ocp.model = model
    
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    
    ocp.dims.N = N_SOLV
    
    R = 0.4 * np.eye(nu)
    E_pos = 10 * np.eye(2)   # penalty on end position
    E_dot = 5 * np.eye(2)   # penalty on final speed (angular + translation)
    
    
    # ocp.cost.cost_type = 'EXTERNAL'
    # ocp.cost.cost_type_e = 'EXTERNAL'
    # ocp.model.cost_expr_ext_cost = model.u.T @ R @ model.u
    # ocp.model.cost_expr_ext_cost_e = (model.x[0:2] - robot_end).T @ E_pos @ (model.x[0:2] - robot_end) + model.x[3:].T @ E_dot @ model.x[3:]
    
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.Vx = np.zeros((nu, nx))
    ocp.cost.Vu = np.eye(nu)
    ocp.cost.yref = np.zeros(nu)
    ocp.cost.Vx_0 = ocp.cost.Vx
    ocp.cost.Vu_0 = ocp.cost.Vu
    ocp.cost.yref_0 = ocp.cost.yref
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref_e = np.hstack((robot_end.reshape(2), np.zeros(nx-2)))
    ocp.cost.W = 2 * R
    ocp.cost.W_0 = ocp.cost.W
    # build the matrix for the terminal costs
    W_e_1 = np.hstack((E_pos, np.zeros((2, nx - 2))))
    W_e_2 = np.zeros((1, nx))
    W_e_3 = np.hstack((np.zeros((2, nx - 2)), E_dot))
    ocp.cost.W_e = 2* np.vstack((W_e_1, W_e_2, W_e_3))
    
    # ocp.cost.cost_type = "LINEAR_LS"
    # ocp.cost.cost_type_e = "LINEAR_LS"
    # ocp.cost.Vx = np.zeros((nu, nx))
    # ocp.cost.Vu = np.eye(nu)
    # ocp.cost.yref = np.zeros(nu)
    # ocp.cost.Vx_0 = ocp.cost.Vx
    # ocp.cost.Vu_0 = ocp.cost.Vu
    # ocp.cost.yref_0 = ocp.cost.yref
    # ocp.cost.Vx_e = np.eye(nx)
    # ocp.cost.yref_e = np.hstack((robot_end.reshape(2), np.zeros(nx-2)))
    # ocp.cost.W = 2 * R
    # ocp.cost.W_0 = ocp.cost.W
    # # build the matrix for the terminal costs
    # W_e_1 = np.hstack((E_pos, np.zeros((2, nx - 2))))
    # W_e_2 = np.zeros((1, nx))
    # W_e_3 = np.hstack((np.zeros((2, nx - 2)), E_dot))
    # ocp.cost.W_e = 2* np.vstack((W_e_1, W_e_2, W_e_3))
    # print(ocp.cost.W_e)
    
    ## fix initial position
    ocp.constraints.x0 = robot_init
    
    # no driving backwards
    # & staying within designated rectangle defined by edges (-2, -2) , (2, 2)
    ocp.constraints.lbx = np.array([-7, -7, 0, -6])
    ocp.constraints.ubx = np.array([7, 7, 6, 6])
    ocp.constraints.idxbx = np.array([0, 1, 3, 4])
    
    
    # # avoid the obstacles
    h = []
    h_lb = []
    h_ub = []
    for o in obstacles:
        h += [(model.x[0] - o.x)**2 + (model.x[1] - o.y)**2]
        h_lb += [(o.r + R_ROBOT + MARGIN)**2]
        h_ub += [1e15]
    
    if len(h) > 0:
        model.con_h_expr = ca.vertcat(*h)
        ocp.constraints.lh = np.array(h_lb)
        ocp.constraints.uh = np.array(h_ub)
        
        # # allow for some slack, only penalize violations on lower bound
        # # mixture of L1 and L2 penalty
        # ocp.constraints.Jsh = np.eye(len(h))
        # ocp.constraints.Jsh_e = np.eye(len(h))
        # ocp.cost.Zl = np.zeros(len(h))
        # ocp.cost.Zl_e = np.zeros(len(h))
        # ocp.cost.Zu = np.zeros_like(ocp.cost.Zl)
        # ocp.cost.Zu_e = np.zeros_like(ocp.cost.Zl_e)
        # ocp.cost.zl = 100 * np.ones(len(h))
        # ocp.cost.zl_e = 100 * np.ones(len(h))
        # ocp.cost.zu = np.zeros(len(h))
        # ocp.cost.zu_e = np.zeros(len(h))
    
    # configure solver
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # 'SQP_RTI'
    ocp.solver_options.tf = TF
    # ocp.solver_options.nlp_solver_max_iter = 20
    
    # specify solver and integrator, using same specification
    ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    ocp_integrator = AcadosSimSolver(ocp, json_file='acados_ocp.json') 
    # ocp_solver.cost_set(N-1, 'yref', np.array([0, 0]))
    
    # initialize control trajectory to all 0
    for i in range(N_SOLV):
        ocp_solver.set(i, 'u', np.array([0, 0]))
    
    
    simX = np.ndarray((0, nx))
    simU = np.ndarray((0, nu))
    subgoals = np.ndarray((0, 2))
    
    x0 = robot_init
    simX = np.append(simX, x0.reshape((1, nx)), axis=0)
    
    # track wether robot hits obstacle along trajectory
    min_margin_traj = np.inf
    # keep book wether robot got close enough to goal
    reached_goal = False
    
    i = 0
    while i < max_iter:
        # set constraint on starting position (x0_bar)
        ocp_solver.set(0, 'ubx', x0)
        ocp_solver.set(0, 'lbx', x0)
        
        # solve the ocp for the new starting position and get control for next step
        print(f"\n\nRUNNING STAGE {i+1}")
        status = ocp_solver.solve()
        ocp_solver.print_statistics()
        
        u = ocp_solver.get(0, 'u')
        x_ref = ocp_solver.get(1, 'x')
        
        # simulate model from current position and computed control
        ocp_integrator.set('x', x0)
        ocp_integrator.set('u', u)
        stat_int = ocp_integrator.solve()
        
        # get the next starting position after simulation
        x0 = ocp_integrator.get('x')
        
        # also get (x,y) coordinate of predicted terminal state xN from the solved OCP
        # will be later needed for training the RL agend
        xN = ocp_solver.get(N_SOLV, 'x')[0:2]
        subgoals = np.append(subgoals, xN.reshape((1, 2)), axis=0)
        
        # info on wether robot did or did not hit an obstacle:
        min_margin = np.inf
        for o in obstacles:
            margin = np.sqrt((x0[0] - o.x)**2 + (x0[1] - o.y)**2) - (o.r + robot_radius)
            if margin < min_margin:
                min_margin = margin
        print(f"margin to closest obstacle: {min_margin}")
        if min_margin < min_margin_traj:
            min_margin_traj = min_margin
        
        simX = np.append(simX, x0.reshape((1, nx)), axis=0)
        simU = np.append(simU, u.reshape((1, nu)), axis=0)
        
        # compute the norm of the state vector (excluding the orientation which we do not consider)
        # in case we are within the tolerance end the closed loop simulation
        if np.linalg.norm(np.take(x0, [0, 1, 3, 4]) - np.append(robot_end, [0, 0])) <= TOL:
            print(f"Reached goal state after iteration {i+1}")
            reached_goal = True
            break
        
        # shift initialization for initial guess
        for j in range(N_SOLV-1):
            ocp_solver.set(j, 'x', ocp_solver.get(j+1, 'x'))
            ocp_solver.set(j, 'u', ocp_solver.get(j+1, 'u'))
        ocp_solver.set(N_SOLV-1, 'x', ocp_solver.get(N_SOLV, 'x'))
        # keep end values for state trajectory (assuming they are similar) but initialize the controls to 0
        ocp_solver.set(N_SOLV-1, 'u', np.array([0, 0]))
            
        i += 1
        
        if i == max_iter // 2:
            # ocp_solver.cost_set(N-1, 'yref', np.array([0, 0]))
            ocp_solver.cost_set(N_SOLV - 1, 'yref', np.array([4, 6, 0, 0, 0]))
            print("set new y_ref")
        
        # # some statistics
        # print(ocp_solver.get_stats('time_tot'))
    
    print(f"Final difference to goal state: {simX[-1][0:2] - robot_end}")
    print(f"Minimal margin to obstacle along trajectory: {min_margin_traj}")
    print(simX[-1], (min_margin_traj <= 0), reached_goal)
    # print(subgoals)
    # # plot = plt.
    print(f"reference value for final cost term: {ocp.cost.yref_e}")
    
    vis = VisStaticRobotEnv((-7.5, 7.5), (-7.5, 7.5), (0, 0), R_ROBOT, obstacles)
    vis.set_trajectory(simX[:,:2].T)
    vis.run_animation()
    
    return simX[-1], (min_margin_traj <= 0), reached_goal

  
if __name__ == '__main__':
    robot_radius = 0.2
    margin = 0.05
    
    # some obstacle grid
    N = 25
    y_min = -5
    y_max = 5
    x_min = -5
    x_max = 5
    r_min = 0.6
    r_max = 1.0
    for seed in range(10):
            
        obstacles = generate_random_obstacles(seed)
        # obstacles = generate_regular_obstacle_grid(3, 3, x_min, x_max, y_min, y_max, r_min, r_max)
        solve_robot_ocp_closed_loop(np.array([-3, -6, np.pi / 4, 0, 0]),
                                    np.array([6, 6]),
                                    obstacles,
                                    400)

from importlib import import_module
from ssl import HAS_ALPN
import sys
from xml.etree.ElementTree import XML

from acados_template.acados_model import AcadosModel
from acados_template.acados_ocp import AcadosOcp
from acados_template.acados_ocp_solver import AcadosOcpSolver
from acados_template.acados_sim_solver import AcadosSimSolver
sys.path.append('../')
from typing import List
from models.robot_model import export_robot_ode_model
from utils.visualization import Obstacle, VisStaticRobotEnv
from utils.obstacle_generator import generate_random_obstacles, generate_regular_obstacle_grid
import numpy as np
import casadi as ca
from matplotlib import pyplot as plt

def solve_robot_ocp_closed_loop(robot_init, robot_end, robot_radius, obstacles: List[Obstacle], margin):
    ocp = AcadosOcp()
    
    model = export_robot_ode_model()
    ocp.model = model
    
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    
    # set time frame for which we solve the trajectory at each step and the number of steps we use to discretize it
    Tf = 0.5
    N = 5
    # Tf = 4
    # N = 40
    
    # end the closed loop optimization if close enough to goal or after a certain number of iterations
    TOL = 0.1
    max_iter = 400
    
    ocp.dims.N = N
    
    R = 0.4 * np.eye(nu)
    E_pos = 10 * np.eye(2)   # penalty on end position
    E_dot = 5 * np.eye(2)   # penalty on final speed (angular + translation)
    
    
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost = model.u.T @ R @ model.u
    # ocp.model.cost_expr_ext_cost = 0
    ocp.model.cost_expr_ext_cost_e = (model.x[0:2] - robot_end).T @ E_pos @ (model.x[0:2] - robot_end) + model.x[3:].T @ E_dot @ model.x[3:]
    
    # # limit controls
    # ocp.constraints.lbu = np.array([-5, -5])
    # ocp.constraints.ubu = np.array([5, 5])
    # ocp.constraints.idxbu = np.array([0, 1])
    
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
        h_lb += [(o.r + robot_radius + margin)**2]
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
    ocp.solver_options.tf = Tf
    # ocp.solver_options.nlp_solver_max_iter = 20
    
    # specify solver and integrator, using same specification
    ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    ocp_integrator = AcadosSimSolver(ocp, json_file='acados_ocp.json') 
    
    # initialize control trajectory to all 0
    for i in range(N):
        ocp_solver.set(i, 'u', np.array([0, 0]))
    
    
    simX = np.ndarray((0, nx))
    simU = np.ndarray((0, nu))
    subgoals = np.ndarray((0, 2))
    
    x0 = robot_init
    simX = np.append(simX, x0.reshape((1, nx)), axis=0)
    
    # track wether robot hits obstacle along trajectory
    min_margin_traj = np.inf
    
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
        xN = ocp_solver.get(N, 'x')[0:2]
        subgoals = np.append(subgoals, xN.reshape((1, 2)), axis=0)
        
        print(f"difference predicted next state vs. simulated next state: {x_ref - x0}")
        print(type(x0))
        
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
            break
        
        # shift initialization for initial guess
        for j in range(N-1):
            ocp_solver.set(j, 'x', ocp_solver.get(j+1, 'x'))
            ocp_solver.set(j, 'u', ocp_solver.get(j+1, 'u'))
        ocp_solver.set(N-1, 'x', ocp_solver.get(N, 'x'))
        # keep end values for state trajectory (assuming they are similar) but initialize the controls to 0
        ocp_solver.set(N-1, 'u', np.array([0, 0]))
        # ocp_solver.set(N, 'x',np.array([0, 0, 0, 0, 0]))
            
        i += 1
        
        
        # # for debugging we can visualize the system after each step
        # vis = VisStaticRobotEnv((-3, 3), (-3, 3), (0, 0), robot_radius, obstacles)
        # vis.set_trajectory(simX[:i+1,:2].T)
        # vis.run_animation()
        
        # # some statistics
        # print(ocp_solver.get_stats('time_tot'))
    print(f"Final difference to goal state: {simX[-1][0:2] - robot_end}")
    print(f"Minimal margin to obstacle along trajectory: {min_margin_traj}")
    print(subgoals)
    # plot = plt.
    
    vis = VisStaticRobotEnv((-7.5, 7.5), (-7.5, 7.5), (0, 0), robot_radius, obstacles)
    vis.set_trajectory(simX[:,:2].T)
    vis.run_animation()


def solve_robot_ocp_open_loop(robot_init, robot_end, robot_radius, obstacles: List[Obstacle]):
    """
        This function needs to be fixed (remove the closed loop part, solver complete OCP for bigger Tf and N)
    """
    ocp = AcadosOcp()
    
    model = export_robot_ode_model()
    ocp.model = model
    
    Tf = 5
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    N = 40
    
    ocp.dims.N = N
    
    # Q = np.eye(nx)
    R = 0.1 * np.eye(nu)
    # R = np.zeros((nu, nu))
    E_pos = 10 * np.eye(2)   # penalty on end position
    E_dot = 5 * np.eye(2)   # penalty on final speed (angular + translation)
    
    
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.solver_options.line_search_use_sufficient_descent = 1
    # ocp.model.cost_expr_ext_cost = model.x.T @ Q @ model.x + model.u.T @ R @ model.u
    ocp.model.cost_expr_ext_cost = model.u.T @ R @ model.u
    ocp.model.cost_expr_ext_cost_e = (model.x[0:2] - robot_end).T @ E_pos @ (model.x[0:2] - robot_end) + model.x[3:].T @ E_dot @ model.x[3:]
    
    # limit controls
    ocp.constraints.lbu = np.array([-1, -1])
    ocp.constraints.ubu = np.array([1, 1])
    ocp.constraints.idxbu = np.array([0, 1])
    
    ## constraints
    ocp.constraints.x0 = robot_init
    
    # no driving backwards, stay within speed limits & staying within designated rectangle
    ocp.constraints.lbx = np.array([-2, -2, 0])
    ocp.constraints.ubx = np.array([2, 2, 1e15])
    ocp.constraints.idxbx = np.array([0, 1, 3])
    
    
    # # avoid the obstacles
    h = []
    h_lb = []
    h_ub = []
    for o in obstacles:
        h += [(model.x[0] - o.x)**2 + (model.x[1] - o.y)**2]
        h_lb += [(o.r + robot_radius)**2]
        h_ub += [1e15]
    
    if len(h) > 0:
        model.con_h_expr = ca.vertcat(*h)
        ocp.constraints.lh = np.array(h_lb)
        ocp.constraints.uh = np.array(h_ub)
    
    # configure solver
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.levenberg_marquardt = 1e-2
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    # ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # SQP
    # ocp.solver_options.nlp_solver_max_iter = 100
    
    ocp.solver_options.tf = Tf
    
    ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    ocp_integrator = AcadosSimSolver(ocp, json_file='acados_ocp.json') 
    ocp_integrator.set('T', Tf / N)
    for i in range(N):
        ocp_solver.set(i, 'u', np.array([0, 0]))
        ocp_solver.set(i+1, 'x', np.array([0, 0, 0, 0, 0]))
    
    simX = np.ndarray((N+1, nx))
    simU = np.ndarray((N, nu))
    
    x0 = robot_init
    simX[0, :] = robot_init
    
    
    ### CLOSED LOOP IMPLEMENTATION
    for i in range(0, 10*N):
        # set constraint on starting position (x0_bar)
        ocp_solver.set(0, 'ubx', x0)         
        ocp_solver.set(0, 'lbx', x0)
        
        # solve the ocp for the new starting position and get control for next step
        status = ocp_solver.solve()
        u = ocp_solver.get(0, 'u')
        print(f"Initial state: {ocp_solver.get(0, 'x')}")
        print(f"CONTROL: {u}")
        
        # simulate model from current position and computed control
        ocp_integrator.set('x', x0)
        ocp_integrator.set('u', u)
        stat_int = ocp_integrator.solve()
        print(f"Status integrator: {stat_int}")
        
        # get the next starting position after simulation
        x0 = ocp_integrator.get('x')
        print(f"next initial state: {x0}")
        
        # store state and control for the trajectory
        simX[i+1, :] = x0
        simU[i, :] = u
        
        # reset initial guess
        for i in range(N):
            ocp_solver.set(i, 'u', np.array([0, 0]))
            ocp_solver.set(i+1, 'x', np.array([0, 0, 0, 0, 0]))
        
        # # for debugging we can visualize the system after each step
        # vis = VisStaticRobotEnv((-3, 3), (-3, 3), (0, 0), robot_radius, obstacles)
        # vis.set_trajectory(simX[:i+1,:2].T)
        # vis.run_animation()
        
        # # some statistics
        # print(f"statistics after solving iteration {i}:")
        # ocp_solver.print_statistics()
        # print(ocp_solver.get_stats('time_tot'))
    ocp_solver.solve()
    print(ocp_solver.get_stats('time_tot'))
    
    for i in range(N):
        simX[i, :] = ocp_solver.get(i, 'x')
        simU[i, :] = ocp_solver.get(i, 'u')
    simX[N, :] = ocp_solver.get(N, 'x')
    vis = VisStaticRobotEnv((-3, 3), (-3, 3), (0, 0), robot_radius, obstacles)
    vis.set_trajectory(simX[:,:2].T)
    vis.run_animation()
    print(simX[N][0:2] - robot_end)
    
    
if __name__ == '__main__':
    robot_radius = 0.1
    margin = 0.05
    
    # sa some obstacle grid
    N = 20
    y_min = -6
    y_max = 6
    x_min = -6
    x_max = 6
    r_min = 0.6
    r_max = 1.0
    obstacles = generate_random_obstacles(N, x_min, x_max, y_min, y_max, r_min, r_max)
    # obstacles = generate_regular_obstacle_grid(3, 3, x_min, x_max, y_min, y_max, r_min, r_max)
    # # build a regular grid
    # for i in range(N):
    #     for j in range(N):
    #         x = x_min + (i+0.5) * (x_max - x_min) / (N)
    #         y = y_min + (j+0.5) * (y_max - y_min) / (N)
    #         obstacles += [Obstacle(x, y, r)]
    
    # # sample a grid
    # x_coordinates = np.random.uniform(x_min, x_max, (N, 1))
    # y_coordinates = np.random.uniform(y_min, y_max, (N, 1))
    # build a regular grid
    # x_coordinates = np.array([[-3, 0, 3, -3, 0, 3]]).T
    # y_coordinates = np.array([[-2, -2, -2, 2, 2, 2]]).T
    # radius = np.random.uniform(0.4, 0.8, (N, 1))
    # coordinates = np.hstack((x_coordinates, y_coordinates, radius))
    # for coord in coordinates:
    #     obstacles += [Obstacle(coord[0], coord[1], coord[2])]
    
    
    print(obstacles)
    solve_robot_ocp_closed_loop(np.array([-6, -6, np.pi / 4, 0, 0]),
                                np.array([6, 6]),
                                robot_radius,
                                obstacles,
                                margin)
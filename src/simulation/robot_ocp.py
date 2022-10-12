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
import numpy as np
import casadi as ca

def solve_robot_ocp(robot_init, robot_end, robot_radius, obstacles: List[Obstacle]):
    print('entered the function')
    ocp = AcadosOcp()
    
    model = export_robot_ode_model()
    ocp.model = model
    
    Tf = 0.5
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    N = 10
    
    ocp.dims.N = N
    
    # Q = np.eye(nx)
    R = 0.1 * np.eye(nu)
    # R = np.zeros((nu, nu))
    E_pos = 10 * np.eye(2)   # penalty on end position
    E_dot = 5 * np.eye(2)   # penalty on final speed (angular + translation)
    
    
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
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
    ocp.constraints.ubx = np.array([2, 2, 3])
    ocp.constraints.idxbx = np.array([0, 1, 3])
    
    
    # # avoid the obstacles
    h = []
    h_lb = []
    h_ub = []
    for o in obstacles:
        h += [(model.x[0] - o.x)**2 + (model.x[1] - o.y)**2]
        h_lb += [(o.r + robot_radius)**2]
        h_ub += [1e15]
    
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
    # ocp.solver_options.nlp_solver_max_iter = 1
    
    ocp.solver_options.tf = Tf
    
    ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    ocp_integrator = AcadosSimSolver(ocp, json_file='acados_ocp.json') 
    # ocp_integrator.set('T', Tf / N)
    for i in range(N):
        ocp_solver.set(i, 'u', np.array([0, 0]))
    
    simX = np.ndarray((10*N+1, nx))
    simU = np.ndarray((10*N, nu))
    
    x0 = robot_init
    simX[0, :] = robot_init
    
    for i in range(0, 10*N):
        ocp_solver.set(0, 'ubx', x0)
        ocp_solver.set(0, 'lbx', x0)
        status = ocp_solver.solve()
        u = ocp_solver.get(0, 'u')
        print(f"Initial state: {ocp_solver.get(0, 'x')}")
        print(f"CONTROL: {u}")
        ocp_integrator.set('x', x0)
        ocp_integrator.set('u', u)
        stat_int = ocp_integrator.solve()
        print(f"Status integrator: {stat_int}")
        x0 = ocp_integrator.get('x')
        print(f"next initial state: {x0}")
        simX[i+1, :] = x0
        simU[i, :] = u
        # vis = VisStaticRobotEnv((-3, 3), (-3, 3), (0, 0), robot_radius, obstacles)
        # vis.set_trajectory(simX[:i+1,:2].T)
        # vis.run_animation()
        
        # print(f"statistics after solving iteration {i}:")
        # ocp_solver.print_statistics()
        # print(ocp_solver.get_stats('time_tot'))
    # ocp_solver.solve()
    # print(ocp_solver.get_stats('time_tot'))
    
    # for i in range(N):
    #     simX[i, :] = ocp_solver.get(i, 'x')
    #     simU[i, :] = ocp_solver.get(i, 'u')
    # simX[N, :] = ocp_solver.get(N, 'x')
    vis = VisStaticRobotEnv((-3, 3), (-3, 3), (0, 0), robot_radius, obstacles)
    vis.set_trajectory(simX[:,:2].T)
    vis.run_animation()
    print(simX[N][0:2] - robot_end)
    
    
if __name__ == '__main__':
    # build some obstacle grid
    obstacles = []
    N = 1
    y_min = -2
    y_max = 2
    x_min = -2
    x_max = 2
    r = 0.4
    for i in range(N):
        for j in range(N):
            x = x_min + (i+0.5) * (x_max - x_min) / (N)
            y = y_min + (j+0.5) * (y_max - y_min) / (N)
            obstacles += [Obstacle(x, y, r)]
    
    solve_robot_ocp(np.array([-2, -2, np.pi / 4, 0, 0]),
                    np.array([2, 2]),
                    0.1,
                    obstacles)
    # solve_robot_ocp(np.array([-2, -2, np.pi / 4, 0, 0]),
    #                 np.array([2, 2]),
    #                 0.1,
    #                 [Obstacle(0, 0, 0.5),
    #                  Obstacle(1, 1.5, 0.5)])
    # solve_robot_ocp(np.ndarray([0, 0, np.pi / 4, 0, 0]), np.ndarray([0, 1]), [])
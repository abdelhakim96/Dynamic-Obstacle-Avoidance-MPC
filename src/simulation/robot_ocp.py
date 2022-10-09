from importlib import import_module
import sys

from acados_template.acados_model import AcadosModel
from acados_template.acados_ocp import AcadosOcp
from acados_template.acados_ocp_solver import AcadosOcpSolver
sys.path.append('../')
from typing import List
from models.robot_model import export_robot_ode_model
from utils.visualization import Obstacle, VisStaticRobotEnv
import numpy as np

def solve_robot_ocp(robot_init, robot_end, obstacles: List[Obstacle]):
    print('entered the function')
    ocp = AcadosOcp()
    
    model = export_robot_ode_model()
    ocp.model = model
    
    Tf = 5.0
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    N = 40
    
    ocp.dims.N = N
    
    # Q = np.eye(nx)
    R = np.eye(nu)
    E_pos = 5 * np.eye(2)   # penalty on end position
    E_dot = 5 * np.eye(2)   # penalty on final speed (angular + translation)
    
    
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    # ocp.model.cost_expr_ext_cost = model.x.T @ Q @ model.x + model.u.T @ R @ model.u
    ocp.model.cost_expr_ext_cost = model.u.T @ R @ model.u
    ocp.model.cost_expr_ext_cost_e = (model.x[0:2] - robot_end).T @ E_pos @ (model.x[0:2] - robot_end) + model.x[3:].T @ E_dot @ model.x[3:]
    
    # constraints
    ocp.constraints.x0 = robot_init
    
    # configure solver
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    
    ocp.solver_options.tf = Tf
    
    ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    
    simX = np.ndarray((N+1, nx))
    simU = np.ndarray((N, nu))
    
    status = ocp_solver.solve()
    
    for i in range(N):
        simX[i, :] = ocp_solver.get(i, 'x')
        simU[i, :] = ocp_solver.get(i, 'u')
    simX[N, :] = ocp_solver.get(N, 'x')
    vis = VisStaticRobotEnv((-2, 2), (-2, 2), (0, 0), 0.2, [])
    vis.set_trajectory(simX[:,:2].T)
    vis.run_animation()
    
    print(simX[N][0:2] - robot_end)
    
    
if __name__ == '__main__':
    solve_robot_ocp(np.array([0, 0, np.pi / 4, 0, 0]), np.array([-2, -2]), [])
    # solve_robot_ocp(np.ndarray([0, 0, np.pi / 4, 0, 0]), np.ndarray([0, 1]), [])
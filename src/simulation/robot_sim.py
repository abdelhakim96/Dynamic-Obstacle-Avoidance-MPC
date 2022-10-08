from acados_template.acados_sim import AcadosSim
from acados_template.acados_sim_solver import AcadosSimSolver

import sys
sys.path.append('../')

from models.robot_model import export_robot_ode_model
from utils.visualization import VisStaticRobotEnv
import numpy as np

def simulate_robot(robot_init, u_traj):
    sim = AcadosSim()
    
    model = export_robot_ode_model()
    
    sim.model = model
    
    Tf = 0.1
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    N = u_traj.shape[0]
    
    sim.solver_options.T = Tf
    
    sim.solver_options.integrator_type = 'IRK'
    sim.solver_options.num_stages = 3
    sim.solver_options.num_steps = 3
    sim.solver_options.newton_iter = 3
    sim.solver_options.collocation_type = 'GAUSS_RADAU_IIA'
    
    acados_integrator = AcadosSimSolver(sim)
    
    simX = np.zeros((N+1, nx))
    x0 = robot_init
    # u0 = np.array([1, 1])
    # acados_integrator.set("u", u_traj[0])
    
    simX[0, :] = x0
    
    for i in range(N):
        acados_integrator.set("x", simX[i, :])
        acados_integrator.set("u", u_traj[i])
        
        if sim.solver_options.integrator_type == 'IRK':
            acados_integrator.set('xdot', np.zeros(nx,))
        
        status = acados_integrator.solve()
        
        simX[i+1,:] = acados_integrator.get("x")
        
    if status != 0:
        raise Exception("acados returned status {}.".format(status))
    
    data = simX[:, 0:2]
    print(data)
    data = np.transpose(data)
    vis = VisStaticRobotEnv((-5, 5), (-5, 5), (0, 0), .2, [])
    vis.set_trajectory(data)
    vis.run_animation()
    
if __name__ == "__main__":
    u_traj = np.zeros((200, 2))
    for i in range(10):
        u_traj[i] = [1, 0.5]
    simulate_robot(np.array([0, 0, np.pi / 4, 0, 0]), u_traj)

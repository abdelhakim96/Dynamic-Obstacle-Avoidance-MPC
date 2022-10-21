import imp
import sys
sys.path.append("../")
from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos
from models.world_specification import R_ROBOT, MARGIN

def export_robot_ode_model(obstacles=None) -> AcadosModel:
    
    model_name = 'robot_model'
    
    ## defining casadi simbols
    # state s
    x = SX.sym('x')
    y = SX.sym('y')
    psi = SX.sym('psi')
    v = SX.sym('v')
    omega = SX.sym('omega')
    
    s = vertcat(x, y, psi, v, omega)
    
    # controls
    u_a = SX.sym('u_a')
    u_alpha = SX.sym('u_alpha')
    u = vertcat(u_a, u_alpha)
    
    # sdot
    xdot = SX.sym('xdot')
    ydot = SX.sym('ydot')
    psidot = SX.sym('psidot')
    vdot = SX.sym('vdot')
    omegadot = SX.sym('omegadot')
    sdot = vertcat(xdot, ydot, psidot, vdot, omegadot)
    
    # parameters
    p = []
    
    # dynamics
    f_expl = vertcat(v * cos(psi),
                     v * sin(psi),
                     omega,
                     u_a,
                     u_alpha)
    
    f_impl = sdot - f_expl
    
    model = AcadosModel()
    
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    
    model.x = s
    model.xdot = sdot
    model.u = u
    model.p = p
    model.name = model_name
    
    # avoid obstacles
    if obstacles is not None and len(obstacles) > 0:
        h = []
        for o in obstacles:
            h += [(model.x[0] - o.x)**2 + (model.x[1] - o.y)**2 - (o.r + R_ROBOT + MARGIN)**2]
            # h_lb += []
            # h_ub += [1e15]
        
        model.con_h_expr = vertcat(*h)
    
    return model
    
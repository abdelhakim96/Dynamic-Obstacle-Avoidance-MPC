import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import sys
sys.path.append('../')

from models.world_specification import R_OBST, R_ROBOT, N_SOLV, TF, X_MIN, X_MAX, Y_MIN, Y_MAX
# from robot_sim import simulate_robot

class Obstacle():
    def __init__(self, x_pos, y_pos, vx, vy) -> None:
        self.x = x_pos
        self.y = y_pos
        self.vx = vx
        self.vy = vy
        self.r = R_OBST
        
        self.traj = np.array([[x_pos, y_pos]])
        print(self.traj.shape)
    
    def step(self):
        """ Moves obstacle according to its velocity and the time discretization defined globally for the problem."""
        self.x, self.vx, self.y, self.vy = self.predict_step(self.x, self.vx, self.y, self.vy)
        self.traj = np.append(self.traj, np.array([[self.x, self.y]]), axis=0)
        
    def predict_step(self, x, vx, y, vy):
        dt = TF / N_SOLV

        if vx < 0:
            t_hit_x = (x - X_MIN) / abs(vx)
        elif vx > 0:
            t_hit_x = (X_MAX - x) / abs(vx)
            
        if t_hit_x <= dt:
            x += (vx * t_hit_x - vx * (dt - t_hit_x))
            vx = - vx
        else:
            x += vx * dt

        if vy < 0:
            t_hit_y = (y - Y_MIN) / abs(vy)
        elif vy > 0:
            t_hit_y = (Y_MAX - y) / abs(vy)
        if t_hit_y <= dt:
            y += (vy * t_hit_y - vy * (dt - t_hit_y))
            vy = - vy
        else:
            y += vy * dt
        return x, vx, y, vy
    
    def predict_trajectory(self, n):
        """
            Returns x and y positions over the next n steps, starting at current obstacle position.
            Returns:
                traj: of shape (n, 2)
        """
        x = self.x
        vx = self.vy
        y = self.y
        vy = self.vy
        
        traj = np.zeros((n+1,2))
        traj[0] = [x, y]
        
        for i in range(n):
            x, vx, y, vy = self.predict_step(x, vx, y, vy)
            traj[i+1] = [x, y]
        return traj
            
    def get_vis(self):
        return plt.Circle((self.x, self.y), self.r, fc='r')
    
    def get_trajectory(self):
        return self.traj

class VisStaticRobotEnv():
    def __init__(self, xlim, ylim, robot_pos, robot_size, obstacles):
        self._xlim = xlim
        self._ylim = ylim
        self._robot_pos_init = robot_pos
        self._robot_size = robot_size
        self._obstacles = []
        self._t_range = 0
        for o in obstacles:
            self._obstacles += [o.get_vis()]
        self._fig = plt.figure()
        self._ax = plt.axes(xlim=self._xlim, ylim=self._ylim)
        self._ax.set_aspect('equal')
        for o in self._obstacles:
            self._ax.add_patch(o)
        self._robot_vis = plt.Circle(self._robot_pos_init, self._robot_size, fc='y')
        self._traj_vis = plt.Line2D([], [])
        self._ax.add_line(self._traj_vis)
        
    def _init_vis(self):
        self._ax.add_patch(self._robot_vis)
        return self._robot_vis,
        
    def _animate(self, t):
        self._robot_vis.center = self._trajectory[:,t]
        return self._robot_vis,
    
    def run_animation(self):
        self._anim = animation.FuncAnimation(self._fig, self._animate, 
                                             init_func=self._init_vis,
                                             frames=self._t_range, interval=20)
        plt.show()
    
    def show_env(self):
        plt.show()
    
    def set_trajectory(self, traj):
        """
            Args:
                traj: array of shape (2, N+1)
        """
        self._trajectory = traj
        self._t_range = self._trajectory.shape[1]
        self._traj_vis.set_data(self._trajectory[0,:], self._trajectory[1,:])

class VisDynamicRobotEnv():
    def __init__(self, obstacles):
        self._xlim = (X_MIN, X_MAX)
        self._ylim = (Y_MIN, Y_MAX)
        self._obstacles = []
        self._t_range = 0
        for o in obstacles:
            self._obstacles += [o.get_vis()]
        self._fig = plt.figure()
        self._ax = plt.axes(xlim=self._xlim, ylim=self._ylim)
        self._ax.set_aspect('equal')
        for o in self._obstacles:
            self._ax.add_patch(o)
        self._robot_vis = plt.Circle((0, 0), R_ROBOT, fc='y')
        self._traj_vis = plt.Line2D([], [])
        self._ax.add_line(self._traj_vis)
        
    def _init_vis(self):
        # self._ax.add_patch(self._robot_vis)
        # return self._robot_vis,
        return self._obstacles
        
    def _animate(self, t):
        # self._robot_vis.center = self._trajectory[:,t]
        for o, traj in zip(self._obstacles, self._obst_trajectories):
            o.center = traj[:,t]
        # return self._robot_vis,
        return self._obstacles
    
    def run_animation(self):
        # self._anim = animation.FuncAnimation(self._fig, self._animate, 
        #                                      init_func=self._init_vis,
        #                                      frames=self._t_range, interval=50)
        self._anim = animation.FuncAnimation(self._fig, self._animate, 
                                             init_func=self._init_vis,
                                             frames=800, interval=50)
        plt.show()
    
    def show_env(self):
        plt.show()
    
    def set_trajectory(self, traj):
        """
            Args:
                traj: array of shape (2, N+1)
        """
        self._trajectory = traj
        self._t_range = self._trajectory.shape[1]
        self._traj_vis.set_data(self._trajectory[0,:], self._trajectory[1,:])
    
    def set_obst_trajectory(self, trajectories):
        self._obst_trajectories = trajectories
    
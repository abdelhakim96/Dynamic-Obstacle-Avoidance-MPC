import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import sys
sys.path.append('../')

from models.world_specification import R_OBST, R_ROBOT, N_SOLV, TF, V_MAX_OBST, X_MIN, X_MAX, Y_MIN, Y_MAX, RANDOMNESS, TOL, N_OBST
# from robot_sim import simulate_robot

class Obstacle():
    def __init__(self, x_pos, y_pos, vx, vy, random_move=False) -> None:
        self.x = x_pos
        self.y = y_pos
        self.vx = vx
        self.vy = vy
        self.r = R_OBST
        self.random_move = random_move
        self.traj = np.array([[x_pos, y_pos]])
    
    def step(self):
        """ Moves obstacle according to its velocity and the time discretization defined globally for the problem."""
        self.x, self.vx, self.y, self.vy = self.predict_step(self.x, self.vx, self.y, self.vy, noise=True)
        self.traj = np.append(self.traj, np.array([[self.x, self.y]]), axis=0)
        
    def predict_step(self, x, vx, y, vy, noise=False):
        dt = TF / N_SOLV
        
        if self.random_move and noise:
            # add some gaussian noise to the motion, relative to the current velocity.
            # at same time make sure velocity or obstacles stays within specification.
            noise_vx, noise_vy = np.random.normal(size=2)
            vx = min(max((1 + RANDOMNESS * noise_vx) * vx, - V_MAX_OBST), V_MAX_OBST)
            vy = min(max((1 + RANDOMNESS * noise_vy) * vy, - V_MAX_OBST), V_MAX_OBST)

        if vx < 0:
            t_hit_x = (x - X_MIN) / abs(vx)
        elif vx > 0:
            t_hit_x = (X_MAX - x) / abs(vx)
        else:
            t_hit_x = np.inf
            
        if t_hit_x <= dt:
            x += (vx * t_hit_x - vx * (dt - t_hit_x))
            vx = - vx
        else:
            x += vx * dt

        if vy < 0:
            t_hit_y = (y - Y_MIN) / abs(vy)
        elif vy > 0:
            t_hit_y = (Y_MAX - y) / abs(vy)
        else:
            t_hit_y = np.inf
        
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
                traj: of shape (n+1, 2)
        """
        x = self.x
        vx = self.vy
        y = self.y
        vy = self.vy
        
        traj = np.zeros((n+1,2))
        traj[0] = [x, y]
        
        for i in range(n):
            x, vx, y, vy = self.predict_step(x, vx, y, vy, noise=False)
            traj[i+1] = [x, y]
        return traj
            
    def get_vis(self):
        return plt.Circle((self.x, self.y), self.r, fc='r')
    
    def get_trajectory(self):
        return self.traj

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
        self._robot_vis = plt.Circle(self._robot_pos_init, self._robot_size, fc='y')
        self._traj_vis = plt.Line2D([], [])
        self._traj_vis_pred = plt.Line2D([], [], c='y')
        self._ax.add_line(self._traj_vis)
        self._ax.add_line(self._traj_vis_pred)
        
    def _init_vis(self):
        self._ax.add_patch(self._robot_vis)
        return [self._robot_vis] + [self._obstacles] + [self._traj_vis_pred]
        
    def _animate(self, t):
        self._robot_vis.center = self._trajectory[:,t]
        self._traj_vis_pred.set_data(self._pred_traj[t,:,0], self._pred_traj[t,:,1])
        for o, traj in zip(self._obstacles, self._obst_trajectories):
            o.center = traj[:,t]
        # return self._robot_vis,
        return [self._robot_vis] + [self._obstacles] + [self._traj_vis_pred]
    
    def get_robot_figure(self):
        self._robot_vis.center = self._trajectory[:, -1]
        self._ax.add_patch(self._robot_vis)
        return self._fig
    
    def image_to_array(self, fig):
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image

    def image_save(self, i):
        self._fig.savefig("figure"+str(i)+".png")

    def run_animation(self):
        # self._anim = animation.FuncAnimation(self._fig, self._gianimate, 
        #                                      init_func=self._init_vis,
        #                                      frames=self._t_range, interval=50)
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
    
    def set_pred_trajectories(self, pred_traj):
        self._pred_traj = pred_traj
    
    def set_obst_trajectory(self, trajectories):
        self._obst_trajectories = trajectories
    
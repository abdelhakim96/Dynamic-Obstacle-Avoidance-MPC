import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from models.world_specification import *
# from robot_sim import simulate_robot

class Obstacle():
    def __init__(self, x_pos, y_pos, r) -> None:
        self.x = x_pos
        self.y = y_pos
        self.r = r
    
    def get_vis(self):
        return plt.Circle((self.x, self.y), self.r, fc='r')

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
        self._goal = plt.Circle((X_MAX_ROBOT, Y_MAX_ROBOT), 0.5, fc='g')
        self._ax.add_patch(self._goal)
        self._traj_vis = plt.Line2D([], [])
        self._ax.add_line(self._traj_vis)
        
    def _init_vis(self):
        self._ax.add_patch(self._robot_vis)
        return self._robot_vis,
        
    def _animate(self, t):
        self._robot_vis.center = self._trajectory[:,t]
        return self._robot_vis,
    
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
        self._anim = animation.FuncAnimation(self._fig, self._animate, 
                                             init_func=self._init_vis,
                                             frames=self._t_range, interval=20)
        # f = r"robot_sim.gif" 
        # writergif = animation.PillowWriter(fps=30) 
        # self._anim.save(f, writer=writergif)

    def save_animation(self, i):
        self._anim = animation.FuncAnimation(self._fig, self._animate, 
                                             init_func=self._init_vis,
                                             frames=self._t_range, interval=20)
        f = r"gifs/robot_sim"+str(i)+".gif"
        writergif = animation.PillowWriter(fps=5)
        self._fig.savefig("figure.png")
        self._anim.save(f, writer=writergif)
    
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
    

if __name__ == "__main__":
    # some dummy trajectory    
    vis = VisStaticRobotEnv((-5, 5), (-5, 5), (0, 0), 0.2,
                            [Obstacle(3, 3, 0.5), Obstacle(5, 5, 0.5)])

    x = np.linspace(-5, 5, 360)
    y = np.sin(x)
    trajectory = np.vstack(x, y)
    
    print(trajectory.shape)
    vis.set_trajectory(trajectory)
    vis.run_animation()

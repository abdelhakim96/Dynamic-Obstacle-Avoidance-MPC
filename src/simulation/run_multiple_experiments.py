import os
import sys

def run_multiple_experiments():
    TFL = [0.5, 0.5, 1, 1.5, 2, 2.5, 3]
    N_OBSTL = [5, 5, 10, 15, 20, 25, 30]
    for i in range(len(TFL)-1):
        with open('models/world_specification.py', 'r') as file:
            filedata = file.read()
            filedata = filedata.replace(f'TF = {TFL[i]}', f'TF = {TFL[i+1]}')
        with open('models/world_specification.py', 'w') as file:
                file.write(filedata)
        for j in range(len(N_OBSTL)-1):
        # Read and Overwrite the file with the new values
            with open('models/world_specification.py', 'r') as file:
                filedata = file.read()
                filedata = filedata.replace(f'N_OBST = {N_OBSTL[j]}', f'N_OBST = {N_OBSTL[j+1]}')
            with open('models/world_specification.py', 'w') as file:
                file.write(filedata)
            os.system('python3 simulation/experiments.py')

        # Restore the N_OBST to the original values
        with open('models/world_specification.py', 'r') as file:
                filedata = file.read()
                filedata = filedata.replace(f'N_OBST = {N_OBSTL[-1]}', f'N_OBST = 5')
        with open('models/world_specification.py', 'w') as file:
                file.write(filedata)
if __name__ == '__main__':
    run_multiple_experiments()
import sys
sys.path.append('.')
from subprocess import Popen

dir_prefix = "RewardEval"
preset = 'CartPole_NStepQ'  # 'Mujoco_DDPG'
levels = [None]  # ["hopper", "ant", "walker2d"]
num_seeds = 3
num_workers = 8

processes = []
for level in levels:
    for seed in range(num_seeds):
        command = ['python3', 'coach.py', '-ns', '-p', '{}'.format(preset),
                    '--seed', '{}'.format(seed), '-n', '{}'.format(num_workers), '-ew']
        if level is not None:
            command.extend(['-lvl', '{}'.format(level)])
            command.extend(['-e', '{}{}/{}'.format(dir_prefix, preset, level)])
        else:
            command.extend(['-e', '{}_{}'.format(dir_prefix, preset)])
        print(command)

        p = Popen(command)
        processes.append(p)

for p in processes:
    p.wait()

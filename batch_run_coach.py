import sys
sys.path.append('.')
import json
import os
from subprocess import Popen

#TODO - this file is irrelevant without the hyperparameter_sweep.py file (from coach-kubernetes)
#       probably should move this file to this repo

# os.environ['coach_json_params'] = '{"env_params.level-BreakoutDeterministic-v4-preset-Breakout_Dueling_DDQN": {"env_params.level": "BreakoutDeterministic-v4", "preset": "Breakout_Dueling_DDQN"}, "env_params.level-PongDeterministic-v4-preset-Breakout_Dueling_DDQN": {"env_params.level": "PongDeterministic-v4", "preset": "Breakout_Dueling_DDQN"}}'
params = json.loads(os.environ['coach_json_params'])
processes = []
for run_name, run_params in params.items():
    custom_params = []
    for k, v in run_params.items():
        if k == 'preset':
            continue
        #if isinstance(v, str):
        #    custom_params.append("{} = \"{}\"".format(k, v))
        #else:
        custom_params.append("{} = {}".format(k, v))
    formatted_custom_params = '; '.join(custom_params)
    command = ['python3', 'coach.py', '-p', '{}'.format(run_params['preset']), '-cp', ' {}'.format(formatted_custom_params), '-e', '{}'.format(run_name)]
    print(command)
    p = Popen(command)
    processes.append(p)

for p in processes:
    p.wait()

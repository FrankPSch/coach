import inspect
import re

import presets
from utils import dynamic_import

preset_names = [m[0] for m in inspect.getmembers(presets, inspect.isclass) if m[1].__module__ == 'presets']
for preset in preset_names:
    f = open('/'.join(['presets', preset]) + '.py', 'w')
    f.write('from configurations import *\n')
    f.write('from block_factories.basic_rl_factory import BasicRLFactory\n\n\n')
    c = dynamic_import('presets', preset)
    preset_source_code = inspect.getsource(c)
    a = preset_source_code.split('\n')
    e = []
    v = []
    for i, line in enumerate(a):
        class_definition = re.search(r"class (.*)\(Preset\):", line)
        if class_definition is not None:
            a[i] = line.replace('Preset', 'AgentParameters')
            a[i] = a[i].replace(class_definition.group(1), 'AgentParams')
        old_init_line = re.search(r".*(Preset).*\(.*,.*,(.*),.*\)",  line)
        if old_init_line:
            new_init_line = re.sub(r".*Preset.*\(.*,(.*),(.*),(.*)\)", r"        AgentParameters.__init__(self,\1,\3, None)", line)
            env_class = old_init_line.group(2).strip()
            a[i] = new_init_line
        m = re.search(r".*self.agent.*=.*", line)
        if m:
            a[i] = line.replace('self.agent', 'self.algorithm')

        m = re.search(r".*self.env.*=.*", line)
        if m:
            e.append(line.replace('.env', ''))
            a[i] = '        pass'

        m = re.search(r".*self.visualization.*=.*", line)
        if m:
            v.append(line.replace('.visualization', ''))
            a[i] = '        pass'

    # write the agent class
    agent_parameters_source_code = '\n'.join(a)
    f.write(agent_parameters_source_code)

    # write the env class
    if 'env_class' in locals():
        e.insert(0, '\n\nclass {}({}):\n    def __init__(self):\n        super().__init__()'.format('EnvParams', env_class))
        env_source_code = '\n'.join(e)
        f.write(env_source_code)

    # write the env class

    v.insert(0,
             '\n\n\nclass {}(VisualizationParameters):\n    def __init__(self):\n        super().__init__()'.format('VisParams'))
    vis_source_code = '\n'.join(v)
    f.write(vis_source_code)



    f.write('\n\nfactory = BasicRLFactory(agent_params=AgentParams, env_params=EnvParams, vis_params=VisParams)\n'.format(preset))
    """
    Hopper_DPPO_SimpleRL = RLBlockFactory(actor=Hopper_DPPO.agent, environment=Hopper_DPPO.environment,
                                          visualization=MyVisualizationParameters)
    """

    f.close()



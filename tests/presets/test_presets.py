# nasty hack to deal with issue #46
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import os
import time
import shutil
from subprocess import Popen, DEVNULL
from logger import screen


@pytest.mark.integration_test
def test_all_presets_are_running():
    # os.chdir("../../")
    test_failed = False
    all_presets = sorted([f.split('.')[0] for f in os.listdir('presets') if f.endswith('.py') and f != '__init__.py'])
    for preset in all_presets:
        print("Testing preset {}".format(preset))
        p = Popen(["python", "coach.py", "-p", preset, "-ns", "-e", ".test"], stdout=DEVNULL)

        # wait 30 seconds for distributed training + 10 seconds overhead of initialization etc.
        time.sleep(40)
        return_value = p.poll()

        if return_value is None:
            screen.success("{} passed successfully".format(preset))
        else:
            test_failed = True
            screen.error("{} failed".format(preset), crash=False)

        p.kill()
        if os.path.exists("experiments/.test"):
            shutil.rmtree("experiments/.test")

    assert not test_failed


# test_all_presets_are_running()

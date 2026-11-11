import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

source = Path('/content/drive/MyDrive/COMLLM/s7_fixed_code_20261108/collapse_pair_v5.py')
local = Path('/content/s7_collapse_pair_v5.py')
expected = '66615b07692a83221ab33ed25192af38fe1595e3bd3781c7d3a0f90abe8797ad'
payload = source.read_bytes()
assert hashlib.sha256(payload).hexdigest() == expected
assert not Path('/content/drive/MyDrive/COMLLM/s7_collapse_pair_v5_20261108').exists()
assert shutil.which('timeout')
local.write_bytes(payload)
assert hashlib.sha256(local.read_bytes()).hexdigest() == expected
cmd_env = dict(os.environ, PYTHONPATH='/content/comllm_s7_fixed',
               PYTHONDONTWRITEBYTECODE='1')
log_path = Path('/content/s7_collapse_pair_v5_20261108.log')
with log_path.open('x') as log:
    command = ['timeout', '--kill-after=10s', '480s', sys.executable, '-u', str(local)]
    process = subprocess.Popen(command, cwd='/content/comllm_s7_fixed', env=cmd_env,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
receipt = {'pid': process.pid, 'command': command, 'source_sha256': expected,
           'log': str(log_path), 'max_updates': 20, 'prior_failed_updates': 4, 'timeout_seconds': 480}
with Path('/content/s7_collapse_pair_v5_launch.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
print('COLLAPSE_PROBE_LAUNCHED', json.dumps(receipt), flush=True)

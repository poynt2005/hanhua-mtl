"""Small remote controls for the rented machine, over SSH.

Only what the pipeline needs between stages: stop the OCR service so the
translation model can have the GPU, and start it again for the next archive.

The service runs under a tiny restart loop at /root/ocr_loop.sh. It has its own
name on purpose: `pkill -f ocr_server.py` also matches the loop's command line
(and the ssh command that sends it), so the kill takes down the restarter too and
the service never comes back.
"""
import os
import subprocess

LOOP = "/root/ocr_loop.sh"
LOOP_BODY = ("#!/bin/bash\n"
             "while true; do PORT=8080 python /root/ocr_server.py; sleep 2; done\n")


def _ssh(host, port, key, remote_command):
    known_hosts = "NUL" if os.name == "nt" else "/dev/null"
    cmd = ["ssh", "-i", key, "-p", str(port),
           "-o", "StrictHostKeyChecking=no",
           "-o", "UserKnownHostsFile=" + known_hosts,
           "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           "root@" + host, remote_command]
    return subprocess.run(cmd, capture_output=True, text=True)


def write_loop(host, port, key):
    remote = ("echo '#!/bin/bash' > %s; "
              "echo 'while true; do PORT=8080 python /root/ocr_server.py; sleep 2; done' >> %s; "
              "chmod +x %s; echo ok") % (LOOP, LOOP, LOOP)
    return _ssh(host, port, key, remote)


def stop_ocr(host, port, key):
    """Stop the loop and the worker; the GPU is then free for the model."""
    remote = ("pkill -f 'ocr_[l]oop.sh'; sleep 1; "
              "pkill -x -f 'python /root/ocr_server.py'; sleep 3; "
              "nvidia-smi --query-gpu=memory.used --format=csv,noheader")
    res = _ssh(host, port, key, remote)
    return res.returncode == 0, (res.stdout or res.stderr or "").strip()[:40]


def start_ocr(host, port, key):
    res = _ssh(host, port, key,
               "nohup bash %s >/dev/null 2>&1 & sleep 2; pgrep -f 'ocr_[l]oop.sh' | head -1" % LOOP)
    return res.returncode == 0, (res.stdout or "").strip()[:20]

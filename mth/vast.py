"""vast.ai lifecycle helper: search / rent+deploy / status / destroy.

The agent only needs these calls; the heavy lifting stays inside the tool.
API key is read from VAST_API_KEY or the --api-key flag, never written to disk.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE = "https://console.vast.ai"
DEFAULT_IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"


def _cli():
    exe = os.path.join(os.path.dirname(sys.executable), "vastai.exe")
    if os.path.exists(exe):
        return exe
    return "vastai"


def _cli_run(key, args):
    cmd = [_cli(), "--api-key", key] + args
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout, proc.stderr

def _rest(key, path, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method="POST" if data else "GET")
    req.add_header("Authorization", "Bearer " + key)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def account(key):
    for path in ("/api/v0/users/current/", "/api/v0/users/current"):
        try:
            return _rest(key, path)
        except urllib.error.HTTPError:
            continue
    return {}


def search_offers(key, min_vram=24, min_down=200, min_up=50, max_price=0.30,
                  gpu=None, order="reliable", limit=12, min_reliability=0.99):
    if order == "value":
        sort = [["dlperf_per_dphtotal", "desc"]]
    elif order == "cheap":
        sort = [["dph_total", "asc"]]
    else:
        # "reliable": reliability is already floored by the filter, so pick the
        # cheapest machine inside that band (do NOT sort by reliability first --
        # advertised reliability does not predict real success, so paying a
        # premium for rel=1.000 is wasted).
        sort = [["dph_total", "asc"]]
    query = {
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "num_gpus": {"eq": 1},
        "gpu_ram": {"gte": min_vram * 1024},
        "direct_port_count": {"gte": 2},
        "cuda_max_good": {"gte": 12.1},
        "reliability2": {"gte": min_reliability},
        "order": sort,
        "type": "on-demand",
        "limit": limit,
    }
    if gpu:
        query["gpu_name"] = {"eq": gpu}
    if min_down:
        query["inet_down"] = {"gte": min_down}
    if min_up:
        query["inet_up"] = {"gte": min_up}
    if max_price:
        query["dph_total"] = {"lte": max_price}
    data = _rest(key, "/api/v0/bundles/", query)
    return data.get("offers", [])


def list_instances(key):
    _, out, _ = _cli_run(key, ["--raw", "show", "instances"])
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return []


def create_instance(key, offer_id, image=DEFAULT_IMAGE, disk=60, ports=("8000", "8080"),
                    onstart_cmd=None):
    env = " ".join("-p %s:%s" % (p, p) for p in ports)
    args = ["--raw", "create", "instance", str(offer_id),
            "--image", image, "--disk", str(disk),
            "--ssh", "--direct", "--env", env,
            "--label", "hanhua", "--cancel-unavail"]
    if onstart_cmd:
        args += ["--onstart-cmd", onstart_cmd]
    rc, out, err = _cli_run(key, args)
    try:
        return json.loads(out).get("new_contract")
    except json.JSONDecodeError:
        # Surface WHY the offer was refused. vast reports the reason on stderr
        # (e.g. "no_such_ask" for a stale bundle id, or the 16384-char args
        # limit). Swallowing it turned a clear error into a mystery loop.
        reason = (err or out or "").strip().replace("\n", " ")
        if reason:
            print("  create refused for offer %s: %s" % (offer_id, reason[:300]))
        return None


def bootstrap_script(deploy_dir, model):
    """Bash run by the instance at boot (no SSH needed afterwards).

    Includes an HF speed gate: probe HuggingFace throughput first and write it to
    /root/hf_mbps.txt (exposed via the OCR /health). Only pull the model when HF
    is fast enough, otherwise skip so the controller fails fast and switches
    machines instead of stalling for 30 minutes on a bad-network host.
    """
    import base64
    with open(os.path.join(deploy_dir, "ocr_server.py"), "rb") as handle:
        ocr_b64 = base64.b64encode(handle.read()).decode("ascii")
    probe_url = ("https://huggingface.co/SakuraLLM/Sakura-GalTransl-14B-v3.8/"
                 "resolve/main/Sakura-Galtransl-14B-v3.8.gguf")
    return "\n".join([
        "set -x",
        "export PATH=/usr/local/bin:$PATH",
        "apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0",
        "python -m pip install --quiet --upgrade pip",
        "python -m pip install --quiet manga-ocr easyocr 'transformers==4.46.3' "
        "fastapi 'uvicorn[standard]' python-multipart pillow",
        "python -m pip install --quiet simple-lama-inpainting || true",
        "python -m pip install --quiet 'git+https://github.com/dmMaze/comic-text-detector.git' || true",
        # --- HF speed gate (written before anything that blocks) ---
        "HF=$(curl -sL -r 0-52428799 -m 25 -o /dev/null -w '%%{speed_download}' '%s' 2>/dev/null || echo 0)" % probe_url,
        "HF=${HF:-0}",
        "HF_MBPS=$(( ${HF%%.*} / 1048576 ))",
        "echo \"$HF_MBPS\" > /root/hf_mbps.txt",
        "echo \"HF_MBPS=$HF_MBPS\"",
        # --- OCR server FIRST so /health (with hf_mbps) is reachable during the pull ---
        "echo %s | base64 -d > /root/ocr_server.py" % ocr_b64,
        "nohup env PORT=8080 python /root/ocr_server.py >/root/ocr.log 2>&1 </dev/null &",
        # --- ollama ---
        "curl -fsSL https://ollama.com/install.sh | sh",
        "nohup env OLLAMA_HOST=0.0.0.0:8000 OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_KEEP_ALIVE=30m "
        "ollama serve >/root/ollama.log 2>&1 </dev/null &",
        "sleep 8",
        "if [ \"$HF_MBPS\" -ge 8 ]; then OLLAMA_HOST=127.0.0.1:8000 ollama pull %s; "
        "else echo SKIP_PULL_SLOW_HF; fi" % model,
        "echo SETUP_OK",
    ])


def wait_http(host, sakura_port, ocr_port, timeout=1800, log=print):
    """Poll the two HTTP endpoints until BOTH are truly ready (SSH-free).

    Readiness for the LLM means /v1/models returns a NON-EMPTY model list (a
    server with no model answers 200 with an empty list). The model is baked into
    the image, so boots do not depend on HuggingFace throughput; the measured
    hf_mbps from /health is reported for information only.
    """
    import json as _json
    import urllib.request
    start = time.time()
    sakura_ok = ocr_ok = False
    while time.time() - start < timeout:
        if not ocr_ok:
            try:
                body = urllib.request.urlopen(
                    "http://%s:%s/health" % (host, ocr_port), timeout=10).read().decode("utf-8", "replace")
                info = _json.loads(body)
                ocr_ok = True
                log("  ocr endpoint up (hf_mbps=%s)" % info.get("hf_mbps"))
            except Exception:
                pass
        if not sakura_ok:
            try:
                body = urllib.request.urlopen(
                    "http://%s:%s/v1/models" % (host, sakura_port), timeout=10).read().decode("utf-8", "replace")
                if _json.loads(body).get("data"):
                    sakura_ok = True
                    log("  sakura model ready")
            except Exception:
                pass
        if sakura_ok and ocr_ok:
            return True
        time.sleep(20)
    return False


def instance_info(key, iid):
    _, out, _ = _cli_run(key, ["--raw", "show", "instance", str(iid)])
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {}


def wait_running(key, iid, timeout=1500, log=print):
    """Wait for the instance to run while logging image-pull progress.

    A large image has big layers that download for minutes with no status change,
    so we do NOT use a generic stall timeout. The only early give-up is a pull
    that is repeatedly *retrying* (a real failure).
    """
    start = time.time()
    last_msg = None
    retry_since = None
    info = {}
    while time.time() - start < timeout:
        info = instance_info(key, iid)
        if info.get("actual_status") == "running":
            return info
        msg = (info.get("status_msg") or "").strip()
        now = time.time()
        if msg != last_msg:
            last_msg = msg
            log("  pull %4ds | %s" % (int(now - start), msg[:70]))
            retry_since = now if "retrying" in msg.lower() else None
        if retry_since is not None and now - retry_since > 120:
            log("  pull stuck retrying for 120s -> giving up")
            return info
        time.sleep(15)
    return info


def ssh_targets(key, iid):
    """Return candidate (host, port) targets: direct first, then vast proxy."""
    info = instance_info(key, iid)
    targets = []
    rc, out, _ = _cli_run(key, ["ssh-url", str(iid)])
    out = out.strip().replace("ssh://", "")
    userhost, _, port = out.rpartition(":")
    _, _, host = userhost.rpartition("@")
    if host and port:
        targets.append((host, int(port)))
    proxy_host, proxy_port = info.get("ssh_host"), info.get("ssh_port")
    if proxy_host and proxy_port and (proxy_host, int(proxy_port)) not in targets:
        targets.append((proxy_host, int(proxy_port)))
    return targets


def wait_ssh(priv, targets, timeout=180):
    """Try each target until SSH answers (direct ports may be closed; proxy works)."""
    start = time.time()
    last_err = ""
    while time.time() - start < timeout:
        for host, port in targets:
            rc, out, err = _ssh(priv, host, port, "echo SSH_OK", timeout=30)
            if "SSH_OK" in out:
                return (host, port)
            last_err = (err or "").strip().splitlines()[-1:] or [""]
            last_err = last_err[0]
        time.sleep(10)
    return None


def destroy(key, iid):
    proc = subprocess.run([_cli(), "--api-key", key, "destroy", "instance", str(iid)],
                          input="y\n", capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    return proc.returncode


def destroy_all(key):
    ids = [i.get("id") for i in list_instances(key)]
    for iid in ids:
        destroy(key, iid)


# --- single-instance invariant -------------------------------------------------
# HARD RULE: at most ONE instance of ours may exist at any time. A leftover
# script must never be able to rent a second machine, so every rental path
# records the instance it owns and a watchdog destroys anything else.

CURRENT_FILE = "vast_current.txt"


def current_path(workdir):
    return os.path.join(workdir, CURRENT_FILE)


def set_current(workdir, iid):
    with open(current_path(workdir), "w", encoding="ascii") as handle:
        handle.write(str(iid) + "\n")


def get_current(workdir):
    try:
        with open(current_path(workdir), encoding="ascii") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def clear_current(workdir):
    try:
        os.remove(current_path(workdir))
    except OSError:
        pass


def owned_instances(key):
    """Every instance that carries OUR label (never the user's own ones)."""
    return [i for i in list_instances(key) if i.get("label") == "hanhua"]


def enforce_single(key, workdir, keep=None, log=print):
    """Destroy every instance of ours except `keep` (default: the recorded one).

    This is the guard that makes "two machines at once" impossible: it is called
    before renting, and by the watchdog while a run is in flight.
    """
    keep_ids = {str(keep)} if keep else set()
    recorded = get_current(workdir)
    if recorded:
        keep_ids.add(str(recorded))
    removed = []
    for inst in owned_instances(key):
        iid = str(inst.get("id"))
        if iid in keep_ids:
            continue
        log("ENFORCE-SINGLE: destroying extra instance %s (%s)" % (
            iid, inst.get("actual_status")))
        destroy(key, iid)
        removed.append(iid)
    return removed
    return ids


def _blacklist_path(workdir):
    return os.path.join(workdir, "vast_blacklist.txt")


def load_blacklist(workdir):
    path = _blacklist_path(workdir)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return {line.strip() for line in handle if line.strip()}
    return set()


def add_blacklist(workdir, host):
    if not host:
        return
    with open(_blacklist_path(workdir), "a", encoding="utf-8") as handle:
        handle.write(host + "\n")


def _ssh(key_path, host, port, command, timeout=600):
    cmd = ["ssh", "-i", key_path, "-p", str(port),
           "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
           "-o", "ConnectTimeout=20", "root@%s" % host, command]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def _scp(key_path, host, port, local, remote):
    cmd = ["scp", "-i", key_path, "-P", str(port),
           "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
           local, "root@%s:%s" % (host, remote)]
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def _generate_key(priv):
    import stat
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    key = ed25519.Ed25519PrivateKey.generate()
    data = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    )
    with open(priv, "wb") as handle:
        handle.write(data)
    try:
        os.chmod(priv, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    if os.name == "nt":
        user = os.environ.get("USERNAME", "")
        subprocess.run(["icacls", priv, "/inheritance:r", "/grant:r", user + ":F"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    pub = key.public_key().public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH,
    ).decode("ascii")
    with open(priv + ".pub", "w", encoding="ascii") as handle:
        handle.write(pub + "\n")
    return pub


def ensure_ssh_key(key, workdir):
    """Generate + register a workspace SSH key if needed. Returns private key path."""
    priv = os.path.join(workdir, "vast_ssh")
    if not os.path.exists(priv):
        _generate_key(priv)
    pub = open(priv + ".pub", encoding="ascii").read().strip()
    rc, out, _ = _cli_run(key, ["show", "ssh-keys"])
    if pub not in out:
        _cli_run(key, ["create", "ssh-key", pub, "-y"])
    return priv


DEFAULT_MODEL = "hf.co/SakuraLLM/Sakura-GalTransl-14B-v3.8:Q4_K_M"


def up(key, workdir, deploy_dir, model=DEFAULT_MODEL, min_vram=24, min_down=200,
       min_up=100, max_price=0.18, gpu=None, image=DEFAULT_IMAGE, disk=60,
       dry_run=False, yes=False, attempts=4, baked=False, log=print):
    offers = search_offers(key, min_vram, min_down, min_up, max_price, gpu, "reliable", 20)
    if not offers:
        raise SystemExit("no offers matched (relax --min-down / --max-price / --gpu)")
    if not dry_run and not yes:
        raise SystemExit("refusing to rent without --yes")

    # Safety: at most ONE machine of ours may exist (never the user's others).
    busy = [i for i in owned_instances(key)
            if i.get("actual_status") in ("running", "loading")]
    if busy:
        raise SystemExit(
            "refusing to rent: %s is already %s - one machine at a time. "
            "Use it, or destroy it first." % (
                [i.get("id") for i in busy],
                [i.get("actual_status") for i in busy]))
    enforce_single(key, workdir, keep=None, log=log)

    # baked=True means the image already has the whole stack (Dockerfile). But a
    # vast SSH instance runs its own entrypoint (sshd), so the image CMD does NOT
    # execute -- we must still start the services via onstart.
    if baked:
        script = ("nohup bash /root/container_start.sh >/root/start.log 2>&1 "
                  "</dev/null & echo started")
    else:
        script = bootstrap_script(deploy_dir, model)
    failed_hosts = load_blacklist(workdir)
    tried = 0
    for offer in offers:
        if tried >= attempts:
            break
        host_ip = offer.get("public_ipaddr") or offer.get("host_id")
        if host_ip in failed_hosts:
            continue
        plan = "1 x %s @ $%.3f/hr (inet_down %.0f, reliability %.3f) offer=%s host=%s" % (
            offer.get("gpu_name"), offer.get("dph_total", 0.0), offer.get("inet_down", 0.0),
            offer.get("reliability2", 0.0), offer.get("id"), host_ip)
        log("plan: " + plan)
        if dry_run:
            return None

        iid = create_instance(key, offer["id"], image, disk, onstart_cmd=script)
        if not iid:
            log("create failed for offer %s (unavailable?) -> next" % offer.get("id"))
            continue
        tried += 1
        try:
            log("instance %s created; waiting for running..." % iid)
            info = wait_running(key, iid, log=log)
            if info.get("actual_status") != "running":
                destroy(key, iid)
                failed_hosts.add(host_ip)
                add_blacklist(workdir, host_ip)
                log("instance %s never reached running -> destroyed, next" % iid)
                continue

            host = info.get("public_ipaddr") or host_ip
            ports = info.get("ports", {})
            sakura = ports.get("8000/tcp", [{}])[0].get("HostPort")
            ocr = ports.get("8080/tcp", [{}])[0].get("HostPort")
            if not sakura or not ocr:
                destroy(key, iid)
                failed_hosts.add(host_ip)
                add_blacklist(workdir, host_ip)
                log("no port mapping on %s -> destroyed, next" % iid)
                continue

            log("polling HTTP on %s (sakura %s / ocr %s)" % (host, sakura, ocr))
            if not wait_http(host, sakura, ocr, timeout=1800, log=log):
                destroy(key, iid)
                failed_hosts.add(host_ip)
                add_blacklist(workdir, host_ip)
                log("endpoints never came up on %s -> destroyed, next" % iid)
                continue
            log("endpoints ready")
            set_current(workdir, iid)
            return {
                "instance_id": iid,
                "api_base": "http://%s:%s/v1" % (host, sakura),
                "ocr_url": "http://%s:%s/prepare" % (host, ocr),
                "text_model": model,
            }
        except Exception as exc:  # noqa: BLE001
            # Never leak a billing instance on an unexpected error.
            log("unexpected error on %s: %s -> destroying" % (iid, exc))
            destroy(key, iid)
            raise
    raise SystemExit("all %d attempts failed; stopping (per playbook R2.7)" % attempts)

import shutil
import subprocess
import threading
import uuid

IMAGE = "localbrain-sandbox:1"


def run_python(code, timeout=10):
    """Execute only in an existing Docker image. Never fall back to host exec."""
    if not isinstance(code, str) or len(code.encode("utf-8")) > 65536:
        raise ValueError("Code must be a string no larger than 64 KiB")
    if not isinstance(timeout, int) or not 1 <= timeout <= 30:
        raise ValueError("timeout must be 1..30 seconds")
    docker = shutil.which("docker")
    if not docker:
        raise ValueError("Docker unavailable. Sandbox disabled; host execution is not allowed")
    probe = subprocess.run([docker, "image", "inspect", IMAGE],
                           capture_output=True, timeout=15)
    if probe.returncode:
        raise ValueError("Sandbox image unavailable. Build: docker build -t localbrain-sandbox:1 sandbox")
    name = "localbrain-" + uuid.uuid4().hex
    command = [docker, "run", "--rm", "--pull=never", "--name", name,
               "--network=none", "--read-only", "--cap-drop=ALL",
               "--security-opt=no-new-privileges", "--pids-limit=32",
               "--memory=128m", "--memory-swap=128m", "--cpus=1",
               "--user=65534:65534", "--workdir=/work",
               "--tmpfs=/work:rw,noexec,nosuid,size=16m,mode=1777",
               "--tmpfs=/tmp:rw,noexec,nosuid,size=16m,mode=1777",
               "-i", IMAGE, "python", "-I", "-B", "-u", "-"]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = bytearray()
    exceeded = threading.Event()

    def drain():
        while True:
            block = proc.stdout.read1(4096)
            if not block:
                break
            remaining = 65536 - len(output)
            output.extend(block[:remaining])
            if len(block) > remaining:
                exceeded.set()
                proc.terminate()
                break

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    timed_out = False
    try:
        proc.stdin.write(code.encode("utf-8"))
        proc.stdin.close()
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
    except BrokenPipeError:
        pass
    finally:
        # Killing the CLI alone does not guarantee the container has stopped.
        subprocess.run([docker, "rm", "-f", name], capture_output=True, timeout=15)
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
        reader.join(timeout=5)
        proc.stdout.close()
    return {"returncode": proc.returncode,
            "output": output.decode("utf-8", errors="replace"),
            "timed_out": timed_out, "output_truncated": exceeded.is_set()}

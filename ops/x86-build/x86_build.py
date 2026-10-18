#!/usr/bin/env python3
"""Run an x86-64 build over SSH and bring selected artifacts back."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid


VERSION = "0.1.0"
DEFAULT_CONFIG = Path.home() / ".config" / "x86-build" / "config.json"
REMOTE_PREFIX = "x86-build-"
HOST_RE = re.compile(r"^[A-Za-z0-9_.:@-]+$")


class CliError(RuntimeError):
    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def emit(message: str) -> None:
    print(f"[x86-build] {message}", file=sys.stderr, flush=True)


def load_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CliError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CliError(f"config {path} must contain a JSON object")
    return data


def save_config(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".config-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def validate_host(host: str) -> str:
    if not HOST_RE.fullmatch(host) or host.startswith("-"):
        raise CliError("remote host contains unsupported characters")
    return host


def validate_remote_root(remote_root: str) -> str:
    path = PurePosixPath(remote_root)
    if not path.is_absolute() or path == PurePosixPath("/") or ".." in path.parts:
        raise CliError("remote root must be a safe absolute path other than /")
    return str(path)


def non_negative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return number


def resolve_settings(args: argparse.Namespace) -> tuple[str, str, Path]:
    config_path = Path(args.config).expanduser()
    config = load_config(config_path)
    host = args.host or os.environ.get("X86_BUILD_HOST") or config.get("host")
    remote_root = (
        args.remote_root
        or os.environ.get("X86_BUILD_REMOTE_ROOT")
        or config.get("remote_root")
        or "/tmp"
    )
    if not isinstance(host, str) or not host:
        raise CliError(
            "remote host is missing; run 'x86-build init --host HOST' or set X86_BUILD_HOST"
        )
    if not isinstance(remote_root, str):
        raise CliError("remote root must be a string")
    return validate_host(host), validate_remote_root(remote_root), config_path


def ssh_base() -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
    ]


def remote_run(
    host: str,
    command: list[str],
    *,
    capture: bool = False,
    stdout=None,
) -> subprocess.CompletedProcess[str]:
    remote_command = shlex.join(command)
    return subprocess.run(
        ssh_base() + [host, remote_command],
        check=False,
        text=True,
        capture_output=capture,
        stdout=stdout,
    )


def validate_output_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or value in {"", "."} or ".." in path.parts:
        raise argparse.ArgumentTypeError(
            "output must be a relative file or directory inside the source tree"
        )
    return value


def ensure_local_tools() -> None:
    missing = [name for name in ("ssh", "rsync", "tar") if shutil.which(name) is None]
    if missing:
        raise CliError("missing local tools: " + ", ".join(missing))


def create_remote_workspace(host: str, remote_dir: str) -> None:
    script = (
        "from pathlib import Path; import sys; "
        "p=Path(sys.argv[1]); p.mkdir(parents=True, exist_ok=False); "
        "(p/'src').mkdir(); (p/'.x86-build-managed').write_text('v1\\n')"
    )
    result = remote_run(host, ["python3", "-c", script, remote_dir], capture=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise CliError(f"cannot create remote workspace: {detail}")


def cleanup_remote_workspace(host: str, remote_root: str, remote_dir: str) -> bool:
    script = """
from pathlib import Path
import shutil
import sys

path = Path(sys.argv[1])
root = Path(sys.argv[2])
if path.parent != root or not path.name.startswith("x86-build-"):
    raise SystemExit("refusing unsafe cleanup path")
if not (path / ".x86-build-managed").is_file():
    raise SystemExit("refusing to clean an unmanaged directory")
shutil.rmtree(path, ignore_errors=False)
""".strip()
    result = remote_run(
        host,
        ["python3", "-c", script, remote_dir, remote_root],
        capture=True,
    )
    if result.returncode != 0:
        emit(f"remote cleanup failed: {result.stderr.strip() or result.stdout.strip()}")
        return False
    return True


def upload_source(
    host: str,
    source: Path,
    remote_dir: str,
    excludes: list[str],
) -> None:
    command = [
        "rsync",
        "-a",
        "--delete",
        "-e",
        "ssh -o BatchMode=yes -o ConnectTimeout=10",
    ]
    for pattern in excludes:
        command.extend(["--exclude", pattern])
    command.extend([f"{source}/", f"{host}:{remote_dir}/src/"])
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise CliError("source upload failed", result.returncode)


def run_build(host: str, remote_dir: str, command: list[str], json_mode: bool) -> int:
    remote_command = f"cd {shlex.quote(remote_dir + '/src')} && {shlex.join(command)}"
    try:
        output_stream = sys.stderr if json_mode else None
        result = subprocess.run(
            ssh_base() + [host, remote_command],
            check=False,
            stdout=output_stream,
            stderr=output_stream,
        )
        return result.returncode
    except KeyboardInterrupt:
        emit("interrupted; requesting remote cleanup")
        return 130


def verify_remote_outputs(
    host: str,
    remote_dir: str,
    outputs: list[str],
) -> None:
    script = """
from pathlib import Path
import json
import os
import sys

root = Path(sys.argv[1])
missing = [item for item in sys.argv[2:] if not os.path.lexists(root / item)]
print(json.dumps({"missing": missing}))
raise SystemExit(1 if missing else 0)
""".strip()
    result = remote_run(
        host,
        ["python3", "-c", script, remote_dir + "/src", *outputs],
        capture=True,
    )
    if result.returncode != 0:
        try:
            missing = json.loads(result.stdout).get("missing", outputs)
        except json.JSONDecodeError:
            missing = outputs
        raise CliError("expected output was not produced: " + ", ".join(missing))


def download_outputs(
    host: str,
    source: Path,
    remote_dir: str,
    outputs: list[str],
) -> None:
    remote_command = (
        f"cd {shlex.quote(remote_dir + '/src')} && "
        f"tar -cf - -- {' '.join(shlex.quote(item) for item in outputs)}"
    )
    ssh_process = subprocess.Popen(
        ssh_base() + [host, remote_command],
        stdout=subprocess.PIPE,
    )
    assert ssh_process.stdout is not None
    tar_result = subprocess.run(
        ["tar", "-xf", "-", "-C", str(source)],
        stdin=ssh_process.stdout,
        check=False,
    )
    ssh_process.stdout.close()
    ssh_code = ssh_process.wait()
    if ssh_code != 0 or tar_result.returncode != 0:
        raise CliError("artifact download failed")


def command_init(args: argparse.Namespace) -> int:
    host = validate_host(args.init_host)
    remote_root = validate_remote_root(args.init_remote_root)
    path = Path(args.config).expanduser()
    save_config(
        path,
        {"host": host, "remote_root": remote_root},
    )
    result = {"ok": True, "config": str(path), "host": host}
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        emit(f"saved config to {path}")
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    host, remote_root, config_path = resolve_settings(args)
    local_tools = {name: shutil.which(name) for name in ("ssh", "rsync", "tar")}
    script = """
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

root = Path(sys.argv[1])
tools = {}
for name in ("gcc", "g++", "make", "cmake", "ninja", "pkg-config", "rsync", "tar", "python3"):
    tools[name] = shutil.which(name)
stale = 0
if root.exists():
    stale = sum(
        1
        for path in root.glob("x86-build-*")
        if path.is_dir() and (path / ".x86-build-managed").is_file()
    )
print(json.dumps({
    "architecture": platform.machine(),
    "hostname": platform.node(),
    "python": platform.python_version(),
    "remote_root": str(root),
    "remote_root_exists": root.exists(),
    "stale_workspaces": stale,
    "tools": tools,
}))
""".strip()
    result = remote_run(host, ["python3", "-c", script, remote_root], capture=True)
    remote: dict[str, object] | None = None
    error: str | None = None
    if result.returncode == 0:
        try:
            remote = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            error = f"invalid doctor response: {exc}"
    else:
        error = result.stderr.strip() or "SSH connection failed"
    required_remote = ("gcc", "g++", "make", "rsync", "tar", "python3")
    remote_tools = remote.get("tools", {}) if isinstance(remote, dict) else {}
    ok = (
        all(local_tools.values())
        and remote is not None
        and remote.get("architecture") == "x86_64"
        and isinstance(remote_tools, dict)
        and all(remote_tools.get(name) for name in required_remote)
    )
    payload = {
        "ok": ok,
        "version": VERSION,
        "config": str(config_path),
        "host": host,
        "local_tools": local_tools,
        "remote": remote,
        "error": error,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"host: {host}")
        print(f"remote architecture: {remote.get('architecture') if remote else 'unavailable'}")
        print(f"remote hostname: {remote.get('hostname') if remote else 'unavailable'}")
        print(f"remote stale workspaces: {remote.get('stale_workspaces') if remote else 'unknown'}")
        print(f"status: {'ready' if ok else 'not ready'}")
        if error:
            print(f"error: {error}")
    return 0 if ok else 1


def command_run(args: argparse.Namespace) -> int:
    host, remote_root, _ = resolve_settings(args)
    ensure_local_tools()
    source = Path(args.source).expanduser().resolve()
    if not source.is_dir():
        raise CliError(f"source directory does not exist: {source}")
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise CliError("build command is missing; place it after --")
    outputs = list(dict.fromkeys(args.output))
    build_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    remote_dir = f"{remote_root}/{REMOTE_PREFIX}{build_id}"
    excludes = [".git/", "__pycache__/", "*.pyc", *args.exclude]
    plan = {
        "build_id": build_id,
        "host": host,
        "source": str(source),
        "remote_dir": remote_dir,
        "command": command,
        "outputs": outputs,
        "excludes": excludes,
    }
    if args.dry_run:
        if args.json:
            print(json.dumps({"ok": True, "dry_run": True, **plan}, ensure_ascii=False))
        else:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    remote_created = False
    cleaned = False
    build_code = 1
    error: str | None = None
    try:
        emit(f"creating {remote_dir}")
        create_remote_workspace(host, remote_dir)
        remote_created = True
        emit(f"uploading {source}")
        upload_source(host, source, remote_dir, excludes)
        emit("running: " + shlex.join(command))
        build_code = run_build(host, remote_dir, command, args.json)
        if build_code != 0:
            error = f"remote build failed with exit code {build_code}"
        else:
            verify_remote_outputs(host, remote_dir, outputs)
            emit("downloading: " + ", ".join(outputs))
            download_outputs(host, source, remote_dir, outputs)
    except CliError as exc:
        error = str(exc)
        build_code = exc.exit_code
    finally:
        if remote_created:
            emit(f"cleaning {remote_dir}")
            cleaned = cleanup_remote_workspace(host, remote_root, remote_dir)

    ok = build_code == 0 and error is None and cleaned
    if build_code == 0 and not cleaned:
        build_code = 74
        error = "build succeeded but remote cleanup failed"
    payload = {
        "ok": ok,
        "build_id": build_id,
        "host": host,
        "source": str(source),
        "command": command,
        "outputs": outputs if ok else [],
        "remote_cleaned": cleaned,
        "exit_code": build_code,
        "error": error,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    elif ok:
        emit("build completed; artifacts returned and remote workspace removed")
    else:
        emit(error or "build failed")
    return build_code


def command_gc(args: argparse.Namespace) -> int:
    host, remote_root, _ = resolve_settings(args)
    cutoff_seconds = 0 if args.all else int(args.older_than_hours * 3600)
    script = """
from pathlib import Path
import json
import shutil
import sys
import time

root = Path(sys.argv[1])
cutoff = int(sys.argv[2])
now = time.time()
removed = []
if root.exists():
    for path in root.glob("x86-build-*"):
        if path.parent != root or not path.is_dir():
            continue
        if not (path / ".x86-build-managed").is_file():
            continue
        age = now - path.stat().st_mtime
        if cutoff == 0 or age >= cutoff:
            shutil.rmtree(path)
            removed.append(path.name)
print(json.dumps({"removed": removed}))
""".strip()
    result = remote_run(
        host,
        ["python3", "-c", script, remote_root, str(cutoff_seconds)],
        capture=True,
    )
    if result.returncode != 0:
        raise CliError(result.stderr.strip() or "remote cleanup failed")
    payload = json.loads(result.stdout)
    payload.update({"ok": True, "host": host})
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        removed = payload["removed"]
        print("removed: " + (", ".join(removed) if removed else "none"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="x86-build",
        description="Build a local source tree on a temporary remote x86-64 workspace.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="config JSON path")
    parser.add_argument("--host", help="SSH host; overrides config and X86_BUILD_HOST")
    parser.add_argument("--remote-root", help="remote temporary root; default /tmp")
    parser.add_argument("--json", action="store_true", help="emit a JSON result")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    init_parser = subparsers.add_parser("init", help="save host configuration")
    init_parser.add_argument("--host", dest="init_host", required=True)
    init_parser.add_argument("--remote-root", dest="init_remote_root", default="/tmp")
    init_parser.set_defaults(handler=command_init)

    doctor_parser = subparsers.add_parser("doctor", help="check SSH and build tools")
    doctor_parser.set_defaults(handler=command_doctor)

    run_parser = subparsers.add_parser("run", help="upload, build, download, and clean")
    run_parser.add_argument("--source", default=".", help="source directory; default current directory")
    run_parser.add_argument(
        "--output",
        action="append",
        required=True,
        type=validate_output_path,
        help="relative artifact path to download; repeatable",
    )
    run_parser.add_argument("--exclude", action="append", default=[], help="extra rsync exclusion")
    run_parser.add_argument("--dry-run", action="store_true", help="show the plan without connecting")
    run_parser.add_argument("command", nargs=argparse.REMAINDER, help="command after --")
    run_parser.set_defaults(handler=command_run)

    gc_parser = subparsers.add_parser("gc", help="remove stale remote workspaces")
    gc_parser.add_argument("--older-than-hours", type=non_negative_float, default=24.0)
    gc_parser.add_argument("--all", action="store_true", help="remove every managed x86-build workspace")
    gc_parser.set_defaults(handler=command_gc)
    return parser


def main() -> int:
    signal.signal(signal.SIGTERM, lambda _signum, _frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        emit("interrupted")
        return 130
    except CliError as exc:
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        else:
            emit(str(exc))
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

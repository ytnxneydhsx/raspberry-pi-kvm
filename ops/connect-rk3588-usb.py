"""Run on the Pi: configure the verified RK3588 USB link and dedicated SSH key.

Requires the RK3588 Ubuntu ACM console's existing huangnv automatic login.
No firmware, boot configuration, or RK network settings are changed.
"""
import base64
import os
from pathlib import Path
import select
import shlex
import subprocess
import termios
import time
import tty


def run(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


home = Path.home()
serial = Path('/dev/serial/by-id/usb-RK3588_Linux_RK3588_Ubuntu_Console_22b2617583d2c0b7-linux-if00')
if not serial.exists():
    raise SystemExit('Expected RK3588 USB console is absent')
if subprocess.run(['/usr/bin/fuser', str(serial.resolve())], capture_output=True).returncode == 0:
    raise SystemExit('The USB console is already in use')

state = home / '.config/rk3588-host'
state.mkdir(parents=True, exist_ok=True, mode=0o700)
profile = '0048f195-afa1-34ac-a6b0-ebef57a1b2e2'
active = subprocess.check_output(['nmcli', '-g', 'GENERAL.CON-UUID', 'device', 'show', 'usb0'], text=True).strip()
bound_interface = subprocess.check_output(['nmcli', '-g', 'connection.interface-name', 'connection', 'show', 'uuid', profile], text=True).strip()
if active not in (profile, '--', '') or bound_interface != 'usb0':
    raise SystemExit('USB network profile changed; inspect before rerunning')
backup = state / 'usb-profile-before.txt'
if not backup.exists():
    backup.write_text(subprocess.check_output(['nmcli', 'connection', 'show', 'uuid', profile], text=True))
run(['sudo', '-n', 'nmcli', 'connection', 'modify', 'uuid', profile,
     'connection.id', 'rk3588-usb', 'connection.interface-name', 'usb0',
     'connection.autoconnect', 'yes', 'ipv4.method', 'manual',
     'ipv4.addresses', '169.254.4.2/16', 'ipv4.gateway', '', 'ipv4.dns', '',
     'ipv4.never-default', 'yes', 'ipv6.method', 'link-local', 'ipv6.never-default', 'yes'])
run(['sudo', '-n', 'nmcli', 'connection', 'up', 'uuid', profile])

ssh = home / '.ssh'
ssh.mkdir(mode=0o700, exist_ok=True)
ssh.chmod(0o700)
identity = ssh / 'id_ed25519_rk3588_usb'
if not identity.exists():
    run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(identity), '-C', 'rpi4b-rk3588-usb'])
identity.chmod(0o600)
public_key = identity.with_suffix('.pub').read_text().strip()

remote_code = f'''from pathlib import Path
import shutil,time
p=Path.home()/'.ssh';p.mkdir(mode=0o700,exist_ok=True);p.chmod(0o700)
a=p/'authorized_keys'
old=a.read_text() if a.exists() else ''
key={public_key!r}
if key not in old.splitlines():
 if a.exists():shutil.copy2(a,p/('authorized_keys.before-rpi-usb-'+time.strftime('%Y%m%dT%H%M%S')))
 with a.open('a') as f:f.write(('' if not old or old.endswith(chr(10)) else chr(10))+key+chr(10))
a.chmod(0o600)
h=Path('/etc/ssh/ssh_host_ed25519_key.pub').read_text().split()
print('__RK_HOST_KEY__ '+h[0]+' '+h[1],flush=True)
print('__RK_KEY_READY__',flush=True)
'''
# Keep the command comfortably below the console's canonical input limit.
payload = base64.b64encode(remote_code.encode()).decode()
command = ' HISTFILE=/dev/null; python3 -c ' + shlex.quote('import base64;exec(base64.b64decode(' + repr(payload) + '))') + chr(13)
fd = os.open(serial, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
try:
    # Leave the host tty raw/no-echo: restoring ECHO would reflect console text
    # back into the RK shell when no terminal reader is attached.
    tty.setraw(fd, termios.TCSANOW)
    attr = termios.tcgetattr(fd)
    attr[4] = attr[5] = termios.B115200
    attr[2] |= termios.CLOCAL | termios.CREAD
    attr[2] &= ~termios.HUPCL
    termios.tcsetattr(fd, termios.TCSANOW, attr)
    while select.select([fd], [], [], 0)[0]:
        os.read(fd, 8192)
    data = command.encode()
    while data:
        select.select([], [fd], [], 3)
        count = os.write(fd, data)
        data = data[count:]
    response = bytearray()
    deadline = time.monotonic() + 10
    host_key = None
    ready = False
    while time.monotonic() < deadline:
        if select.select([fd], [], [], 0.3)[0]:
            response.extend(os.read(fd, 8192))
        lines = response.splitlines()
        keys = [line.removeprefix(b'__RK_HOST_KEY__ ') for line in lines if line.startswith(b'__RK_HOST_KEY__ ssh-ed25519 ')]
        if keys:
            host_key = keys[-1].decode()
        ready = b'__RK_KEY_READY__' in lines
        if host_key and ready:
            break
    if not host_key or not ready:
        raise SystemExit('RK console did not confirm key installation')
finally:
    os.close(fd)

known_hosts = ssh / 'known_hosts_rk3588_usb'
new_host = 'rk3588-usb ' + host_key + chr(10)
if known_hosts.exists() and known_hosts.read_text() != new_host:
    raise SystemExit('Existing RK pinned host key differs from the console key')
known_hosts.write_text(new_host)
known_hosts.chmod(0o600)
config = ssh / 'config'
old_config = config.read_text() if config.exists() else ''
if 'Host rk3588-usb' not in old_config.splitlines():
    stanza = f'''
# RK3588 USB NCM link, host key verified through its ACM console
Host rk3588-usb
    HostName 169.254.4.1
    User huangnv
    IdentityFile {identity}
    IdentitiesOnly yes
    UserKnownHostsFile {known_hosts}
    HostKeyAlias rk3588-usb
    StrictHostKeyChecking yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
'''
    config.write_text(old_config + stanza)
    config.chmod(0o600)

print('USB network and dedicated SSH identity configured; host key pinned from ACM')
run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'rk3588-usb',
     "echo RK_USB_SSH_READY; hostname; whoami; uname -r; ip -br address show usb0"])

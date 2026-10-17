#!/usr/bin/env python3
"""Require SSH keys for connections delivered locally by the FRP client."""
from pathlib import Path
import datetime
import subprocess

path = Path('/etc/ssh/sshd_config')
begin = '# BEGIN RPI4B PUBLIC TUNNEL'
end = '# END RPI4B PUBLIC TUNNEL'
original = path.read_text()
backup = path.with_name('sshd_config.before-rpi4b-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
backup.write_text(original)
base = original
if begin in base:
    left, remaining = base.split(begin, 1)
    _, right = remaining.split(end, 1)
    base = left + right
block = '''
# BEGIN RPI4B PUBLIC TUNNEL
Match Address 127.0.0.1,::1
    PasswordAuthentication no
    KbdInteractiveAuthentication no
    AuthenticationMethods publickey
    PermitRootLogin no
# END RPI4B PUBLIC TUNNEL
'''
try:
    path.write_text(base.rstrip() + '\n' + block)
    subprocess.run(['/usr/sbin/sshd', '-t'], check=True)
    for source, password in [('127.0.0.1', 'no'), ('192.168.77.1', 'yes')]:
        text = subprocess.check_output(['/usr/sbin/sshd', '-T', '-C', f'user=huangnv,host=localhost,addr={source}'], text=True)
        options = dict(line.split(' ', 1) for line in text.splitlines())
        assert options['passwordauthentication'] == password, (source, options['passwordauthentication'])
        if source == '127.0.0.1':
            assert options['authenticationmethods'] == 'publickey'
    subprocess.run(['systemctl', 'reload', 'ssh'], check=True)
except BaseException:
    path.write_text(original)
    raise
print('SSH config validated and reloaded: local tunnel requires public key; LAN password login retained.')
print('Backup:', backup)

"""Set up the user's shell environment without containing login credentials."""
from pathlib import Path
import shutil
import time

home = Path.home()
block = '''# BEGIN RPI CODEX PROXY
if [ -r "$HOME/.config/mihomo/proxy-env.sh" ]; then
    . "$HOME/.config/mihomo/proxy-env.sh"
fi
# END RPI CODEX PROXY
'''
stamp = time.strftime('%Y%m%dT%H%M%S')
for name in ('.bashrc', '.profile'):
    path = home / name
    old = path.read_text() if path.exists() else ''
    if '# BEGIN RPI CODEX PROXY' not in old:
        if path.exists():
            shutil.copy2(path, path.with_name(name + '.before-codex-' + stamp))
        path.write_text(block + '\n' + old)

codex_dir = home / '.codex'
codex_dir.mkdir(mode=0o700, exist_ok=True)
codex_dir.chmod(0o700)
config = codex_dir / 'config.toml'
if not config.exists():
    config.write_text('model = "gpt-6.1-sol"\nmodel_reasoning_effort = "high"\ncli_auth_credentials_store = "file"\n')
    config.chmod(0o600)
print('Shell environment and minimal Codex configuration ready')

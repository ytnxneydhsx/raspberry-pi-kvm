#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
install_dir=${XDG_BIN_HOME:-"$HOME/.local/bin"}

install -d "$install_dir"
install -m 0755 "$script_dir/x86_build.py" "$install_dir/x86-build"

printf 'installed: %s/x86-build\n' "$install_dir"
case ":$PATH:" in
    *":$install_dir:"*) ;;
    *) printf 'add this directory to PATH: %s\n' "$install_dir" ;;
esac

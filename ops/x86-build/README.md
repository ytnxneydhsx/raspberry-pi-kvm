# x86-build

`x86-build` 从树莓派上传一个源码目录到临时的 x86-64 SSH 工作区，在远端执行指定命令，取回显式声明的产物，然后删除远端工作区。

## 安装

```bash
cd /path/to/raspberry-pi-kvm/ops/x86-build
./install.sh
x86-build init --host 10.191.32.121
x86-build doctor
```

配置保存在 `~/.config/x86-build/config.json`，不包含 SSH 私钥。SSH 用户和密钥由 `~/.ssh/config` 管理。

## 构建

```bash
x86-build run \
  --source ./project \
  --output build/app \
  -- make -j4
```

多个产物可以重复声明：

```bash
x86-build run \
  --source ./project \
  --output build/app \
  --output build/app.map \
  -- make -j4
```

简单 GCC 示例：

```bash
x86-build run \
  --source ./hello \
  --output hello \
  -- gcc -O2 -Wall hello.c -o hello
```

Rockchip `boot_merger` 示例：

```bash
x86-build run \
  --source ~/rk3588-official/rkbin \
  --output rk3588_loader_v1.24.114.bin \
  -- ./tools/boot_merger RKBOOT/RK3588MINIALL.ini
```

## 命令

- `x86-build init --host HOST`：保存默认 SSH 主机和远端临时根目录。
- `x86-build doctor`：检查 SSH、x86-64 架构和 GCC/G++/Make/rsync/tar/Python。
- `x86-build run ... -- COMMAND ...`：执行一次临时远端构建。
- `x86-build gc`：删除超过 24 小时的遗留工作区。
- `x86-build gc --all`：删除所有名为 `/tmp/x86-build-*` 的工作区。

`--json` 放在子命令之前，例如：

```bash
x86-build --json doctor
```

JSON 结果始终包含 `ok`。构建结果还包含 `build_id`、`exit_code`、`outputs` 和 `remote_cleaned`。远端构建日志保持在标准错误流，避免污染 JSON 标准输出。

## 边界

- 源目录中的 `.git/`、`__pycache__/` 和 `*.pyc` 默认不上传。
- `--output` 只接受源目录内的相对路径。
- 远端清理只允许删除指定临时根目录下、同时带有 `.x86-build-managed` 标记的 `x86-build-*` 目录。
- 构建失败时不取回产物，但仍会尝试清理远端工作区。
- 网络中断或远端崩溃可能使最后一次清理无法到达；恢复连接后使用 `x86-build gc` 清理过期目录。

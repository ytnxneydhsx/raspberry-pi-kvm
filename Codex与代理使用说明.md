# Codex 与代理使用说明

适用对象：树莓派 4B 用户 `huangnv`。

## 已安装配置

- SSH 别名：`rpi4b-aliyun`
- mihomo：`v1.19.31`，路径 `/usr/local/bin/mihomo`
- 服务：systemd 服务 `mihomo`
- 代理：`127.0.0.1:7890`，HTTP/SOCKS 混合端口，仅本机监听
- TUN：未启用
- FRP：保持现状，不修改
- Mac Shadowrocket：当前仅迁移选中的 `Azure-JP-Direct` 节点，不做订阅全量同步
- Codex CLI：官方 ARM64 独立版 `0.163.0`，路径 `~/.local/bin/codex`
- 默认模型：`gpt-6.1-sol`，默认思考强度 `high`
- Codex 登录凭证：使用本机 ChatGPT 登录副本，文件为 `~/.codex/auth.json`
- 凭证权限要求：目录 `~/.codex` 为 `700`，`auth.json` 为 `600`

## SSH 登录

在 Mac 上执行：

```bash
ssh rpi4b-aliyun
```

## 项目仓库与 GitHub

树莓派上的项目路径：

```text
/home/huangnv/src/raspberry-pi-kvm
```

仓库远程为 `git@github.com:ytnxneydhsx/raspberry-pi-kvm.git`。树莓派已配置 Git 提交身份 `娜娜米 <ytnxneydhsx@users.noreply.github.com>`，并使用独立 SSH 密钥访问 GitHub 账号 `ytnxneydhsx`。私钥只保存于树莓派 `~/.ssh/`，仓库不保存密钥内容。

在树莓派开始修改前：

```bash
cd ~/src/raspberry-pi-kvm
git status
git pull --ff-only
```

完成修改后按项目 `AGENTS.md` 的要求提交并推送。Mac 端需在后续修改前拉取远程更新，使两端基于同一个提交继续工作。

## mihomo 服务

查看状态：

```bash
sudo systemctl status mihomo
```

重启服务：

```bash
sudo systemctl restart mihomo
```

查看日志：

```bash
sudo journalctl -u mihomo
```

## 终端代理环境

代理环境脚本位于：

```text
~/.config/mihomo/proxy-env.sh
```

该脚本由 `.bashrc` 和 `.profile` 加载。启用该环境的终端程序会使用 `127.0.0.1:7890` 代理；未加载该环境的程序不会自动使用代理。当前未启用 TUN，因此不会接管所有系统流量。

## Codex CLI

登录树莓派后，在需要工作的目录运行：

```bash
codex
```

查看 CLI：

```bash
codex --version
```

查看登录状态：

```bash
codex login status
```

CLI 位于：

```text
~/.local/bin/codex
```

凭证文件位置和权限：

```text
~/.codex/auth.json  # 600
~/.codex/           # 700
```

文档：

- [Codex CLI 官方文档](https://learn.chatgpt.com/docs/codex/cli)
- [Codex 登录与认证](https://learn.chatgpt.com/docs/auth)
- [mihomo v1.19.31 官方发布页](https://github.com/MetaCubeX/mihomo/releases/tag/v1.19.31)

## 节点迁移说明

当前只迁移 Mac Shadowrocket 已选中的 `Azure-JP-Direct` 节点。更换 Mac 上的选中节点不会自动同步到树莓派；节点过期或更换时，需要更新树莓派的 `/etc/mihomo/config.yaml` 并重启服务。配置文件含节点凭证，权限为 `root:huangnv 640`，不要公开分享。

Codex 使用独立保存的登录副本，能够自行刷新有效登录。若以后提示登录失效，可在树莓派上执行 `codex login --device-auth`，或从已登录的可信电脑重新安全迁移凭证。不要公开分享 `auth.json`。

FZU 校园网仍需要有效的校园网认证会话；会话到期时需重新认证。mihomo 已设为开机启动，终端代理环境随新终端加载。

## 验证状态

2026-10-16 在实际树莓派上完成：

- 通过 `rpi4b-aliyun` 公网 SSH 登录成功，原有 `frpc-rpi4b` 服务正常。
- mihomo 服务正常、已启用开机启动，实际只监听 `127.0.0.1:7890`。
- 新 SSH 会话自动加载代理环境，外部 HTTPS 检查返回 `204`；OpenAI API 未携带凭证的请求返回 `401`，确认网络连通。
- `codex --version` 返回 `codex-cli 0.163.0`；`codex login status` 返回 `Logged in using ChatGPT`。
- 使用 `gpt-6.1-sol` 发起一次临时只读对话，模型实际返回 `RPI_CODEX_READY`，退出码为 `0`；未调用工具或修改项目文件。
- 对话启动时，账户云端 MCP 连接曾出现一次传输告警，随后模型正常回复；第三方连接器功能尚未验证。
- 官方 ARM64 安装包在 Mac 下载、完成 SHA-256 校验后通过 SSH 传入树莓派，远端再次校验。mihomo 安装包同样通过官方 SHA-256 校验。
- 安装暂存的重复节点配置及大型压缩包已清理；运行所需的配置与登录文件保留在上述受限位置。

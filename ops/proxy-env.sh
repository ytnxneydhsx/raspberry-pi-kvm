# Local-only mixed HTTP/SOCKS listener managed by mihomo.service.
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5h://127.0.0.1:7890
export NO_PROXY=localhost,127.0.0.1,::1,huangnv.local,121.41.75.52,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export all_proxy="$ALL_PROXY"
export no_proxy="$NO_PROXY"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) export PATH="$HOME/.local/bin:$PATH" ;;
esac

"""通过 pySerial 实现串口能力，构造对象不会打开设备。"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from device_discovery.collector import DeviceResource

from .base import SerialChannel
from .errors import (
    ChannelStateError,
    ResourceUnavailableError,
    SerialConnectionError,
    SerialWriteTimeoutError,
)
from .models import ChannelState, SerialConfig


class PySerialChannel(SerialChannel):
    def __init__(
        self,
        resource: DeviceResource,
        config: SerialConfig,
        *,
        ensure_available: Callable[[], None],
        lock: Any,
    ) -> None:
        self._resource = resource
        self._config = config
        self._ensure_available = ensure_available
        # 与 USB 资源更新共享同一把锁，避免打开和拔出处理交叉执行。
        self._lock = lock
        self._read_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._serial: Any | None = None
        self._state = ChannelState.CLOSED
        self._last_error: str | None = None

    @property
    def resource(self) -> DeviceResource:
        with self._lock:
            return self._resource

    @property
    def config(self) -> SerialConfig:
        with self._lock:
            return self._config

    @property
    def state(self) -> ChannelState:
        with self._lock:
            return self._state

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def configure(self, config: SerialConfig) -> None:
        if not isinstance(config, SerialConfig):
            raise TypeError("config 必须是 SerialConfig")
        with self._lock:
            if self._state is ChannelState.OPEN:
                raise ChannelStateError("请先关闭连接，再修改串口配置")
            self._config = config

    def open(self) -> None:
        with self._lock:
            if self._state is ChannelState.OPEN:
                raise ChannelStateError("串口已经打开")
            try:
                self._ensure_available()
            except ResourceUnavailableError as exc:
                self._state = ChannelState.DISCONNECTED
                self._last_error = str(exc)
                raise
            try:
                import serial
            except ImportError as exc:
                self._last_error = "缺少 pyserial，请安装 code/requirements.txt 中的依赖"
                raise SerialConnectionError(self._last_error) from exc

            config = self._config
            port = None
            try:
                # 延迟打开，先设置 DTR/RTS 请求值；不发送探测字节或命令。
                port = serial.Serial(
                    port=None,
                    baudrate=config.baudrate,
                    bytesize=config.bytesize,
                    parity=config.parity,
                    stopbits=config.stopbits,
                    timeout=config.timeout,
                    write_timeout=config.write_timeout,
                    xonxoff=config.flow_control == "xonxoff",
                    rtscts=config.flow_control == "rtscts",
                    dsrdtr=False,
                    exclusive=True,
                )
                port.dtr = config.dtr
                port.rts = config.rts
                port.port = self._resource.device_node
                port.open()
            except (serial.SerialException, OSError, ValueError, OverflowError) as exc:
                if port is not None:
                    try:
                        port.close()
                    except OSError:
                        pass
                self._last_error = str(exc)
                raise SerialConnectionError(f"无法打开串口：{self._resource.device_node}") from exc
            self._serial = port
            self._state = ChannelState.OPEN
            self._last_error = None

    def close(self) -> None:
        with self._lock:
            try:
                self._release()
            except OSError as exc:
                self._last_error = str(exc)
                raise SerialConnectionError("关闭串口失败") from exc
            finally:
                self._state = ChannelState.CLOSED

    def read(self, size: int = 4096) -> bytes:
        if type(size) is not int or size <= 0:
            raise ValueError("size 必须是正整数")
        with self._read_lock:
            with self._lock:
                port = self._require_open()
            try:
                data = port.read(size)
            except OSError as exc:
                self._disconnect_if_current(port, str(exc))
                raise SerialConnectionError("串口读取失败") from exc
            except TypeError as exc:
                # 某些平台在并发关闭文件描述符后会从底层抛 TypeError。
                with self._lock:
                    if self._serial is not port:
                        raise ChannelStateError("读取期间连接已经关闭或断开") from exc
                raise
            with self._lock:
                if self._serial is not port:
                    raise ChannelStateError("读取期间连接已经关闭或断开")
            return data

    def write(self, data: bytes) -> int:
        if not isinstance(data, bytes):
            raise TypeError("data 必须是 bytes；文本由调用方显式编码")
        with self._write_lock:
            with self._lock:
                port = self._require_open()
            import serial

            try:
                written = port.write(data)
            except serial.SerialTimeoutException as exc:
                message = "串口写入超时，数据可能已部分发送"
                with self._lock:
                    if self._serial is port:
                        self._last_error = message
                raise SerialWriteTimeoutError(message) from exc
            except OSError as exc:
                self._disconnect_if_current(port, str(exc))
                raise SerialConnectionError("串口写入失败，数据可能已部分发送") from exc
            except TypeError as exc:
                with self._lock:
                    if self._serial is not port:
                        raise ChannelStateError("写入期间连接已经关闭，数据可能已部分发送") from exc
                raise
            with self._lock:
                if self._serial is not port:
                    raise ChannelStateError("写入期间连接已经关闭，数据可能已部分发送")
            return written

    def refresh_resource(self, resource: DeviceResource) -> None:
        """仅同步相同入口的新发现元数据，由 USB 容器在共享锁内调用。"""
        with self._lock:
            self._resource = resource

    def mark_disconnected(self, reason: str) -> None:
        """由资源变化或实际 I/O 错误触发；不会自动重连。"""
        with self._lock:
            try:
                self._release()
            except OSError as exc:
                reason = f"{reason}；释放连接失败：{exc}"
            self._state = ChannelState.DISCONNECTED
            self._last_error = reason

    def _require_open(self) -> Any:
        if self._state is not ChannelState.OPEN or self._serial is None:
            raise ChannelStateError("串口尚未打开或已断开，请手动连接")
        return self._serial

    def _disconnect_if_current(self, port: Any, reason: str) -> None:
        with self._lock:
            if self._serial is port:
                self.mark_disconnected(reason)

    def _release(self) -> None:
        port = self._serial
        self._serial = None
        if port is not None:
            try:
                port.cancel_read()
                port.cancel_write()
            finally:
                port.close()

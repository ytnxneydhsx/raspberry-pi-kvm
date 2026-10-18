"""设备能力接入层的公开异常。"""


class DeviceAccessError(Exception):
    """设备能力调用失败。"""


class InvalidSerialConfigError(DeviceAccessError, ValueError):
    """串口配置的字段或取值不合法。"""


class ResourceUnavailableError(DeviceAccessError):
    """设备或选中的资源当前不可用。"""


class ChannelStateError(DeviceAccessError):
    """连接状态不允许执行当前操作。"""


class SerialConnectionError(DeviceAccessError):
    """串口打开、关闭或收发失败。"""


class SerialWriteTimeoutError(SerialConnectionError):
    """写入超时；数据可能已部分发送，调用方不得盲目重试。"""


class ProfileStoreError(DeviceAccessError):
    """命名串口配置方案读写失败。"""

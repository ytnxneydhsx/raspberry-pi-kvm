"""系统设备采集模块公开异常。"""


class DeviceCollectorError(Exception):
    """设备采集失败的基类。"""


class CollectorConfigurationError(DeviceCollectorError):
    """设备采集配置缺失或格式不正确。"""


class CollectorUnavailableError(DeviceCollectorError):
    """当前系统无法提供 Linux udev 采集能力。"""


class DeviceScanError(DeviceCollectorError):
    """扫描当前 USB 设备失败。"""


class DeviceMonitorError(DeviceCollectorError):
    """设备热插拔监听启动或停止失败。"""


class CollectorStateError(DeviceCollectorError):
    """设备采集器当前状态不允许执行请求。"""

"""设备连接历史模块公开异常。"""


class DeviceHistoryError(Exception):
    """设备连接历史操作失败的基类。"""


class HistoryConfigurationError(DeviceHistoryError):
    """历史模块配置缺失或格式不正确。"""


class InvalidHistoryRecordError(DeviceHistoryError):
    """待保存的历史记录字段不符合约定。"""


class HistoryFileError(DeviceHistoryError):
    """历史文件或锁文件操作失败。"""

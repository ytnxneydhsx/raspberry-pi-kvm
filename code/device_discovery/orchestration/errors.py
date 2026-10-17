"""设备发现编排层公开异常。"""


class OrchestrationError(Exception):
    """设备发现编排失败的基类。"""


class OrchestratorStateError(OrchestrationError):
    """编排器当前状态不允许执行请求。"""


class OrchestratorStartError(OrchestrationError):
    """监听建立或启动扫描失败。"""


class OrchestratorStopError(OrchestrationError):
    """停止监听失败。"""


class OrchestratorScanError(OrchestrationError):
    """无法取得当前设备列表。"""

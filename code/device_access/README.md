# USB 多资源接入与串口通信

这个模块接在 `device_discovery` 的正常识别结果后面。一个 USB 父设备保留自己的 VID/PID 和多个资源：例如 RK3588 正常运行时，一个 Gadget 同时提供 ACM 虚拟串口与 NCM 网络接口。`UsbDeviceAccess` 组织这些资源，`SerialChannel` 定义显式配置、打开、关闭和原始字节收发的 ABC 接口，内部由 pySerial 实现。

## 当前范围

- 从 `RecognizedDevice` 创建 USB 能力对象，保留匹配名称、完整观察结果及资源列表。
- `serial_resources` 只包含有 `device_node` 的 `tty` 资源；`network_resources` 保留 `net` 资源。USB interface 元数据仍在 `resources` 中。
- 串口配置项统一提供，ACM 与 CH340 等标准串口共用实现，每个资源独立创建连接和选择配置。
- 保存、读取、列出、删除命名串口方案；方案不绑定设备、USB 位置或板子。
- 由调用方显式 `open()`；发现设备、创建对象、列举配置或获取串口对象都不打开端口。
- `read()` 返回原始 `bytes`，`write()` 返回实际写入字节数；不自动解码、分行、发送探测数据或保存通信正文。
- 调用方把后续发现结果交给 `update()`，资源移除后释放相应连接并标记 `disconnected`；没有自动重连。
- USB 父设备移除后，旧对象保持失效；重新插入时创建新 USB 对象，再由用户选择入口和手动连接。

网络部分当前只提供已有的接口资源信息，不设置 IP、路由，不执行 SSH。网络接口存在不代表 SSH 服务可用。Loader/Maskrom、烧录、网页、整板操作权和持续日志采集均留在后续能力中。默认规则 `USB_OTG = 1d6b:0104` 是当前已观察到的 Gadget 类型规则，不能据此认定所有匹配设备都属于同一块 RK3588，或必定有串口和网络。

## 为什么这样分

原有发现模块已经按 USB 父设备归并资源。新模块消费其公开结果，不改 VID/PID 匹配和先存历史再输出的链路。Linux 和 pySerial 已提供统一串口接口，因此不为 CH340、CH341、CDC ACM 分别复制通信类；新增的薄适配层用于 ABC 调用约定、配置方案、资源生命周期和一致的错误处理。实际收益是调用方可以明确选择资源、确认参数后通信，不据此承诺吞吐量或时延改善。

USB 父对象负责多资源组织；字节流接口只给串口。特殊芯片的能力限制需要以后按实际资料补充，通用配置选项不伪装成驱动精确返回的全部能力。

## 环境与目录

与现有采集器一样，USB 系统资源复核依赖 Linux sysfs。Python 3.10+，依赖见 `../requirements.txt`，新增 `pyserial==3.5`。安装依赖会修改 Python 环境，由实际部署操作显式执行：

```bash
python3 -m pip install -r code/requirements.txt
```

```text
device_access/
  __init__.py       公开类型和工厂
  base.py           SerialChannel 抽象基类
  models.py         SerialConfig、ConfigField、ChannelState
  errors.py         公开异常
  _usb.py           一个 USB 设备及其资源、连接生命周期
  _serial.py        pySerial 适配
  _profiles.py      带文件锁和原子替换的命名方案存储
  _config.py        从 application.yaml 定位方案文件
```

业务代码只从 `device_access` 导入；不要引用下划线实现文件。源码根目录仍为 `code/`，与发现模块使用相同的 `PYTHONPATH=code` 方式。

## 配置项与命名方案

`get_serial_config_fields()` 返回应用定义的通用配置项说明。波特率候选只是常用建议值，可填其他正整数；数据位、校验位、停止位、流控等在 `SerialConfig` 构造时完成格式校验。具体组合、精度以及是否忽略某项，仍由系统驱动和设备实现决定。

默认方案路径在 `application.yaml`：

```yaml
device_access:
  serial_profiles:
    storage_file: ../storage/serial-profiles/profiles.json
```

相对路径以 YAML 所在目录为基准。下面是供阅读的调用示例，没有在本轮执行；`initialize()` 和 `save()` 会写方案文件：

```python
from device_access import SerialConfig, create_profile_store, get_serial_config_fields

fields = get_serial_config_fields()  # 无设备或文件写入
profiles = create_profile_store()  # 构造对象，不创建文件
profiles.initialize()
profiles.save("ACM 调试", SerialConfig(baudrate=115200))
config = profiles.get("ACM 调试")
```

方案文件带版本号，严格检查字段，未知字段和损坏 JSON 报错，不用默认值覆盖。独立 `.lock` 上的 `flock` 保护读改写，使用同目录临时文件、0600 权限和原子替换；同步文件内容但未同步父目录，不承诺突然断电后的完整持久性。运行方案及锁文件不提交 Git。当前只支持 Linux/macOS 的文件锁，USB 接入本身要求 Linux。

## 接入发现回调

发现模块的 `RecognizedDevice` 是观察结果，不会自动变成已连接会话。调用方持有 USB 对象，并把扫描或事件结果交给它。下面示例会在启动发现服务时写设备历史，属于后续运行入口示例，本轮没有执行：

```python
import threading

from device_access import ResourceUnavailableError, create_usb_device
from device_discovery.orchestration import get_orchestrator

devices = {}
devices_lock = threading.RLock()

def on_device(result):
    observation = result.observation
    path = observation.system_path
    with devices_lock:
        device = devices.get(path)
        if device is None:
            # 任意移除通知都不用于新建可操作对象。
            if observation.action == "remove":
                return
            devices[path] = create_usb_device(result)
            return
        try:
            device.update(result)
        except ResourceUnavailableError:
            # 身份变化时旧对象已失效；按新的观察重新创建，并由用户手动连接。
            if not device.is_present:
                devices.pop(path, None)
                if observation.action != "remove":
                    devices[path] = create_usb_device(result)
            else:
                raise
        if not device.is_present:
            devices.pop(path, None)

orchestrator = get_orchestrator()
report = orchestrator.start(result_callback=on_device)
# 应用结束时先停止事件入口，再关闭已有连接。
# orchestrator.stop()
# for device in devices.values():
#     device.close()
```

父设备、接口、tty 和 net 可能逐步出现并触发多次通知。`update()` 更新同一个设备的资源，不把每次通知当作连接请求。子资源移除时，发现层可能保留被移除资源用于历史，新模块会从可操作列表排除该路径和子树。

当前事件处理沿用发现层到达顺序，不声称已经解决任意乱序、监控缺失或系统重启后的会话恢复。上层必须持续转交发现结果；不运行监听时无法承诺立即获知拔出，实际读写错误也会使当前连接断开。不要将扫描中的 `present` 当作物理重新插入。

## 用户显式选择并连接

以下仅为示例，`selected_usb_path` 和 `selected_tty_path` 来自用户选择，不能按 VID/PID 或列表第一项直接认定具体设备。`open()` 会操作串口并应用配置；`read()` 会消费串口缓冲区；`write()` 会向对端发送数据，本轮均未调用：

```python
with devices_lock:
    device = devices[selected_usb_path]
    candidates = device.serial_resources
    networks = device.network_resources
    # 展示 candidates，让用户选择其中一个 system_path。
    channel = device.serial_channel(selected_tty_path, config)

channel.open()
try:
    received = channel.read(4096)  # 超时可能返回 b""，这不表示掉线
    # 调用方决定如何显示或解码 received。
    # 用户主动输入后：written = channel.write(user_input_bytes)
finally:
    channel.close()
```

同一 USB 对象、同一资源入口复用同一个通道对象；每个通道有自己的配置和状态。修改配置需先 `close()`，再 `configure(config)`、`open()`。打开前会复核 sysfs 资源、设备 ID 及可得的 USB 枚举编号，拒绝已消失或重新枚举的旧入口；复核与打开之间仍存在系统热插拔竞争，打开或 I/O 失败会明确报告。

读写等待不持有设备生命周期锁，关闭或资源移除可以取消等待并释放连接；并发关闭期间的读取结果不作为有效新数据交付，写入异常可能已经部分发送。多个读取调用串行、多个写入调用串行，读与写可以同时进行。没有自动重发，旧 I/O 的延迟异常不会关闭后来手动建立的新连接。

`SerialChannel` 的 `ABC + abstractmethod` 阻止缺少抽象方法的实现被实例化；参数签名和实际行为仍须靠代码审查和后续验证保证。

DTR/RTS 默认请求低电平，可通过配置修改；RTS/CTS 流控下 RTS 由驱动控制。pySerial、操作系统或驱动在打开/关闭时仍可能改变控制线，不能据此保证板端完全无副作用。`exclusive=True` 只提供 pySerial/POSIX 对应的端口独占语义，不等同于未来的整板业务锁。

## 本轮证据与验证边界

2026-10-18 对树莓派执行了独立的只读 sysfs／网络状态检查：当前一个 `1d6b:0104` 设备提供 `ttyACM0`、`usb0`，接口驱动为 `cdc_acm`、`cdc_ncm`。这些是系统元数据证据，没有运行本模块业务流程，也没有证明串口收发成功。

本轮只允许语法、静态和只读状态检查。未执行发现编排、方案 CRUD、`open/read/write`、真实拔插、网络配置、SSH 板端登录或烧录。静态结果与实机通信验收分别报告。

参考：[pySerial API](https://pyserial.readthedocs.io/en/latest/pyserial_api.html)、[Python ABC](https://docs.python.org/3/library/abc.html)。

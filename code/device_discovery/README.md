# 设备发现与识别模块

当前已生成规则管理与匹配模块，以及系统设备采集模块，供后续编排层直接调用。代码未接入真实设备，功能与并发行为尚待运行验证。

## 环境与入口

- Python 3.10 或更新版本。规则模块支持 Linux/macOS；设备采集模块依赖 Linux 的 udev。
- YAML 解析使用 `PyYAML==6.0.3`；文件锁采用 `fcntl.flock`，当前不支持 Windows。
- Linux 设备枚举和热插拔监听使用 `pyudev==0.24.5`。
- 源码根目录为项目下的 `code/`。开发阶段从仓库根目录运行时，将它加入 Python 模块搜索路径，例如 `PYTHONPATH=code`。
- 业务代码从 `device_discovery.rules` 或 `device_discovery.collector` 导入，避免引用内部文件。
- 配置文件为 `code/config/application.yaml`，分别保存规则路径和采集资源子系统。
- `rules/base.py` 定义 `RuleStore` 抽象基类，`rules/models.py` 定义 `Rule` 数据结构；`rules/errors.py` 定义公开异常。
- `rules/_json_store.py`、`rules/_validation.py`、`rules/_file_lock.py` 是内部实现，下划线表示不属于稳定公开 API。
- `get_rule_store()` 是日常入口：首次调用时读取 YAML、创建并初始化仓库，之后在同一进程内返回同一个对象。
- `create_rule_store()` 用于显式创建新对象。未传路径时同样读取 YAML，但不会自动初始化；传入路径时使用指定文件。

目录结构：

```text
code/
├── config/
│   └── application.yaml       # 各模块的可读配置
├── requirements.txt           # Python 依赖
├── storage/
│   └── rules/
│       └── device-rules.json # 默认规则数据
└── device_discovery/
    ├── __init__.py
    ├── README.md
    ├── collector/
    │   ├── __init__.py        # 采集模块公开入口和默认实例
    │   ├── base.py            # DeviceCollector 抽象基类
    │   ├── models.py          # 观察结果与资源数据结构
    │   ├── errors.py          # 公开异常
    │   ├── _config.py         # 读取采集模块 YAML 配置
    │   └── _udev_collector.py # Linux udev 实现
    └── rules/
        ├── __init__.py        # 规则模块公开入口和默认实例
        ├── _config.py         # 读取规则模块的 YAML 配置
        ├── base.py            # RuleStore 抽象基类
        ├── models.py          # Rule 数据结构
        ├── errors.py          # 公开异常
        ├── _json_store.py     # JSON 存储实现
        ├── _validation.py     # 输入规范化与唯一性检查
        └── _file_lock.py      # 文件锁实现
```

### 拆分理由与边界

原先单个 `rules.py` 已经可以工作，但公开接口、存储实现和文件细节混在一起。当前拆分让编排层只依赖 `RuleStore` 抽象基类及公开入口，也让内部代码可以分别阅读和修改。它提高的是职责清晰度与后续替换实现的便利性，不改变文件读写次数，也不带来性能提升。

当前只拆到抽象基类、数据结构、异常和有明确职责的内部文件。默认实例使用普通模块缓存和线程锁，没有引入 Singleton 元类、依赖注入容器、插件加载器或多个存储后端。判断边界是否有效的标准是：后续编排层仅从 `device_discovery.rules` 导入，内部实现调整时无需修改编排层。

默认实例只保证同一个 Python 进程内复用。多个进程会拥有各自的对象，并通过现有文件锁安全访问同一份规则文件。YAML 在第一次调用 `get_rule_store()` 时读取，之后修改配置不会替换已经创建的默认对象，需要重启进程后重新加载。

安装依赖：

```bash
python3 -m pip install -r code/requirements.txt
```

## 系统设备采集模块

编排层从 `device_discovery.collector` 导入公开接口：

```python
from device_discovery.collector import get_device_collector

collector = get_device_collector()

# 启动时建议先监听，再扫描；编排层处理两者之间可能出现的重复观察。
collector.start(handle_device_event)
current_devices = collector.scan()

# 服务退出时停止后台监听。
collector.stop()
```

`scan()` 返回当前每个外接 USB 设备的一条 `DeviceObservation`，并把 USB 接口、串口和网络资源归入 `resources`。扫描结果使用 `source=scan`、`action=present`，根集线器不作为外接设备返回。

`start(callback)` 使用 `pyudev.MonitorObserver` 在后台线程监听。USB 父设备以及配置子系统中的事件会转换成相同的 `DeviceObservation`；`event_subsystem`、`event_system_path` 和 `event_sequence_number` 标明本次事件的触发对象。回调也运行在监听线程中，应尽快把结果交给编排层。

当前配置归并 `tty` 和 `net` 子系统，USB interface 始终归并，以保留 `ch341`、`cdc_acm`、`cdc_ncm` 等接口驱动信息：

```yaml
device_discovery:
  collector:
    resource_subsystems:
      - tty
      - net
```

采集模块只读取 Linux 已经公开的设备元数据。它不读取关注规则、不判断 CH340 或板卡类型、不打开 `/dev/ttyUSB*`、不设置波特率、不登录网络接口，也不保存历史。VID/PID 缺失时保留为 `None`，由编排层决定是否交给规则模块。

业务代码使用 `get_device_collector()` 获得进程内共享实例，避免同一进程重复启动监听器；独立场景可使用 `create_device_collector()`。同一个采集器重复 `start()` 会抛出 `CollectorStateError`，`stop()` 可重复调用。

监听没有设置单一 udev 子系统过滤器，因为一个物理 USB 设备可能分别产生 `usb`、`tty` 和 `net` 事件。内部回调按配置和 USB 父子关系筛选；若以后确认事件量造成实际负担，再评估多个带内核过滤的监听器。

## 调用示例

下列示例供阅读，尚未执行；手动运行时会创建并修改指定的本地规则文件。

```python
from device_discovery.rules import get_rule_store

store = get_rule_store()

# 示例新增仅执行一次；同一个 Key 或 VID/PID 再次新增会报冲突。
store.add("CH340", "1a86", "7523")
print(store.list_rules())
print(store.match("0x1A86", "7523"))  # 返回 "CH340"
print(store.match("1234", "5678"))    # 无匹配时返回 None

# 修改示例：保持编号，只修改标签。
store.update("CH340", "1a86", "7523", new_key="调试串口")
store.delete("调试串口")
```

规则文件路径在 YAML 中配置：

```yaml
device_discovery:
  rules:
    storage_file: ../storage/rules/device-rules.json
```

相对路径以 `application.yaml` 所在目录为基准。上例最终指向 `code/storage/rules/device-rules.json`。

需要独立实例时，可以显式创建并初始化：

```python
from device_discovery.rules import create_rule_store

store = create_rule_store("/tmp/another-rules.json")
store.initialize()
```

VID/PID 接收 1 至 4 位十六进制字符串，允许 `0x` 前缀、大小写与首尾空格，统一为四位小写字符串。Key 去掉首尾空格后不能为空，大小写敏感。设备类型标签只匹配配置编号，不证明后面的开发板已经连接或串口可用。

## 接口约定

业务层只依赖以下公开名称：

```python
from device_discovery.rules import (
    Rule,
    RuleConfigurationError,
    RuleError,
    RuleStore,
    create_rule_store,
    get_rule_store,
)
```

`RuleStore` 是编排层依赖的抽象基类；内部的 `JsonFileRuleStore(RuleStore)` 明确继承并实现全部抽象方法。漏掉任何抽象方法时，Python 会阻止实现类实例化。`get_rule_store()` 提供默认共享对象，`create_rule_store()` 提供显式创建能力，两者都隐藏具体文件实现。

| 接口 | 成功结果 |
|---|---|
| `get_rule_store()` | 首次创建并初始化，之后返回进程内同一对象 |
| `create_rule_store(path=None)` | 创建新对象；路径为空时读取环境配置 |
| `initialize()` | 返回 `None`，已有文件保持内容 |
| `list_rules()` | 本次从文件读取的 `list[Rule]` 快照 |
| `match(vid, pid)` | 匹配的 Key；未匹配返回 `None` |
| `add(key, vid, pid)` | 已保存的 `Rule` |
| `update(key, vid, pid, new_key=...)` | 已保存的新 `Rule`，不传 `new_key` 时保留原 Key |
| `delete(key)` | 已删除的 `Rule` |

失败抛出 `RuleError` 的具体子类，调用者可统一捕获基类，也可以区分：

- `InvalidRuleError`：输入参数无效。
- `RuleConfigurationError`：YAML 缺失、无法解析或字段不符合要求。
- `RuleConflictError`：Key 或完整 VID/PID 二元组重复。
- `RuleNotFoundError`：修改或删除的 Key 不存在。
- `RuleFileError`：文件或锁操作失败，包括未初始化、权限、写入失败。
- `RuleFileFormatError`：JSON 损坏、字段不合规、已有文件违反唯一性约束；属于 `RuleFileError`。

未匹配和文件读取失败有不同表现，编排层应分别处理。规则保存接口成功后，编排层才重新扫描设备；本模块不负责扫描。

## 文件和锁

- 规则文件是唯一依据，每次调用重新读取，不长期缓存规则。
- 读取完整内容时持共享锁；修改的读、校验、写入全过程持独占锁。
- 锁文件位于规则文件旁，例如 `device-rules.json.lock`，所有访问者须遵守本模块的锁约定；运行期间不要删除或替换锁文件。
- 写入同目录临时文件，完成序列化、flush 和文件 fsync 后，用 `os.replace` 替换规则文件，成功后返回。
- 新生成的规则和锁文件使用仅当前用户可读写的权限。跨系统用户共享文件未纳入本模块。
- 文件已有但损坏时保留原内容并报错；普通操作发现文件丢失时报错，需要显式初始化才能重新创建。
- 当前保存成功表示文件替换完成；未同步父目录，不承诺突然断电后的完整持久性。异常退出可能留下临时文件，当前不提供自动清理恢复流程。
- 当前面向本机文件系统，锁等待采用阻塞方式，超时机制与网络文件系统适配暂未实现。

## 当前验证状态

仅安排不执行业务调用的语法与静态检查。尚未运行规则增删改、损坏文件恢复场景、多进程锁竞争、真实 USB 扫描或热插拔监听；未操作串口、USB 或板卡。

设计依据见 [设备发现与模块设计](../../DOC/自动发现与识别接入设备-实现逻辑.md)。

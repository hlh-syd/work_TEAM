# Python 环境与依赖管理规范

## 单环境策略

本项目统一使用独立 Python 3.12 环境作为唯一 Python 运行环境：

- Python: 3.12.4
- 解释器: `C:\Users\he\AppData\Local\Programs\Python\Python312\python.exe`
- pip: `C:\Users\he\AppData\Local\Programs\Python\Python312\Lib\site-packages\pip`

开发、测试和部署均应使用上述同一套环境，避免同时混用 Anaconda base、其他 Conda 环境、系统 Python 或虚拟环境。

## 启动环境

在新的 PowerShell 终端中进入项目根目录后执行：

```powershell
cd D:\work-课题
& 'C:\Users\he\AppData\Local\Programs\Python\Python312\python.exe' --version
& 'C:\Users\he\AppData\Local\Programs\Python\Python312\python.exe' -m pip --version
```

确认 Python 输出为 `Python 3.12.4`，且 pip 路径位于 `C:\Users\he\AppData\Local\Programs\Python\Python312`。

## 依赖版本控制

项目根目录的 `requirements.txt` 是唯一的 pip 依赖版本锁定文件。安装依赖时使用：

```powershell
& 'C:\Users\he\AppData\Local\Programs\Python\Python312\python.exe' -m pip install -r requirements.txt
```

每次通过 pip 安装、升级或移除依赖后，必须立即在项目根目录执行：

```powershell
& 'C:\Users\he\AppData\Local\Programs\Python\Python312\python.exe' -m pip freeze > requirements.txt
```

然后将更新后的 `requirements.txt` 一并提交，确保团队成员和部署环境使用完全一致的依赖版本。

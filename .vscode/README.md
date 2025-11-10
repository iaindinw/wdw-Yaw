# VS Code Configuration

This folder contains VS Code workspace settings.

## Settings

- **Python Interpreter**: Points to `static_yaw_challenge/venv/Scripts/python.exe`
- **Jupyter Notebook Root**: Set to `static_yaw_challenge` directory
- **Terminal Activation**: Automatically activates the venv when opening terminal

## Important Notes

### Windows Long Path Issue
This project uses the classic Jupyter `notebook` package instead of JupyterLab to avoid Windows Long Path limitations. The venv in `static_yaw_challenge/venv/` has been configured to work around this issue.

### If VS Code Can't Find the Kernel

1. Press `Ctrl+Shift+P`
2. Type: `Python: Select Interpreter`
3. Select the interpreter at: `static_yaw_challenge/venv/Scripts/python.exe`
4. In your notebook, click the kernel selector (top-right)
5. Choose the Python interpreter from `static_yaw_challenge/venv`

### Reload Window
After changing settings, you may need to reload VS Code:
- Press `Ctrl+Shift+P`
- Type: `Developer: Reload Window`

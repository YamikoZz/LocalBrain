import ctypes
import os
import platform
import shutil


def inspect():
    result = {"python": platform.python_version(), "cpu": platform.processor(),
              "logical_cpus": os.cpu_count(), "docker": shutil.which("docker")}
    if os.name == "nt":
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total_physical", ctypes.c_ulonglong),
                        ("available_physical", ctypes.c_ulonglong),
                        ("total_page", ctypes.c_ulonglong),
                        ("available_page", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong),
                        ("available_virtual", ctypes.c_ulonglong),
                        ("available_extended", ctypes.c_ulonglong)]
        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            result["ram_gib"] = round(memory.total_physical / 2**30, 2)
            result["available_ram_gib"] = round(memory.available_physical / 2**30, 2)
    try:
        import torch
        result["torch"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            free, total = torch.cuda.mem_get_info()
            result.update(gpu=torch.cuda.get_device_name(),
                          vram_gib=round(total / 2**30, 2),
                          free_vram_gib=round(free / 2**30, 2),
                          compute_capability=torch.cuda.get_device_capability())
            # Actual operation verifies that the installed wheel supports the GPU.
            x = torch.randn(32, 32, device="cuda")
            result["cuda_matmul_ok"] = bool(torch.isfinite(x @ x).all().item())
    except ImportError:
        result["torch"] = "not installed"
    return result


def be_polite():
    """Run at below-normal priority so games and apps the user is running stay smooth."""
    try:
        if os.name == "nt":
            kernel = ctypes.windll.kernel32
            kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
        else:
            os.nice(10)
    except (OSError, AttributeError):
        pass

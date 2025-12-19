# ============================================================
# ANSYS Mechanical (IronPython): available/free system RAM
# Tries (in order):
#   1) Microsoft.VisualBasic.Devices.ComputerInfo (Windows, bytes)
#   2) System.Diagnostics.PerformanceCounter (Windows, MB)
#   3) WMI Win32_OperatingSystem (Windows, KB)
#   4) /proc/meminfo MemAvailable (Linux, kB)
# ============================================================

import clr

def _bytes_to_gib(n_bytes):
    return n_bytes / float(1024 ** 3)

def get_available_ram_bytes():
    # --- (1) Microsoft.VisualBasic.Devices.ComputerInfo (Windows) ---
    try:
        clr.AddReference("Microsoft.VisualBasic")
        from Microsoft.VisualBasic.Devices import ComputerInfo
        return int(ComputerInfo().AvailablePhysicalMemory)  # bytes
    except Exception:
        pass

    # --- (2) PerformanceCounter: "Memory", "Available MBytes" (Windows) ---
    try:
        clr.AddReference("System")
        from System.Diagnostics import PerformanceCounter
        pc = PerformanceCounter("Memory", "Available MBytes")
        mb = float(pc.NextValue())
        return int(mb * 1024 * 1024)  # bytes
    except Exception:
        pass

    # --- (3) WMI Win32_OperatingSystem.FreePhysicalMemory (Windows) ---
    try:
        clr.AddReference("System.Management")
        from System.Management import ManagementObjectSearcher

        # FreePhysicalMemory is in KB on Win32_OperatingSystem
        searcher = ManagementObjectSearcher(
            "SELECT FreePhysicalMemory FROM Win32_OperatingSystem"
        )
        for mo in searcher.Get():
            free_kb = int(mo["FreePhysicalMemory"])
            return free_kb * 1024  # bytes
    except Exception:
        pass

    # --- (4) Linux: /proc/meminfo MemAvailable (kB) ---
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb * 1024  # bytes
    except Exception:
        pass

    raise RuntimeError("Unable to determine available system RAM on this platform/environment.")

def get_total_ram_bytes_if_possible():
    # Optional helper (mainly Windows/.NET):
    try:
        clr.AddReference("Microsoft.VisualBasic")
        from Microsoft.VisualBasic.Devices import ComputerInfo
        return int(ComputerInfo().TotalPhysicalMemory)  # bytes
    except Exception:
        return None

# ---- Example usage in Mechanical scripting console ----
try:
    avail_b = get_available_ram_bytes()
    total_b = get_total_ram_bytes_if_possible()

    if total_b is not None:
        print("Available RAM: %.2f GiB / Total RAM: %.2f GiB" % (_bytes_to_gib(avail_b), _bytes_to_gib(total_b)))
    else:
        print("Available RAM: %.2f GiB" % _bytes_to_gib(avail_b))

except Exception as e:
    print("RAM query failed: %s" % e)

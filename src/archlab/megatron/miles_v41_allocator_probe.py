"""Qualify discard/reallocation without any host backup in the pinned runtime."""
import json
import os
import sys


def main():
    from torch_memory_saver.utils import get_binary_path_from_package
    if "torch_memory_saver" not in os.environ.get("LD_PRELOAD", ""):
        os.environ.update(TMS_INIT_ENABLE="0", TMS_INIT_ENABLE_CPU_BACKUP="0",
                          TMS_INIT_ENABLE_DISK_BACKUP="0")
        os.environ["LD_PRELOAD"] = str(get_binary_path_from_package("torch_memory_saver_hook_mode_preload"))
        os.execv(sys.executable, [sys.executable, "-m", __spec__.name])
    import torch
    from torch_memory_saver import torch_memory_saver as saver
    torch.cuda.init()
    with saver.region(tag="resident_probe", enable_cpu_backup=False, enable_disk_backup=False):
        binary = saver._impl._binary_wrapper.cdll
        assert not binary.tms_get_enable_cpu_backup()
        assert not binary.tms_get_enable_disk_backup()
        value = torch.ones(128 * 1024 * 1024, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    active = torch.cuda.mem_get_info()[0]
    saver.pause(tag="resident_probe")
    paused = torch.cuda.mem_get_info()[0]
    assert paused - active > 400 * 1024**2
    saver.resume(tag="resident_probe")
    value.fill_(7)
    assert value[0].item() == 7
    print(json.dumps(dict(allocator_discard_passed=True, reclaimed_bytes=paused-active,
                          cpu_backup=False, disk_backup=False)), flush=True)


if __name__ == "__main__":
    main()

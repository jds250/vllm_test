import pynvml
import time
import multiprocessing as mp
from typing import List, Tuple

class PowerMonitor:
    def __init__(self, gpu_index: int = 0, sample_interval: float = 0.1):
        """初始化功率监测器。

        Args:
            gpu_index (int): 要监测的 GPU 索引。
            sample_interval (float): 采样间隔时间，单位为秒。
        """
        self.gpu_index = gpu_index
        self.sample_interval = sample_interval
        self.power_data: List[Tuple[float, float]] = []

        # 初始化 NVML
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    def start_monitoring(self, duration: float):
        """启动功率监测器，记录指定时间段的功率数据。

        Args:
            duration (float): 监测的总时长，单位为秒。
        """
        self.start_time = time.time()
        while time.time() - self.start_time < duration:
            self._collect_power_data()
        self.end_time = time.time()

    def _collect_power_data(self):
        """采集单次功率数据。"""
        power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        current_time = time.time()
        self.power_data.append((current_time, power))
        time.sleep(self.sample_interval)

    def calculate_energy(self, start_offset: float, end_offset: float) -> float:
        """计算指定时间段的能量消耗。

        Args:
            start_offset (float): 相对于监测起始时间的偏移量，单位为秒。
            end_offset (float): 相对于监测起始时间的偏移量，单位为秒。

        Returns:
            float: 指定时间段的能量消耗（焦耳）。
        """
        start_time = self.start_time + start_offset
        end_time = self.start_time + end_offset

        interval_data = [
            (t, p) for (t, p) in self.power_data if start_time <= t <= end_time
        ]

        if len(interval_data) < 2:
            print("数据不足，无法进行能耗计算。")
            return 0.0

        energy = 0.0
        for i in range(1, len(interval_data)):
            t1, p1 = interval_data[i - 1]
            t2, p2 = interval_data[i]
            energy += ((p1 + p2) / 2) * (t2 - t1)

        return energy

    def print_power_data(self):
        """打印采集的功率数据。"""
        for timestamp, power in self.power_data:
            print(f"Time: {timestamp - self.start_time:.2f}s, Power: {power} W")

    def __del__(self):
        pynvml.nvmlShutdown()


def monitor_power_in_background(gpu_index: int, sample_interval: float, duration: float, result_queue: mp.Queue):
    """作为子进程运行的功率监测任务。"""
    monitor = PowerMonitor(gpu_index=gpu_index, sample_interval=sample_interval)
    monitor.start_monitoring(duration)
    result_queue.put(monitor.power_data)  # 将采集的功率数据放入队列中
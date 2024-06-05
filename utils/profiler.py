import logging
import socket
from datetime import datetime, timedelta
import torch

from torch.autograd.profiler import record_function
from torchvision import models

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
def trace_handler(prof: torch.profiler.profile):
   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"

   # Construct the trace file.
   prof.export_chrome_trace(f"{file_prefix}.json.gz")

   # Construct the memory timeline file.
   prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")
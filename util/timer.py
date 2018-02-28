import time
import sys

isPython3 = sys.version_info >= (3, 0)
timer = time.clock

if isPython3:
  timer = time.perf_counter


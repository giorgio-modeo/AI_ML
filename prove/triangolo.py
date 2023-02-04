import _thread as th
import time as t
def print_time(threadName):
    n=0
    while True:
        n+=1
        print(f"\npdio {n} {threadName}")
# Create two threads as follows
th.start_new_thread( print_time, ("Thread-1",) )
th.start_new_thread( print_time, ("Thread-2",) )

while 1:
   pass
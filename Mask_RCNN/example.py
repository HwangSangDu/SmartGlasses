# from Queue import Queue
from threading import Thread
# from multiprocessing import Queue
# from asyncio import Queue
import queue as Queue

class Worker(Thread):
    """Thread executing tasks from a given tasks queue"""
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()

    def run(self):
        while True:
            func, args, kargs = self.tasks.get()
            try:
                func(*args, **kargs)
            except Exception as e:
                print (e)
            # finally:
                # self.tasks.task_done()


class ThreadPool:
    """Pool of threads consuming tasks from a queue"""
    def __init__(self, num_threads):
        self.tasks = Queue.Queue(num_threads)
        for _ in range(num_threads):
            Worker(self.tasks)

    def add_task(self, func, *args, **kargs):
        """Add a task to the queue"""
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        """Wait for completion of all the tasks in the queue"""
        self.tasks.join()

if __name__ == '__main__':
    from random import randrange
    from time import sleep

    delays = [randrange(1, 10) for i in range(100)]
    print(delays)

    def wait_delay(d):
        print ('sleeping for (%d)sec' % d)
        sleep(d)

    pool = ThreadPool(20)

    for i, d in enumerate(delays):
        pool.add_task(wait_delay, d)

    pool.wait_completion()


















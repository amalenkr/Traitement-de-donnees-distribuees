import time
import multiprocessing as mp
from multiprocessing import Process
from threading import Thread
cpus = mp.cpu_count()

def sleep_fun(seconds):
    print("Sleeping for {} second(s)".format(seconds))
    time.sleep(seconds)
    
def main():
    sleep_times = [1,2,3]
    start = time.time()
    for i in sleep_times:
        sleep_fun(i)
    end = time.time()
    print("Series computation: {} secs\n".format(end - start))
    
    start = time.time()
    threads = []
    for i in sleep_times:
        t = Thread(target=sleep_fun, args=(i,))
        threads.append(t)
        t.start()
      
    for t in threads: t.join()
    end = time.time()
    print("Multithreading computation: {} secs\n".format(end - start))


    start = time.time()
    with mp.Pool(cpus) as p:
        p.map(sleep_fun, sleep_times)
        p.close()
        p.join()
    end = time.time()
    print("Multiprocessing computation (with Pool): {} secs\n".format(end - start))

    start = time.time()
    processes = []
    for i in sleep_times:
        p = Process(target=sleep_fun, args=(i,))
        processes.append(p)
        p.start()
      
    for p in processes: p.join()
    end = time.time()
    print("Multiprocessing computation (with Process): {} secs\n".format(end - start))

if __name__ == '__main__':
    # Better protect your main function when you use multiprocessing
    main()
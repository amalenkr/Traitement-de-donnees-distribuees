import time
import random
from functools import reduce
import multiprocessing as mp
from multiprocessing import Process
from threading import Thread

cpus = mp.cpu_count()

def calculation(number):
    random_list = random.sample(range(10000000), number)
    return reduce(lambda x, y: x*y, random_list)
    
def main():
    numbers = [200000, 200000, 200000]
    start = time.time()
    for i in numbers:
        calculation(i)
    end = time.time()
    print("Series: {} secs\n".format(end - start))
    
    start = time.time()
    threads = []
    for i in numbers:
        t = Thread(target=calculation, args=(i,))
        threads.append(t)
        t.start()
      
    for t in threads: t.join()
    end = time.time()
    print("Multithreading comp: {} secs\n".format(end - start))


    start = time.time()
    with mp.Pool(cpus) as p:
        p.map(calculation, numbers)
        p.close()
        p.join()
    end = time.time()
    print("Multiprocessing computation (with Pool): {} secs\n".format(end - start))

    start = time.time()
    processes = []
    for i in numbers:
        p = Process(target=calculation, args=(i,))
        processes.append(p)
        p.start()
      
    for p in processes: p.join()
    end = time.time()
    print("Multiprocessing computation (with Process): {} secs\n".format(end - start))

if __name__ == '__main__':
    # Better protect your main function when you use multiprocessing
    main()
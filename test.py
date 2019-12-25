import multiprocessing
import time

def f(i,l):
    #time.sleep(2)
    l.append(i)

#p = multiprocessing.Pool(4)
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    l = multiprocessing.Manager().list()
    a = []
    for _ in range(4):
        with multiprocessing.Pool(4) as p:

            for i in range(40):
                p.apply_async(f, args=(i,l))
            p.close()
            p.join()
        print(len(l))

    print(a)
    print(l)
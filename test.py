import multiprocessing

def f(i,l):
    if i%4 not in l:
        l.append(i%4)

p = multiprocessing.Pool(4)
l = multiprocessing.Manager().list()

for i in range(20):
    p.apply_async(f, args=(i,l))
p.close()
p.join()

print(l)
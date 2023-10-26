def done(n):
    return n >= 4


n = 0
while not done(n):
    print(n)
    n += 1


from itertools import count as nonnegative_integers
for n in nonnegative_integers():
    if done(n):
        break
    print(n)

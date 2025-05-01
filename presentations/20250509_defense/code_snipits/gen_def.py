

def example():
    yield 1
    yield 2
    yield 3

gen = example()
print(type(gen))  # <class 'generator'>
print(next(gen))  # 1
print(next(gen))  # 2
print(next(gen))  # 3
print(next(gen))  # StopIteration exception

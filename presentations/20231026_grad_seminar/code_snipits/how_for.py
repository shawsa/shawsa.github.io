

# What you write
for n in range(3):
    #  loop body
    print(n)


# What happens behind the scenes
generator_object = range(3).__iter__()
while True:
    try:
        n = generator_object.__next__()
    except StopIteration:
        break
    #  loop body
    print(n)

# t = 32.00
# print([round((x - 32) * 5/9) for x in t])

# l = [1, 2, 3, 4, 5]
# print([x&1 for x in l])

# def foo(i, x = []):
#     x.append(i)
#     return x
# for i in range(3):
#     print(foo(i))

# def a(n):
#     if n == 0:
#         return 0
#     elif n == 1:
#         return 1
#     else:
#         return a(n - 1) + a(n - 2)
# for i in range(0, 4):
#     print(a(i), end="")

print(list(map((lambda x:x^2), range(10))))
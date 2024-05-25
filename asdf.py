# 2626
# n = int(input())
# t = 0
#
# for i in range(n // 3 + (n % 3 != 0), n // 2 + n % 2):
#     t += i - ((n - i) // 2 + (n - i) % 2) + 1
#
# print(t)

# 2632
n = int(input())
def Climbing(n):
    if n == 1 or n == 0:
        return 1
    elif n > 1:
        return Climbing(n - 1) + Climbing(n - 2)

print(Climbing(n))
## https://www.hackerrank.com/challenges/matrix-script/problem ##
## Matrix Script ##

import re

n, m = map(int, input().split())
matrix = [input() for _ in range(n)]

s = ''.join(matrix[i][j] for j in range(m) for i in range(n))
decoded = re.sub(r'(?<=[a-zA-Z0-9])[\W_]+(?=[a-zA-Z0-9])', ' ', s)

print(decoded)

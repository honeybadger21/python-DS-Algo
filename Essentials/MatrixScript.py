## https://www.hackerrank.com/challenges/matrix-script/problem ##
## Matrix Script ##

# Neo has a complex matrix script. The matrix script is a X grid of strings. It consists of alphanumeric characters, spaces and symbols (!,@,#,$,%,&).
# To decode the script, Neo needs to read each column and select only the alphanumeric characters and connect them. Neo reads the column from top to bottom and 
# starts reading from the leftmost column.
# If there are symbols or spaces between two alphanumeric characters of the decoded script, then Neo replaces them with a single space '' for better readability.
# Neo feels that there is no need to use 'if' conditions for decoding.
# Alphanumeric characters consist of: [A-Z, a-z, and 0-9].

import re

n, m = map(int, input().split())
matrix = [input() for _ in range(n)]

s = ''.join(matrix[i][j] for j in range(m) for i in range(n))
decoded = re.sub(r'(?<=[a-zA-Z0-9])[\W_]+(?=[a-zA-Z0-9])', ' ', s)

print(decoded)

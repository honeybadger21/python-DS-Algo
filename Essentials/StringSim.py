## https://www.hackerrank.com/challenges/string-similarity/problem ##
## HackerRank String Similarity Problem ## 

# For two strings A and B, we define the similarity of the strings to be the length of the longest prefix common to both strings. 
# For example, the similarity of strings "abc" and "abd" is 2, while the similarity of strings "aaa" and "aaab" is 3.
# Calculate the sum of similarities of a string S with each of it's suffixes.

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'stringSimilarity' function below.
#
# The function is expected to return an INTEGER.
# The function accepts STRING s as parameter.
#

def stringSimilarity(s):
    # Complete this function
    result = length = len(s)
    right = 0
    left = 0
    z = [length]
  
    for i in range(1, length):
        z.append(0)
        if i <= right:
            z[i] = min(right - i + 1, z[i - left])
        while i + z[i] < length and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] - 1 > right:
            left = i
            right = i + z[i] - 1
        
        result += z[i]
    
    return result

'''
def stringSimilarity(s):
    # Write your code here
    
    ans=0
    for idx in range(len(s)):
        suf=s[idx:]
        count=0
        for idx in range(len(suf)):
            if suf[idx]==s[idx]:
                count+=1
            else:
                break
        ans+=count
    return ans    
'''

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input().strip())

    for t_itr in range(t):
        s = input()

        result = stringSimilarity(s)

        fptr.write(str(result) + '\n')

    fptr.close()

# Recursion Approach to Fibonacci

'''
Recursion is the process in which a function calls itself until the base cases are reached. 
And during the process, complex situations will be traced recursively and become simpler and simpler. 
The whole structure of the process is tree like. 
Recursion does not store any value until reach to the final stage(base case).
'''

class Solution(object):
  
  def fib(self, n):
    """
    :type n: int
    :rtype: int
    """
    
    # Base Case
    if n == 0:
      return 0
    if n == 1:
      return 1
    
    # Pattern --> F(i) = F(i-1) + F(i-2)
    return self.fib(n-1) + self.fib(n-2)
  
# Dynamic Programming Approach to Fibonacci 
  
'''
Dynamic Programming is mainly an optimization compared to simple recursion. 
The main idea is to decompose the original question into repeatable patterns and then store the results as many sub-answers. 
Therefore, we do not have to re-compute the pre-step answers when needed later. 
In terms of big O, this optimization method generally reduces time complexities from exponential to polynomial. 
'''

class Solution(object):
  def fib(self, n):
    """
    :type n: int
    :rtype: int
    """
    
    # Base Case
    if n == 0:
      return 0
    if n == 1:
      return 1
    
    # Creating an empty DP array
    dp = [0] * (n+1)
    
    # Find Patterns 
    dp[0] = 0
    dp[1] = 0
    
    for i in range(2, n+1):
      dp[i] = dp[i-1] + dp[i-2] 
      
     return dp[n]

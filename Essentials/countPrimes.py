# Given an integer n, return the number of prime numbers that are strictly less than n. 
# Sieve of Eratosthenes

class Solution:
    def countPrimes(self, n: int) -> int:
        prime = [True for i in range(n+1)]
        p = 2
        while (p*p<=n):
            if prime[p]==True:
                for i in range(p*p, n+1, p):
                    prime[i]=False
            p+=1
        count=0
        for idx in range(2, len(prime)-1):
            if prime[idx]==True:
                count+=1
        return count


        '''
        if n==0 or n==1:
            return 0
        count=0
        for x in range(2, n):
            flag = False
            for y in range(2, int(x**(0.5))+1):
                if x%y==0:
                    flag = True
                    break
            if flag == False: 
                count+=1
        return count
        '''

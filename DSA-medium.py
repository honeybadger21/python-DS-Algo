# Leetcode - Top Interview Questions Track
# Difficulty: Medium 

# Accepted Solutions Archive
# Created by Ruchi Sharma [UT Austin '23, IIT Roorkee '21]

# Note: Work in progress, expect regular updates. 

# 15. 3Sum
# Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.
# Notice that the solution set must not contain duplicate triplets.

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:   
        nums, arr = sorted(nums), []
        for i in range(len(nums)):
            if i>0 and nums[i-1]==nums[i]:
                continue                 
            j, k = i+1, len(nums)-1            
            while j<k:
                s = nums[i] + nums[j] + nums[k]                
                if s > 0:
                    k-=1                    
                elif s < 0:
                    j+=1                    
                else:
                    arr.append([nums[i], nums[j], nums[k]])
                    j+=1
                    while nums[j-1] == nums[j] and j<k:
                        j+=1                        
        return arr

# 33. Search in Rotated Sorted Array
# There is an integer array nums sorted in ascending order (with distinct values).
# Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].
# Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.
# You must write an algorithm with O(log n) runtime complexity.

class Solution:
    def search(self, nums: List[int], target: int) -> int:        
        ## hai toh seedha binary search ka classic question yeh        
        start, end = 0, len(nums)-1        
        while start<=end:
            mid = (start+end)>>1            
            if nums[mid] == target:
                return mid            
            elif nums[mid]>=nums[start]:
                if (target >= nums[start] and target < nums[mid]):
                    end = mid-1
                else:
                    start = mid+1                    
            else: 
                if (target<=nums[end] and target>nums[mid]):
                    start = mid+1
                else:
                    end = mid-1                    
        return -1
   
# 46. Permutations
# Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:        
        n, ans = len(nums), []        
        def swap(arr, j):
            if j == n:
                ans.append(arr[:])                
            for i in range(j, n):
                arr[i], arr[j] = arr[j], arr[i]
                swap(arr, j+1)
                arr[i], arr[j] = arr[j], arr[i]                
        swap(nums, 0)
        return ans 

# 49. Group Anagrams 
# Given an array of strings strs, group the anagrams together. You can return the answer in any order. An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:        
        hsh = {}        
        for elem in strs:
            temp = "".join(sorted(elem))
            if temp not in hsh:
                hsh[temp] = [elem]            
            elif temp in hsh:
                hsh[temp].append(elem)                
        ans = []        
        for elem in hsh:
            ans.append(hsh[elem])            
        return ans
        
# 53. Maximum Subarray
# Given an integer array nums, find the subarray with the largest sum, and return its sum.
    
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        
        # Kadane's Algorithm         
        Max = nums[0]
        Sum = 0        
        for num in nums:
            Sum += num
            Max = max(Max, Sum)
            if Sum<0:
                Sum = 0
        return Max
       
        # Brute Force
        '''
        def list_sum(arr):
            ans = 0
            for elem in arr:
                ans+=elem
            return ans        
        ans1 = list_sum(nums)
        k = len(nums)                
        for i in range(k, -1, -1):
            for j in range(0, i):
                ans1 = max(ans1, list_sum(nums[j:i]))
        return ans1
        '''      
    
# 73. Set Matrix Zeroes 
# Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.
# You must do it in place.
        
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        
        imat, jmat = [], []
        
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    imat.append(i)
                    jmat.append(j)
                    
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if i in imat or j in jmat:
                    matrix[i][j] = 0
                    
# 75. Sort Colors
# Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.
# We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.
# You must solve this problem without using the library's sort function.

class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        for i in range(1, len(nums)):            
            j, key = i-1, nums[i]            
            while j >= 0 and key < nums[j]:
                nums[j+1] = nums[j]
                j -= 1                
            nums[j+1] = key
            
# Subsets 
# Given an integer array nums of unique elements, return all possible subsets (the power set). The solution set must not contain duplicate subsets. Return the solution in any order.
   
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:    
    # Cascading     
        n = len(nums)
        output = [[]]        
        for num in nums:
            output += [curr + [num] for curr in output]            
        return output 
    
# 215. Kth Largest Element in an Array     
# Given an integer array nums and an integer k, return the kth largest element in the array.
# Note that it is the kth largest element in the sorted order, not the kth distinct element.
# You must solve it in O(n) time complexity.

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        nums = sorted(nums)
        return nums[len(nums)-k]
        
# 238. Product of Array Except Self
# Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].
# The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
# You must write an algorithm that runs in O(n) time and without using the division operation.

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:        
        prefix, n, answer, postfix = 1, len(nums), [1], 1            
        for i in range(1, n):
            prefix *= nums[i-1]
            answer.append(prefix)        
        for i in range(n-2, -1, -1):
            postfix *= nums[i+1]
            answer[i] *= postfix            
        return answer 

# 287. Find the Duplicate Number 
# Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.
# There is only one repeated number in nums, return this repeated number.
# You must solve the problem without modifying the array nums and uses only constant extra space.

class Solution:
    def findDuplicate(self, nums: List[int]) -> int:        
        hsh = {}        
        for elem in nums:
            if elem not in hsh: 
                hsh[elem]=1
            elif elem in hsh:
                hsh[elem]+=1                
        for elem in hsh:
            if hsh[elem]>1:
                return elem

# 347. Top K Frequent Elements 
# Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:        
        hsh = {}        
        for elem in nums:
            if elem not in hsh:
                hsh[elem] = 1
            elif elem in hsh:
                hsh[elem] += 1     
        final = [m for m, v in sorted(hsh.items(), key=lambda item: item[1], reverse = True)]        
        return final[:k]

# 378. Kth Smallest Element in a Sorted Matrix
# Given an n x n matrix where each of the rows and columns is sorted in ascending order, return the kth smallest element in the matrix.
# Note that it is the kth smallest element in the sorted order, not the kth distinct element.
# You must find a solution with a memory complexity better than O(n2).
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:        
        arr = []        
        for elem in matrix:
            for elem2 in elem:
                arr.append(elem2)                
        arr = sorted(arr)        
        return arr[k-1]
        
# 454. 4SumII
# Given four integer arrays nums1, nums2, nums3, and nums4 all of length n, return the number of tuples (i, j, k, l) such that:
    # 0 <= i, j, k, l < n
    # nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0
    
class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:        
        hsh, count = {}, 0        
        for i in range(len(nums1)):
            for j in range(len(nums2)):
                temp = nums1[i]+nums2[j]
                if temp not in hsh:
                    hsh[temp]=1
                else:
                    hsh[temp]+=1                
        for k in range(len(nums1)):
            for l in range(len(nums1)):
                temp = -nums3[k]-nums4[l]
                if temp in hsh:
                    count+=hsh[temp]                
        return count
                
# 02. Add Two Numbers 
# You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list. You may assume the two numbers do not contain any leading zero, except the number 0 itself.

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        res = temp = ListNode(0)
        carry = 0
        while l1 or l2 or carry:
            if l1:
                carry += l1.val
                l1 = l1.next
            # note: we're not using elif here, both conditions if true need to be executed
            if l2: 
                carry += l2.val
                l2 = l2.next
            carry, val = divmod(carry, 10)
            temp.next = temp = ListNode(val)        
        return res.next
    
# 05. Longest Palindromic Substring
# Given a string s, return the longest palindromic substring in s.

class Solution:
    def longestPalindrome(self, s: str) -> str:

        # Brute Force
        all = []
        for i in range(len(s)):
            for j in range(i, len(s)):
                # [::-1] is for reversing
                if s[i:j+1] == s[i:j+1][::-1]:
                    all.append(s[i:j+1])
        max_len = 0
        for elem in all:
            max_len = max(max_len, len(elem))
        for elem in all:
            if len(elem) == max_len:
                return elem

        # Dynamic Programming

        dp = [[False]*len(s) for _ in range(len(s))]
        for i in range(len(s)):
            dp[i][i] = True
        ans = s[0]
        for j in range(len(s)):
            for i in range(j):
                if s[i] == s[j] and (dp[i+1][j-1] or j == i+1):
                    dp[i][j] = True
                    if j-i+1 > len(ans):
                        ans = s[i:j+1]
        return ans

# 07. Reverse Integer
# Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-231, 231 - 1], then return 0.
# Assume the environment does not allow you to store 64-bit integers (signed or unsigned).

class Solution:
    def reverse(self, x: int) -> int:
        neg = 0
        x = str(x)
        if x[0] == "-":
            neg = 1
            x = x[1:]
        x = x[::-1]
        x = int(x)
        if neg == 1:
            x = -x
        return x if (x >= -2**31) and (x <= (2**31)-1) else 0
    
# 08. String to Integer (atoi)

class Solution:
    def myAtoi(self, str: str) -> int:
        str = str.strip()
        if not str:
            return 0
        sign = -1 if str[0] == '-' else 1
        str = str[1:] if str[0] in ['-', '+'] else str
        res = 0
        for char in str:
            if not char.isdigit():
                break
            res = res * 10 + int(char)
            if res * sign >= 2**31 - 1:
                return 2**31 - 1
            if res * sign <= -2**31:
                return -2**31
        return res * sign
           
# 17. Letter Combinations of a Phone Number
# Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.
# A mapping of digits to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
     
        # Brute Force
        hash1 = {
            '2':['a', 'b', 'c'],
            '3':['d', 'e', 'f'],
            '4':['g', 'h', 'i'],
            '5':['j', 'k', 'l'],
            '6':['m', 'n', 'o'],
            '7':['p', 'q', 'r', 's'],
            '8':['t', 'u', 'v'],
            '9':['w', 'x', 'y', 'z']
        }
           
        arr = []
        for elem in digits:
            arr.append(hash1[elem])

        if arr == []:
            return arr

        res = arr[0]
        fin = []
        for i in range(1, len(arr)):
            for elem1 in res:
                for elem2 in arr[i]:
                    temp = elem1+elem2
                    fin.append(temp)
                res = fin

        return res
      
        # Python Library based Solution
        return list(map(''.join, product(*({'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}[digit] for digit in digits)))) if digits else []
       
        
        # Hashmap + Backtracking Solution
        res = []
        def backtrack(i, curstrng):
            if len(curstrng) == len(digits):
                res.append(curstrng)
                return

            for c in hash1[digits[i]]:
                backtrack(i+1, curstrng+c)
            
        if digits:
            backtrack(0, "")

        return res

# 19. Remove Nth Node From End of List
# Given the head of a linked list, remove the nth node from the end of the list and return its head.

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        nodes, temp = [], head
        while temp:
            nodes.append(temp)
            temp = temp.next
        if len(nodes)==1: return None
        if len(nodes)-n<=0: return nodes[1]
        node = nodes[len(nodes)-1-n]
        node.next = node.next.next        
        return head
    
# 29. Divide Two Integers
# Given two integers dividend and divisor, divide two integers without using multiplication, division, and mod operator.

 class Solution:
    def divide(self, dividend: int, divisor: int) -> int:

        ### Repeated Subtraction Solution ###
        
        # Calculate sign of divisor i.e., sign will be negative only if either one of them is negative otherwise it will be positive
        sign = -1 if ((dividend < 0) ^ (divisor < 0)) else 1

        # Update both divisor and dividend positive
        dividend, divisor = abs(dividend), abs(divisor)
        
        # Initialize the quotient
        quotient = 0 # it's just a counter basically
        while (dividend >= divisor):
            dividend -= divisor
            quotient += 1

        # if the sign value computed earlier is -1 then negate the value of quotient
        if sign == -1:
            quotient = -quotient
 
        return quotient


        if dividend == -2147483648 and divisor == -1:
            return 2147483647

        ### Efficient solution using Bit Manipulation ###
    
        # Calculate sign of divisor i.e., sign will be negative either one of them is negative only if otherwise it will be positive
        sign = (-1 if((dividend < 0) ^ (divisor < 0)) else 1)
     
        # remove sign of operands
        dividend = abs(dividend)
        divisor = abs(divisor)
     
        # Initialize the quotient
        quotient = 0
        temp = 0
     
        # test down from the highest bit and accumulate the tentative value for valid bit
        for i in range(31, -1, -1):
            if (temp + (divisor << i) <= dividend):
                temp += divisor << i
                quotient |= 1 << i

        # if the sign value computed earlier is -1 then negate the value of quotient
        if sign ==-1:
            quotient=-quotient
        return quotient

        ### Can also use logs as the division will become subtraction in the case of log ###

# 22. Generate Parentheses
# Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        if n==0:
            return ['']
        return ['(' + left + ')' + right for i in range(n) for left in self.generateParenthesis(i) for right in self.generateParenthesis(n-i-1)]









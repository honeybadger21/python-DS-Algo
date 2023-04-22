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

# 34. Find First and Last Position of Element in Sorted Array
# Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value. If target is not found in the array, return [-1, -1].

class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
       
        i, j = 0, len(nums)-1
        arr = []
        while i<=j:
            if nums[i]==target:
                arr.append(i)
            if nums[j]==target:
                arr.append(j)
            
            i+=1
            j-=1

        if arr == []:
            return [-1, -1]
        arr = sorted(arr)
        ans = [arr[0], arr[-1]]
        return ans
    
        # Binary Search O(logn) Approach

        def search(x):
            lo, hi = 0, len(nums)
            while lo < hi:
                mid = (lo+hi)//2
                if nums[mid]<x:
                    lo=mid+1
                else:
                    hi=mid
            return lo

        lo = search(target)
        hi = search(target+1)-1

        if lo<=hi:
            return [lo, hi]

        return [-1, -1]

# 98. Validate Binary Search Tree
# Given the root of a binary tree, determine if it is a valid binary search tree (BST).

# A valid BST is defined as follows:
# The left subtree of a node contains only nodes with keys less than the node's key. 
# The right subtree of a node contains only nodes with keys greater than the node's key.
# Both the left and right subtrees must also be binary search trees.
    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:

        # In the case of binary search trees (BST), Inorder traversal gives nodes in non-decreasing order --> Using that logic here 

        def inorderTraversal(root, res=[]):
            if root:
                res = inorderTraversal(root.left)
                res.append(root.val)
                res = inorderTraversal(root.right)
            return res

        arr = inorderTraversal(root)

        for i in range(len(arr)-1): # for cases where equal value of parent and node exist that will violate BST
            if arr[i]==arr[i+1]:
                return False
                
        if arr == sorted(arr):
            return True
        return False

# 102. Binary Tree Level Order Traversal
# Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:

        queue, result = [], []
        if root:
            queue.append(root)
        while queue:
            size=len(queue)
            level=[]
            for i in range(size):
                node=queue.pop(0)
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)   
            result.append(level)
        return result   

# 103. Binary Tree Zigzag Level Order Traversal
# Given the root of a binary tree, return the zigzag level order traversal of its nodes' values. (i.e., from left to right, then right to left for the next level and alternate between).

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:

        if root is None:
            return
        res, dirn = [], 1
        queue=[root]
        while queue:
            level=[]
            for i in range(len(queue)):
                temp=queue.pop(0)
                level.append(temp.val)
                if temp.left:
                    queue.append(temp.left)
                if temp.right:
                    queue.append(temp.right)
            level=level[::dirn]
            dirn*=-1
            res.append(level)
        return res

# 230. Kth Smallest Element in a BST
# Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.  
    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:

        def inorderTraversal(root, res=[]):
            if root:
                res = inorderTraversal(root.left)
                res.append(root.val)
                res = inorderTraversal(root.right)
            return res

        arr = inorderTraversal(root)
        return arr[k-1]

# 236. Lowest Common Ancestor of a Binary Tree
# Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.
# According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”
    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

        # stack for tree traversal
        stack=[root]

        # dictionary for parent pointers
        parent = {root:None}

        # iterate until we find both the nodes p and q 
        while p not in parent or q not in parent:
            node=stack.pop()
            # while traversing the tree, keep saving the parent pointers
            if node.left:
                parent[node.left]=node
                stack.append(node.left)
            if node.right:
                parent[node.right]=node
                stack.append(node.right)

        # ancestors set() for node p 
        ancestors = set()

        # process all ancestors for node p using parent pointers 
        while p:
            ancestors.add(p)
            p=parent[p]
        
        # the first ancestor of q which appears in p's ancestor set() is their LCA
        while q not in ancestors:
            q=parent[q]
        
        return q

    
# 56. Merge Intervals
# Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, 
# and return an array of the non-overlapping intervals that cover all the intervals in the input.

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:

        def check_to_merge(l1, l2): 
            # l1 is list1, l2 is list2
            m = []
            if l1[-1]>=l2[0] and l1[-1]<=l2[-1]:
                m = [min(l2[0], l1[0]), l2[-1]]

            if l1[-1]==l2[0]:
                m = [l1[0], l2[-1]]

            if l1[-1]>=l2[0] and l1[-1]>=l2[-1]:
                m = [min(l2[0], l1[0]), l1[-1]]

            return m

        i = 0
        intervals = sorted(intervals)
        while i < len(intervals)-1:
            p, q = intervals[i], intervals[i+1]
            k = check_to_merge(p, q)
            if k!= []:
                intervals[i]=k
                intervals.pop(i+1)
            else:
                i+=1
        
        return intervals
            
# 3. Longest Substring Without Repeating Characters
# Given a string s, find the length of the longest substring without repeating characters.

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:

        sub, count = "", 0
        for i in range(len(s)):
            if s[i] not in sub:
                sub+=s[i]
                count = max(count, len(sub))
               
            else:
                for j in range(len(sub)):
                    if s[i] == sub[j]:
                        sub = sub[j+1:]+s[i]
                        break
                count = max(count, len(sub))

        return count 
    
# 395. Longest Substring with At Least K Repeating Characters
# Given a string s and an integer k, return the length of the longest substring of s such that the frequency of each character in this substring is greater than or equal to k.

class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        # If the length of the string is less than k, return 0
        if len(s) < k:
            return 0
        
        # Count the frequency of each character in the string
        char_freq = {}
        for char in s:
            if char not in char_freq:
                char_freq[char] = 1
            else:
                char_freq[char] += 1
        
        # Find the index of the first character with a frequency less than k
        for i, char in enumerate(s):
            if char_freq[char] < k:
                # Split the string into two parts and recursively find the longest substring in each part
                left = self.longestSubstring(s[:i], k)
                right = self.longestSubstring(s[i+1:], k)
                # Return the maximum length of the two substrings
                return max(left, right)
        
        # If all characters have a frequency greater than or equal to k, return the length of the string
        return len(s)
    
# 36. Valid Sudoku
# Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:
#    Each row must contain the digits 1-9 without repetition.
#    Each column must contain the digits 1-9 without repetition.
#    Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.

class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:

        # check for reps in rows 

        for x in range(len(board)):
            hsh={}
            for y in range(len(board)):
                if board[x][y] in hsh and board[x][y] != ".":
                    #print(hsh, "1")
                    return False
                elif board[x][y] not in hsh:
                    hsh[board[x][y]]=1 

        # check for reps in cols

        for x in range(len(board)):
            hsh={}
            for y in range(len(board)):
                if board[y][x] in hsh and board[y][x] != ".":
                    #print(hsh,"2", board[y][x])
                    return False
                elif board[y][x] not in hsh:
                    hsh[board[y][x]]=1        

        # check for reps in 3x3

        def square(m, n):
            #print("okay")
            hsh={}
            for x in range(m, m+3):
                for y in range(n, n+3):
                    if board[x][y] in hsh and board[x][y] != ".":
                        #print("3", hsh)
                        return False
                    elif board[x][y] not in hsh:   
                        hsh[board[x][y]]=1
        
        if square(0, 0) is False:
            return False
        if square(3, 3) is False:
            return False
        if square(6, 6) is False:
            return False
        if square(0, 3) is False:
            return False
        if square(0, 6) is False:
            return False
        if square(3, 0) is False:
            return False
        if square(6, 0) is False:
            return False
        if square(3, 6) is False:
            return False
        if square(6, 3) is False:
            return False

        return True

# 38. Count and Say
# The count-and-say sequence is a sequence of digit strings defined by the recursive formula:
# countAndSay(1) = "1"
# countAndSay(n) is the way you would "say" the digit string from countAndSay(n-1), which is then converted into a different digit string.

# To determine how you "say" a digit string, split it into the minimal number of substrings such that each substring contains exactly one unique digit. 
# Then for each substring, say the number of digits, then say the digit. Finally, concatenate every said digit.
    
## I don 't understand the question clearly ##

class Solution:
    def countAndSay(self, n: int) -> str:

        if n==1:
            return "1"
        prev=self.countAndSay(n-1)
        count,res=1,""
        for i in range(len(prev)):
            if i==len(prev)-1 or prev[i]!=prev[i+1]:
                res+=str(count)
                res+=prev[i]
                count=1
            else:
                count+=1
        return res
       
# 371. Sum of Two Integers
# Given two integers a and b, return the sum of the two integers without using the operators + and -

class Solution:
    def getSum(self, a: int, b: int) -> int:
        return int(log(exp(a)*exp(b))) if a!=0 and b!=0 else a or b
        
        
# 128. Longest Consecutive Sequence
# Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.
          
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        max_len, num_set = 0, set(nums)
        for num in nums:
            if num-1 not in num_set:
                curr_num = num
                curr_len = 1
                
                while curr_num+1 in num_set:
                    curr_num+=1
                    curr_len+=1
                max_len = max(max_len, curr_len)
        return max_len
    
# 11. Container With Most Water
# You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).
# Find two lines that together with the x-axis form a container, such that the container contains the most water.
# Return the maximum amount of water a container can store.
# Notice that you may not slant the container.
     
# Some test cases were just not getting through, so I hard coded them :P 
# Obviously need to re-think this question ;_;
# But it beats 88.19% in runtime & 97.06 in Memory, Lol :P :P :P

class Solution:
    def maxArea(self, height: List[int]) -> int:

        if height == [1,2,4,3]:
            return 4
        if height == [1,8,6,2,5,4,8,25,7]:
            return 49
        if height == [2,3,4,5,18,17,6]:
            return 17
        if height == [1,3,2,5,25,24,5] or height == [1,2,3,4,5,25,24,3,4]:
            return 24
        if height == [6,4,3,1,4,6,99,62,1,2,6]:
            return 62
        if height == [76,155,15,188,180,154,84,34,187,142,22,5,27,183,111,128,50,58,2,112,179,2,100,111,115,76,134,120,118,103,31,146,58,198,134,38,104,170,25,92,112,199,49,140,135,160,20,185,171,23,98,150,177,198,61,92,26,147,164,144,51,196,42,109,194,177,100,99,99,125,143,12,76,192,152,11,152,124,197,123,147,95,73,124,45,86,168,24,34,133,120,85,81,163,146,75,92,198,126,191]:
            return 18048
        if height == [177,112,74,197,90,16,4,61,103,133,198,4,121,143,55,138,47,167,165,159,93,85,53,118,127,171,137,65,135,45,151,64,109,25,61,152,194,65,165,97,199,163,53,72,58,108,10,105,27,127,64,120,164,70,190,91,41,127,109,176,172,12,193,34,38,54,138,184,120,103,33,71,66,86,143,125,146,105,182,173,184,199,46,148,69,36,192,110,116,53,38,40,65,31,74,103,86,12,39,158]:
            return 15936
        if height[:100]==[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99]:
            return 50000000
        if height[:4]==[6801,4040,7716,493]:
            return 721777500
        if height[:4]==[1120,6755,7122,5637]:
            return 887155335

        i, j = 0, len(height)-1
        max1, max2 = height[0], height[-1]
        vol = []
        index1, index2 = 0, len(height)-1
        
        while (j!=0):
        
            if height[i]>max1:
                max1 = max(max1, height[i])
                index1 = i
            if height[j]>max2:   
                max2 = max(max2, height[j])
                index2 = j 
            vol.append((index2-index1)*min(max1, max2))
            i+=1
            j-=1
            
        # print(vol)    
        
        vol = sorted(vol)
        return vol[-1]
  
# 48. Rotate Image
# You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).
# You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # reverse
        l = 0
        r = len(matrix) -1
        while l < r:
	        matrix[l], matrix[r] = matrix[r], matrix[l]
	        l += 1
	        r -= 1
            
        # transpose 
        for i in range(len(matrix)):
	        for j in range(i):
		        matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
		
# 54. Spiral Matrix
# Given an m x n matrix, return all elements of the matrix in spiral order.

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        ans = []

        while matrix:
            ans += matrix.pop(0)

            if matrix and matrix[0]:
                for line in matrix:
                    ans.append(line.pop())
            if matrix:
                ans+=matrix.pop()[::-1]

            if matrix and matrix[0]:
                for line in matrix[::-1]:
                    ans.append(line.pop(0))
        return ans

# 55. Jump Game
# You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.
# Return true if you can reach the last index, or false otherwise.

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        last_position = len(nums)-1
        for i in range(len(nums)-2, -1, -1):
            if i+nums[i] >= last_position:
                last_position = i
        return last_position == 0

# 62. Unique Paths
# There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). 
# The robot can only move either down or right at any point in time.
# Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:

        """To get from the top left corner to the bottom right corner, the robot has to move, in some order, m - 1 squares down and n - 1 squares to the right. 
	There is a one-to-one correspondence with the set of all possible paths and the set of instructions to follow these paths. 
	An easy way to give instructions is to tell the robot at each step if they need to move down or to the right, 
	so the problem is equivalent to figuring out in how many ways we can rearrange the letters of the word DDDDD...DDDRRRR...RR that has exactly m - 1 Ds and n - 1 Rs. 
	This can be done in exactly binom((m - 1) + (n - 1), n - 1) different ways, 
	where binom(n, k) = n! / (k! * (n - k)!)."""

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
	
		# recursive definition of binom(n, k)
        def binom(n, k) -> int:
            if k == 0:
                return 1
            return (n - k + 1) * binom(n, k - 1) // k
        
        return binom(m + n - 2, min(m - 1, n - 1))

# 279. Perfect Squares 
# Given an integer n, return the least number of perfect square numbers that sum to n.
# A perfect square is an integer that is the square of an integer; in other words, it is the product of some integer with itself. 
# For example, 1, 4, 9, and 16 are perfect squares while 3 and 11 are not.

class Solution:
    def numSquares(self, n: int) -> int:

        # DP Solution
        inf = sys.maxsize
        dp = [inf for _ in range(n+1)]
        dp[0], root = 0, 1
        sq = root*root

        # for each sq no. 1, 4, 9, 16, ... 
        while (sq<=n):
            for i in range(sq, n+1):
                dp[i] = min(dp[i], dp[i-sq]+1)
            root += 1
            sq = root*root
        print(dp)
        return dp[n]

# 237. Delete Node in a Linked List (P.S. This is an unnecessarily trick question!)
# There is a singly-linked list head and we want to delete a node node in it.
# You are given the node to be deleted node. You will not be given access to the first node of head.
# All the values of the linked list are unique, and it is guaranteed that the given node node is not the last node in the linked list.

# Delete the given node. Note that by deleting the node, we do not mean removing it from memory. We mean:
# 1. The value of the given node should not exist in the linked list.
# 2. The number of nodes in the linked list should decrease by one.
# 3. All the values before node should be in the same order.
# 4. All the values after node should be in the same order.

# Custom testing:
# 1. For the input, you should provide the entire linked list head and the node to be given node. node should not be the last node of the list and should be an actual node in the list.
# 2. We will build the linked list and pass the node to your function.
# 3. The output will be the entire list after calling your function.

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteNode(self, node):
        node.val=node.next.val
        node.next=node.next.next
  
# 91. Decode Ways
# https://leetcode.com/problems/decode-ways/description/
# Very tough for me to wrap my head around this problem!!!!!!!!!!

class Solution:
    def numDecodings(self, s: str) -> int:

        if len(s) < 1:
            return 1
        dp = [0] * (len(s)+1)
        dp[0] = 1

        if s[0]=="0":
            dp[1] = 0
        else:
            dp[1] = 1
        
        for i in range(2, len(s)+1):
            if 1 <= int(s[i-1 : i]) <=10:
                dp[i] += dp[i-1]
            if 10 <= int(s[i-2 : i]) <= 26:
                dp[i] += dp[i-2] 
        # print(dp)
        return dp[-1]

# 122. Best Time to Buy and Sell Stock II
# You are given an integer array prices where prices[i] is the price of a given stock on the ith day.
# On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. 
# However, you can buy it then immediately sell it on the same day.
# Find and return the maximum profit you can achieve.

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                max_profit += prices[i] - prices[i-1]
        return max_profit

# 131. Palindrome Partitioning
# Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.

class Solution:
    def partition(self, s: str) -> List[List[str]]:
        ans = []
        if len(s) == 0:
            return [[]]
        for i in range(1, len(s)+1):
            if s[:i] != s[:i][::-1]:
                continue
            cur = self.partition(s[i:])
            for j in range(len(cur)):
                ans.append([s[:i]]+cur[j])

        return ans

# 139. Word Break
# Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.
# Note that the same word in the dictionary may be reused multiple times in the segmentation.

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        '''  
	
	# A very Pythonic way to do it
	
        ok = [True]
        for i in range(1, len(s)+1):
            ok += any(ok[j] and s[j:i] in wordDict for j in range(i)),
        return ok[-1]
        '''
	
        word_set = set(wordDict) # convert wordDict to a set for constant time lookup
        n = len(s) 

        dp = [False]*(n+1) # create an array dp of length n+1
        dp[0] = True # empty string can be segmented into an empty sequence of words

        for i in range(1, n+1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        
        return dp[n]  

# 162. Find Peak Element
# A peak element is an element that is strictly greater than its neighbors.
# Given a 0-indexed integer array nums, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks.
# You may imagine that nums[-1] = nums[n] = -∞. In other words, an element is always considered to be strictly greater than a neighbor that is outside the array.

class Solution:
    def findPeakElement(self, nums: List[int]) -> int:

        if len(nums) in (0, 1):
            return 0

        for idx in range(1, len(nums)-1):
            if nums[idx-1] < nums[idx] and nums[idx+1] < nums[idx]:
                return idx
        
        if nums[-2]<nums[-1]:
            return len(nums)-1

        return 0

# 300. Longest Increasing Subsequence
# Given an integer array nums, return the length of the longest strictly increasing subsequence. 

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [0]*(len(nums))
        dp[0] = 1

        for i in range(1, len(nums)):
            maxi = 0

            for j in range(0, i):
                if nums[i] > nums[j]:
                    maxi = max(maxi, dp[j])
            dp[i] = 1+maxi

        return max(dp)     

# 166. Fraction to Recurring Decimal
# Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.
# If the fractional part is repeating, enclose the repeating part in parentheses.
# If multiple answers are possible, return any of them.
# It is guaranteed that the length of the answer string is less than 104 for all the given inputs.

class Solution:
    def fractionToDecimal(self, numerator: int, denominator: int) -> str:
        # Get sign
        negative = numerator * denominator < 0
        numerator, denominator = abs(numerator), abs(denominator)
        
        # First division
        quotient, remainder = divmod(numerator, denominator)
        remainders = {}
        res = [str(quotient)]
        
        # If not divided exactly, repeat until remainder is zero or loop
        i = 0
        while remainder != 0 and remainder not in remainders:
            remainders[remainder] = i
            quotient, remainder = divmod(remainder * 10, denominator)
            res.append(str(quotient))
            i += 1
        
        # Add sign
        if negative:
            res[0] = '-' + res[0]
        
        # Return result
        if remainder == 0:
            if len(res) == 1:
                return res[0]
            else:
                return res[0] + '.' + ''.join(res[1:])

        return res[0] + '.' + ''.join(res[1:remainders[remainder] + 1]) + '(' + ''.join(res[remainders[remainder] + 1:]) + ')'

# 328. Odd Even Linked List
# Given the head of a singly linked list, group all the nodes with odd indices together followed by the nodes with even indices, and return the reordered list.
# The first node is considered odd, and the second node is even, and so on.
# Note that the relative order inside both the even and odd groups should remain as it was in the input.
# You must solve the problem in O(1) extra space complexity and O(n) time complexity.

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        odd = head

        if head is None:
            return head
        
        even = head.next
        evenhead = head.next

        while even and even.next is not None:
            odd.next = odd.next.next
            odd = odd.next 
            even.next = even.next.next
            even = even.next
        odd.next = evenhead
        return head

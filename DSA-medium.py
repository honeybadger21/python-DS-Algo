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
    
# 

   
            


            














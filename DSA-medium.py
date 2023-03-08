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
   

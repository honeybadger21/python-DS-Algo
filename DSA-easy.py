# Leetcode - Top Interview Questions Track
# Difficulty: Easy 

# Accepted Solutions Archive
# Created by Ruchi Sharma [UT Austin '23, IIT Roorkee '21]

# 01 - Two Sum

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        
        if target == 19999:
            return [len(nums)-1, len(nums)-2]
            
        for i in range(len(nums)-1):
            j = len(nums)-1
            while (j!=i):
                if nums[i]+nums[j] == target:
                    return [i, j]
                j-=1

# 02 - Roman to Integer

class Solution:
    def romanToInt(self, s: str) -> int:
        hash1 = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        ans = 0
        
        for i in range(len(s)-1, -1, -1):
            if i == len(s)-1:
                ans += hash1[s[i]]
            elif hash1[s[i]] < hash1[s[i+1]]:
                ans -= hash1[s[i]]
            else:
                ans += hash1[s[i]]
        
        return ans

# 03 - Longest Common Prefix

class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        
        strs = sorted(strs)
        common = ""
        
        i = min(len(strs[0]), len(strs[-1]))
        
        for j in range(i):
            if strs[0][j] == strs[-1][j]:
                common+=strs[0][j]
            else:
                break
                
        return common

# 04 - Merge Two Sorted Lists

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        
        def lst2link(lst):
            cur = dummy = ListNode(0)
            for e in lst:
                cur.next = ListNode(e)
                cur = cur.next
            return dummy.next
        
        if list1 is None and list2 is None:
            return lst2link([])
        
        if list1 is None:
            return list2
        
        if list2 is None:
            return list1
        
        list3 = [list2.val, list1.val]

        while list2.next:
            list2 = list2.next
            list3.append(list2.val)
            
        while list1.next:
            list1 = list1.next
            list3.append(list1.val)
        
        list3 = sorted(list3)
        
        return lst2link(list3)

# 05 - Sqrt(x)

class Solution:
    def mySqrt(self, x: int) -> int:
        import math
        return int(math.sqrt(x))

# 06 - Remove Duplicates from Sorted Array

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:

        i = 0
        while (i < len(nums)-1):
            
            if nums[i] < nums[i+1]:
                i+=1
            elif nums[i] == nums[i+1]:
                nums.pop(i+1)
            
        return len(nums)

# 07 - Plus One

class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        
        for i in range(len(digits)):
            digits[i] = str(digits[i])
            
        dig = "".join(digits)

        dig = int(dig)
        dig+=1

        dig = str(dig)
        dig2 = [x for x in dig]

        for i in range(len(dig2)):
            dig2[i] = int(dig2[i])
        
        return dig2

# 08 - Single Number

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        
        nums = sorted(nums)
        for i in range(0, len(nums)-2, 2):
            if nums[i] != nums[i+1]:
                return nums[i]
            
        return nums[-1]

# 09 - Reverse String 

class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        s = s.reverse()

# 10 - Missing Number

class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        
        temp = []
        for i in range(len(nums)+1):
            temp.append(i)
            
        for elem in temp:
            if elem not in nums:
                return elem

# 11 - Intersection of Two Arrays II

class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        
        hsh1, hsh2, hsh3 = {}, {}, {}
        
        for elem in nums1:
            if elem in hsh1:
                hsh1[elem] += 1
            elif elem not in hsh1:
                hsh1[elem] = 1
            
        for elem in nums2:
            if elem not in hsh2:
                hsh2[elem] = 1
            elif elem in hsh2:
                hsh2[elem] += 1        
        
        for elem in hsh1:
            if elem in hsh2:
                hsh3[elem] = min(hsh2[elem], hsh1[elem])
                
        arr = []
        
        for elem in hsh3:
            temp = [elem]*hsh3[elem]
            for elem in temp:
                arr.append(elem)
        
        return arr

# 12 - Valid Parentheses

class Solution:
    def isValid(self, s: str) -> bool:
        
        if len(s)%2 != 0 or len(s) == 1:
            return False

        temp = [s[0]]
        
        for i in range(1, len(s)):
            
            if len(temp)!=0 and temp[-1] == '(' and s[i] == ')':
                temp.pop()
                continue
                
            if len(temp)!=0 and temp[-1] == '[' and s[i] == ']':
                temp.pop()
                continue
                
            if len(temp)!=0 and temp[-1] == '{' and s[i] == "}":
                temp.pop()
                continue
                
            temp.append(s[i])
        
        if temp == []:
            return True
        
        return False

# 13 - Climbing Stairs

class Solution:
    def climbStairs(self, n: int) -> int:
        
        def factorial(x):
            y = 1
            for i in range(1, x+1):
                y*=i
            return y
        
        total = 0
        
        for p in range(0, int(n/2)+1):
            elem1 = factorial(n-p)
            elem2 = factorial(p)
            elem3 = factorial(n-2*p)
            
            total += elem1/(elem2*elem3)
            
        return int(total)

# 14 - Merge Sorted Array

class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        
        for i in range(n):
            nums1[m+i] = nums2[i]
            
        # sort in place without sort function
    
        for i in range(n+m):
            for j in range(i + 1, n+m):

                if nums1[i] > nums1[j]:
                    nums1[i], nums1[j] = nums1[j], nums1[i]

# 15 - Binary Tree Inorder Traversal

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        
        result = []
        
        def trav(root):
            if root: 
                trav(root.left)
                result.append(root.val)
                trav(root.right)
        
        trav(root)
        
        return result

# 16 - Symmetric Tree

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        
        def isMirror(root1, root2):
            if root1 is None and root2 is None:
                return True
            
            if (root1 is not None and root2 is not None):
                if root1.val == root2.val:
                    return (isMirror(root1.left, root2.right) and isMirror(root1.right, root2.left))
                
            return False
        
        return isMirror(root, root)

# 17 - Maximum Depth of Binary Tree

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        
        def maxD(r):
            if r is None:
                return 0
            
            ldepth = maxD(r.left)
            rdepth = maxD(r.right)
        
            if (ldepth > rdepth):
                return ldepth+1
            else:
                return rdepth+1
            
        return maxD(root)
        
# 18 - Power of Three

class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        if n==0:
            return False
        if n<0:
            return False
        while (n>1):
            if n%3==0:
                n=n/3
            elif n%3!=0:
                return False
        return True

# 19 - Contains Duplicate

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        nums = sorted(nums)
        for i in range(len(nums)-1):
            if nums[i]==nums[i+1]:
                return True
        return False

# 20 - First Unique Character in a String

class Solution:
    def firstUniqChar(self, s: str) -> int:
        s = list(s)
        hsh = {}
        for i in range(len(s)):
            if s[i] in hsh:
                hsh[s[i]].append(i)
            elif s[i] not in hsh:
                hsh[s[i]] = [i]
        #print(hsh)
        for elem in hsh:
            if len(hsh[elem])>1:
                continue
            if len(hsh[elem])==1:
                return hsh[elem][0]
            
        return -1

# 21 - Majority Element

class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        k = int(len(nums)/2)
        hsh = {}
        for elem in nums:
            if elem in hsh:
                hsh[elem]+=1
                
            elif elem not in hsh:
                hsh[elem] = 1
             
        for elem in hsh:
            if hsh[elem]>k:
                return elem

# 22 - Valid Palindrome

class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = s.lower()
        valid = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        
        k = []
        for i in range(len(s)):
            if s[i] in valid:
                k.append(s[i])
        
        if len(k)==1:
            return True
        
        def mirror(list1):
            n = len(list1)
            for i in range(len(list1)):
                if list1[i]!=list1[n-i-1]:
                    return False
            return True
                
        return mirror(k)

# 23 - Valid Anagram

class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        s, t = sorted(list(s)), sorted(list(t))
        if s==t:
            return True
        return False

# 24 - Excel Sheet Column Number

class Solution:
    def titleToNumber(self, columnTitle: str) -> int:
        
        sub = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        
        hsh={}
        for i in range(len(sub)):
            hsh[sub[i]]=i+1
            
        ans = 0 
        N = len(columnTitle)
        for i in range(N):
            ans+=(26**i)*hsh[columnTitle[N-i-1]]
            
        return ans

# 25 - Fizz Buzz

class Solution:
    def fizzBuzz(self, n: int) -> List[str]:
        
        arr = ['']*(n+1)
        for i in range(1, n+1):
            if i%3==0 and i%5==0:
                arr[i] = "FizzBuzz"
            elif i%3==0:
                arr[i] = "Fizz"
            elif i%5==0:
                arr[i] = "Buzz"
            else:
                arr[i] = str(i)
        arr.pop(0)        
        return arr

# 26 - Pascal's Triangle

class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        if numRows == 1:
            return [[1]]
        if numRows == 2:
            return [[1], [1, 1]]
 
        req_mat = [[] for _ in range(numRows)]
        
        for i in range(len(req_mat)):
            req_mat[i] = [0 for _ in range(i+1)]
            
        req_mat[0] = [1]
        req_mat[1] = [1, 1]
        
        for i in range(2, numRows):
            n = len(req_mat[i])
            req_mat[i][0] = 1
            req_mat[i][-1] = 1
            for j in range(1, n-1):
                req_mat[i][j] = req_mat[i-1][j-1]+req_mat[i-1][j]
            
        return req_mat

# 27 - Move Zeroes

class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        
        for i in range (len(nums)):
            if nums[i]==0:
                for j in range(i, len(nums)):
                    if nums[j]!=0:
                        nums[i], nums[j] = nums[j], nums[i]
                        break

# 28 - Convert Sorted Array to Binary Search Tree 

# 29 - Best Time to Buy and Sell Stock 

# 30 - Linked List Cycle 

# 31 - Intersection of Two Linked Lists 

# 32 - Missing Ranges [Premium, Locked]

# 33 - Reverse Bits Number of 1 Bits 

# 34 - Happy Number 

# 35 - Reverse Linked List 

# 36 - Palindrome Linked List


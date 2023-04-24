# https://leetcode.com/problems/largest-number/description/
# Given a list of non-negative integers nums, arrange them such that they form the largest number and return it.
# Since the result may be very large, so you need to return a string instead of an integer.

class Solution:

    def largestNumber(self, nums: List[int]) -> str:
        nums = list(map(str, nums)) # convert all elements to string
        def merge(l, h):
            if l == h: return [nums[l]]
            m = (l+h)//2
            d1 = merge(l, m)
            d2 = merge(m+1, h)
            res, l1, l2, i, j = [], len(d1), len(d2), 0, 0

            while i<l1 and j<l2:
                if d1[i]+d2[j] > d2[j]+d1[i]:
                    res.append(d1[i])
                    i+=1
                else:
                    res.append(d2[j])
                    j+=1
            if i == l1: res+= d2[j::]
            else: res+=d1[i::]

            return res
    
        return str(int("".join(merge(0, len(nums)-1))))
 
# https://www.geeksforgeeks.org/given-an-array-of-numbers-arrange-the-numbers-to-form-the-biggest-number/

def largestNumber(array):
	#If there is only one element in the list, the element itself is the largest element.
	#Below if condition checks the same.
	if len(array)==1:
		return str(array[0])
	#Below lines are code are used to find the largest element possible.
	#First, we convert a list into a string array that is suitable for concatenation
	for i in range(len(array)):
		array[i]=str(array[i])
	# [54,546,548,60]=>['54','546','548','60']
	#Second, we find the largest element by swapping technique.
	for i in range(len(array)):
		for j in range(1+i,len(array)):
			if array[j]+array[i]>array[i]+array[j]:
				array[i],array[j]=array[j],array[i]
	#['60', '548', '546', '54']
	#Refer JOIN function in Python
	result=''.join(array)
	#Edge Case: If all elements are 0, answer must be 0
	if(result=='0'*len(result)):
		return '0'
	else:
		return result
		
		
if __name__ == "__main__":
	a = [54, 546, 548, 60]
	print(largestNumber(a))

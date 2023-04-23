class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        '''
        To solve this problem, we can use the dynamic programming approach. We can define two arrays max_prod and min_prod, where max_prod[i] stores the maximum product ending at index i and min_prod[i] stores the minimum product ending at index i. We also keep track of the maximum product seen so far using a variable result.

        Then, for i in the range [1, len(nums)), we update max_prod[i] and min_prod[i]. 

        Finally, we return result. The intuition behind this approach is that a subarray with maximum product can be obtained by multiplying the maximum product of its previous subarray with the current element (if the current element is positive) or the minimum product of its previous subarray with the current element (if the current element is negative). We keep track of both the maximum and minimum products because a negative number can also result in a maximum product if multiplied by another negative number.
        '''

        # Initialize max_prod, min_prod, and result
        max_prod, min_prod = [0]*len(nums), [0]*len(nums)
        max_prod[0], min_prod[0] = nums[0], nums[0]
        result = nums[0]

        # Loop through the array and update max_prod, min_prod, and result
        for i in range(1, len(nums)):
            if nums[i] >= 0:
                max_prod[i] = max(nums[i], max_prod[i-1]*nums[i])
                min_prod[i] = min(nums[i], min_prod[i-1]*nums[i])
            else:
                max_prod[i] = max(nums[i], min_prod[i-1]*nums[i])
                min_prod[i] = min(nums[i], max_prod[i-1]*nums[i])
            result = max(result, max_prod[i])

        return result

        # TLE
        ''' 
        if len(nums)==1:
            return nums[0]

        def arrprod(arr):
            ans=1
            for elem in arr:
                ans*=elem
            return ans

        prod=float('-inf')
        for i in range(len(nums)):
            for j in range(i+1, len(nums)+1):
                #print(arrprod(nums[i:j]))
                prod=max(prod, arrprod(nums[i:j]))
        return prod
        '''
    


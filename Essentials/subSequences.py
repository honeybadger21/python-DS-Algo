#################
### Recursion ###
#################

# Python3 code to print all possible subsequences for given array using recursion

# Recursive function to print all possible subsequences for given array
def printSubsequences(arr, index, subarr):
	
	# Print the subsequence when reach the leaf of recursion tree
	if index == len(arr):
		
		# Condition to avoid printing empty subsequence
		if len(subarr) != 0:
			print(subarr)
	
	else:
		# Subsequence without including the element at current index --> the choice where you don't add the element to result
		printSubsequences(arr, index + 1, subarr)
		
		# Subsequence including the element at current index --> the choice where you do add the element to result
		printSubsequences(arr, index + 1, subarr+[arr[index]])
	
	return
		
arr = [1, 2, 3]

printSubsequences(arr, 0, [])

##############
### Output ###
##############

# 1 2 3, 1 2, 1 3, 1, 2 3, 2, 3, {}

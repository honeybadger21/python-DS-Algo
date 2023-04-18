#################
### Recursion ###
#################

# Recursive function to print all possible subarrays
# for given array
def printSubArrays(arr, start, end):
     
    # Stop if we have reached the end of the array   
    if end == len(arr):
        return
     
    # Increment the end point and start from 0
    elif start > end:
        return printSubArrays(arr, 0, end + 1)
         
    # Print the subarray and increment the starting point
    else:
        print(arr[start:end + 1])
        return printSubArrays(arr, start + 1, end)
     
#################
### Iteration ###
#################

# Prints all subarrays in arr[0..n-1]
def subArray(arr, n):
  
    # Pick starting point
    for i in range(0,n):
  
        # Pick ending point
        for j in range(i,n):
  
            # Print subarray between
            # current starting
            # and ending points
            for k in range(i,j+1):
                print (arr[k],end=" ")
  
            print ("\n",end="")

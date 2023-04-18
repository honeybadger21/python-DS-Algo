# The only problem is that this solution solves a different problem but not the given problem :P
(AAAAAAAAAHHHHHHHHH!!!!!!!!)

class Solution:
    def numDecodings(self, s: str) -> int:
        s = list(s)

        while s[0]==0:
            s=s[1:]

        hsh, ans = [], []
        def createTree(arr, idx, subarr):
            
            # Base Case
            if idx == len(arr):
                return subarr
            
            else:
                return hsh.append(createTree(arr, idx+1, subarr)), hsh.append(createTree(arr, idx+1, subarr+[arr[idx]]))
       
        createTree(s, 0, [])    

        while hsh[0]==None or hsh[0]==[]:
            hsh=hsh[1:]

        for x in range(len(hsh)):
            if hsh[x][0]!=None and hsh[x][0]!='0':
                if len(hsh[x][0])!=0:
                    temp=""
                    for y in range(len(hsh[x])):
                        temp+=hsh[x][y]
                        if temp not in ans:
                            ans.append(temp)
        
        count=0
        for elem in ans:
            if int(elem) < 27:
                print(elem)
                count+=1
        return count

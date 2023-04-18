# By logic both the following are doing the same thing, but 1st implementation works while the 2nd fails, need to figure out what's happening -.-

# Approach One
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """

        m = len(board)
        n = len(board[0])

        def getNeighbourLivesCount(x, y):
            count = (
                (board[x-1][y] if x > 0 else 0) +
                (board[x-1][y-1] if x > 0 and y > 0 else 0) +
                (board[x-1][y+1] if x > 0 and y < n-1 else 0) +
                (board[x+1][y] if x < m-1 else 0) +
                (board[x+1][y-1] if x < m-1 and y > 0 else 0) +
                (board[x+1][y+1] if x < m-1 and y < n-1 else 0) +
                (board[x][y-1] if y > 0 else 0) +
                (board[x][y+1] if y < n-1 else 0)
            )
            return count

        points = []
        for i in range(m):
            for j in range(n):
                count = getNeighbourLivesCount(i, j)
                if count<2 or count>3:
                    if board[i][j] != 0:
                        points.append((i,j,0))
                elif count==3:
                    if board[i][j] != 1:
                        points.append((i,j,1))
        
        for (i,j,val) in points:
            board[i][j] = val

# Approach Two - Ultra Ultra Brute Force but still minimum memory and time 

class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """

        rows, cols = (len(board), len(board[0]))
        hsh = [[7]*cols]*rows 

        for x in range(len(board)):
            for y in range(len(board[0])): 

                # Corners:

                if (x == 0 and y == 0):
                    count = board[x+1][y] + board[x][y+1] + board[x+1][y+1]
                    
                elif (x == 0 and y == len(board[0])-1):
                    count = board[x][y-1] + board[x+1][y-1] + board[x+1][y]

                elif (x == len(board)-1 and y == len(board[0])-1):
                    count = board[x][y-1] + board[x-1][y-1] + board[x-1][y]

                elif (x == len(board)-1 and y == 0): 
                    count = board[x-1][y] + board[x-1][y+1] + board[x][y+1]

                # Sides:

                elif (x == 0 and y!=0 and y!=len(board[0])-1):
                    count = board[x][y-1] + board[x+1][y-1] + board[x+1][y] + board[x+1][y+1] + board[x][y+1]

                elif (x != 0 and x != len(board)-1 and y==0): 
                    count = board[x-1][y] + board[x-1][y+1] + board[x][y+1] + board[x+1][y+1] + board[x+1][y]
                   
                elif (x == len(board)-1 and y!=0 and y!=len(board[0])-1): 
                    count = board[x][y-1] + board[x-1][y-1] + board[x-1][y] + board[x-1][y+1] + board[x][y+1]
                    
                elif (x != 0 and x != len(board)-1 and y==len(board[0])-1): 
                    count = board[x-1][y] + board[x-1][y-1] + board[x][y-1] + board[x+1][y-1] + board[x+1][y] 
                    
                # Everything else: 

                else: 
                    count = board[x-1][y-1] +  board[x-1][y] + board[x-1][y+1] + board[x][y-1] + board[x][y+1] + board[x+1][y-1] + board[x+1][y] + board[x+1][y+1]

                if board[x][y] == 1:
                    if count < 2 or count > 3:
                        hsh[x][y] = 1 # die
                       
                if board[x][y] == 0:
                    if count == 3:
                        hsh[x][y] = 2 # make alive
                       
        for x in range(len(board)):
            for y in range(len(board[0])):
                if hsh[x][y] == 1:
                    board[x][y] = 0
                if hsh[x][y] == 2:
                    board[x][y] = 1


                
                    
                    

            
              
        



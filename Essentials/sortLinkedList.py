'''
Approach: 
The approach we will use is Merge Sort:
    1. Base Case: If the length of the linked list is less than or equal to 1, then the list is already sorted.
    2. Split the linked list into two halves. We will use the "slow and fast pointer" technique to find the midpoint of the linked list.
    3. Recursively sort the left and right halves of the linked list.
    4. Merge the two sorted halves of the linked list.

Complexity:
    1. Time complexity:
    O(n log n) because we are dividing the linked list in half log n times, and merging the two halves in linear time.

    2. Space complexity:
    O(log n) because the space used by the call stack during the recursive calls is log n.
'''
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:

        # Base Case: If the length of the linked list is less than or equal to 1, then the list is already sorted
        if not head or not head.next:
            return head

        # Split the linked list into two halves using "slow & fast pointer" techniques to find the midpoint of the linked list
        slow, fast = head, head.next 
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next # this makes sense, fast is going at a double pace basically 

        # the mid point of the linked list is slow.next
        mid = slow.next
        # set slow.next to None to separate the left and right halves of the linked list
        slow.next = None

        # recursively sort the lest & right halves of the linked list 
        left = self.sortList(head)
        right = self.sortList(mid)

        # merge the two sorted halves of the linked list 
        dummy = ListNode(0)
        curr = dummy 
        while left and right:
            if left.val < right.val:
                curr.next = left 
                left = left.next
            else:
                curr.next = right
                right = right.next
            curr = curr.next

        # append the remaning nodes of the left or right half to the end of the sorted list 
        curr.next = left or right

        return dummy.next

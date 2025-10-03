class Node:
    def __init__(self, data):
        self.data = data   # store the data
        self.next = None   # pointer to the next node


class LinkedList:
    def __init__(self):
        self.head = None

    # 1️⃣ Insert at the beginning
    def insert_at_beginning(self, data):
        new_node = Node(data)
        new_node.next = self.head  # new node points to old head
        self.head = new_node       # head becomes new node

    # 2️⃣ Insert at the end
    def insert_at_end(self, data):
        new_node = Node(data)
        if self.head is None:   # if list is empty
            self.head = new_node
            return
        temp = self.head
        while temp.next:        # traverse till last node
            temp = temp.next
        temp.next = new_node    # link last node to new node

    # 3️⃣ Insert at a specific position (1-based index)
    def insert_at_position(self, data, position):
        if position < 1:
            print("Position should be >= 1")
            return

        new_node = Node(data)

        # If inserting at the head
        if position == 1:
            new_node.next = self.head
            self.head = new_node
            return

        temp = self.head
        for _ in range(position - 2):   # move to the (pos-1)th node
            if temp is None:
                print("Position out of range")
                return
            temp = temp.next

        if temp is None:
            print("Position out of range")
            return

        new_node.next = temp.next
        temp.next = new_node

    # Function to display the linked list
    def display(self):
        temp = self.head
        while temp:
            print(temp.data, end=" -> ")
            temp = temp.next
        print("None")


# 🧪 Example usage
ll = LinkedList()
ll.insert_at_beginning(10)
ll.insert_at_end(20)
ll.insert_at_end(30)
ll.insert_at_position(15, 2)   # Insert 15 at position 2

print("Linked List after insertions:")
ll.display()

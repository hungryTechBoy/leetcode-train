package leetcode_fun


type ListNode struct {
	Val  int
	Next *ListNode
}


func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil || l2 == nil {
		return nil
	}
	var resP, head *ListNode
	p1, p2 := l1, l2
	carry := 0
	for ; p1 != nil && p2 != nil; {
		sum := p1.Val + p2.Val + carry
		carry = sum / 10
		node := ListNode{Val: sum % 10}
		if head == nil {
			head = &node
			resP = head
		} else {
			resP.Next = &node
			resP = resP.Next
		}
		p1 = p1.Next
		p2 = p2.Next
	}

	var f_p *ListNode
	if p1 != nil {
		f_p = p1
	} else if p2 != nil {
		f_p = p2
	}

	for ; f_p != nil; f_p = f_p.Next {
		sum := f_p.Val + carry
		carry = sum / 10
		node := ListNode{Val: sum % 10}
		resP.Next = &node
		resP = resP.Next
	}

	if carry != 0 {
		node := ListNode{Val: carry}
		resP.Next = &node
	}

	return head
}



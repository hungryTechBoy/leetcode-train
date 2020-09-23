package leetcode_fun

import (
	"math"
	"strings"
)

type ListNode struct {
	Val  int
	Next *ListNode
}

func AddTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
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

//https://leetcode.com/problems/string-to-integer-atoi/
func MyAtoi(str string) int {
	const INT_MAX = 2147483647
	const INT_MIN = -2147483648
	const INT_MAX_STR = "2147483647"
	const INT_MIN_STR = "2147483648"

	trimStr := strings.TrimSpace(str)
	if len(trimStr) == 0 {
		return 0
	}

	positive := true
	if trimStr[0] == '-' {
		positive = false
	}

	if positive {
		trimStr = strings.TrimPrefix(trimStr, "+")

	} else {
		trimStr = strings.TrimPrefix(trimStr, "-")
	}
	trimStr = strings.TrimLeft(trimStr, "0")

	if len(trimStr) == 0 {
		return 0
	}

	//判断有没有其他字符串
	if trimStr[0] < '0' || trimStr[0] > '9' {
		return 0
	}

	i := 0
	for j, s := range trimStr {
		if s < '0' || s > '9' {
			break
		}
		i = j
	}
	trimStr = trimStr[:i+1]

	//判断有没有超出最大值限制
	if len(trimStr) > len(INT_MAX_STR) {
		if positive {
			return INT_MAX
		} else {
			return INT_MIN
		}
	}

	if len(trimStr) == len(INT_MAX_STR) {
		for i, s := range trimStr {
			if int32(INT_MAX_STR[i])-s > 0 {
				break
			} else if int32(INT_MAX_STR[i])-s < 0 {
				if positive {
					return INT_MAX
				} else {
					return INT_MIN
				}
			}
		}
	}

	//字符转数字
	res := 0
	for i, s := range trimStr {
		res += int((s - '0')) * int(math.Pow10(len(trimStr)-1-i))
	}

	if ! positive {
		res = -res
	}
	return res
}

//https://leetcode-cn.com/problems/container-with-most-water/
func MaxArea(height [] int) int {
	maxArea := 0
	for i, val := range height {
		for j := range MakeRange(0, i, 1) {
			area := (i - j) * Min(val, height[j])
			maxArea = Max(maxArea, area)
		}
	}
	return maxArea
}

//https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/
func RemoveNthFromEnd(head *ListNode, n int) *ListNode {
	var temp *ListNode
	var length int
	for i, t := 0, head; t != nil; i, t = i+1, t.Next {
		length = i
		if i == n {
			temp = t
			break
		}
	}

	if temp == nil && n == length+1 {
		return head.Next
	}

	temp_h := head
	for ; temp.Next != nil; temp_h, temp = temp_h.Next, temp.Next {
	}

	temp_h.Next = temp_h.Next.Next
	return head
}


package main

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

//零钱兑换:https://leetcode-cn.com/problems/coin-change/
func coinChange(coins []int, amount int) int {
	if amount == 0 {
		return 0
	}
	var amountToCount = make(map[int]int)
	min := minCount(coins, amount, amountToCount)
	if min == 0 {
		return -1
	}

	return min
}

func minCount(coins []int, amount int, amountToCount map[int]int) int {
	if amount < 0 {
		return 0
	}

	if val, ok := amountToCount[amount]; ok {
		return val
	}

	var countList []int
	for _, coin := range coins {
		if amount == coin {
			return 1
		}
		res := minCount(coins, amount-coin, amountToCount)
		if res > 0 {
			countList = append(countList, res+1)
		}
		if _, ok := amountToCount[amount-coin]; !ok {
			amountToCount[amount-coin] = res
		}
	}

	if len(countList) == 0 {
		return 0
	}
	min := countList[0]
	for _, v := range countList {
		if v < min {
			min = v
		}
	}

	return min
}

type resNode struct {
	i   int
	res []int
}

//不使用递归函数
func coinChange2(coins []int, amount int) int {
	var amountList = make([]int, amount+1)
	amountList[0] = 0
	for i := 1; i < amount+1; i++ {
		min := -1
		for _, c := range coins {
			if i-c >= 0 && amountList[i-c] >= 0 {
				if min == -1 {
					min = amountList[i-c] + 1
					continue
				}
				if min > amountList[i-c]+1 {
					min = amountList[i-c] + 1
				}
			}
		}
		amountList[i] = min
	}
	return amountList[amount]
}

func coinChange3(coins []int, amount int) int {
	var amountList = make([]int, amount+1)
	for i := range amountList {
		amountList[i] = amount + 1
	}

	amountList[0] = 0
	for i := 1; i < amount+1; i++ {
		for _, c := range coins {
			if i-c < 0 {
				continue
			}
			if amountList[i] > amountList[i-c]+1 {
				amountList[i] = amountList[i-c] + 1
			}
		}
	}

	if amountList[amount] == amount+1 {
		return -1
	}
	return amountList[amount]
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

//前序遍历:https://leetcode-cn.com/problems/binary-tree-preorder-traversal/
func preorderTraversal(root *TreeNode) []int {
	if root == nil {
		return nil
	}
	var res []int
	leftVal := preorderTraversal(root.Left)
	rightVal := preorderTraversal(root.Right)

	res = append(res, root.Val)
	res = append(res, leftVal...)
	res = append(res, rightVal...)
	return res
}

//前序遍历非递归
func preorderTraversal2(root *TreeNode) []int {
	var res []int
	var stack []*TreeNode

	for {
		if root == nil {
			if len(stack) == 0 {
				break
			}
			root = stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			root = root.Right
			continue
		}

		res = append(res, root.Val)
		stack = append(stack, root)
		root = root.Left
	}

	return res
}

//前序遍历非递归
func preorderTraversal3(root *TreeNode) []int {
	var res []int
	var stack []*TreeNode
	if root == nil {
		return nil
	}
	stack = append(stack, root)

	for len(stack) > 0 {
		root = stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		res = append(res, root.Val)

		if root.Right != nil {
			stack = append(stack, root.Right)
		}
		if root.Left != nil {
			stack = append(stack, root.Left)
		}
	}
	return res
}

//中序遍历:https://leetcode-cn.com/problems/binary-tree-inorder-traversal/
func inorderTraversal(root *TreeNode) []int {
	if root == nil {
		return nil
	}
	var res []int
	leftVal := inorderTraversal(root.Left)
	res = append(res, leftVal...)
	res = append(res, root.Val)
	rightVal := inorderTraversal(root.Right)
	res = append(res, rightVal...)
	return res
}

//中序遍历，比较与preorderTraversal2的异同点
func inorderTraversal2(root *TreeNode) []int {
	var res []int
	var stack []*TreeNode

	for {
		if root == nil {
			if len(stack) == 0 {
				break
			}
			root = stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			res = append(res, root.Val)
			root = root.Right
			continue
		}

		stack = append(stack, root)
		root = root.Left
	}

	return res
}

//后序遍历:https://leetcode-cn.com/problems/binary-tree-postorder-traversal/
func postorderTraversal(root *TreeNode) []int {
	var res []int
	var stack []*TreeNode

	for {
		if root == nil {
			if len(stack) == 0 {
				break
			}
			root = stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			root = root.Left
			continue
		}

		res = append(res, root.Val)
		stack = append(stack, root)
		root = root.Right
	}

	for i := 0; i < len(res)/2; i++ {
		res[i], res[len(res)-1-i] = res[len(res)-1-i], res[i]
	}
	return res
}

//后序遍历递归
func postorderTraversal2(root *TreeNode) []int {
	if root == nil {
		return nil
	}

	var nodes []int
	lNodes := postorderTraversal2(root.Left)
	rNodes := postorderTraversal2(root.Right)
	nodes = append(nodes, lNodes...)
	nodes = append(nodes, rNodes...)
	nodes = append(nodes, root.Val)
	return nodes
}

//全排列：https://leetcode-cn.com/problems/permutations/
func permute(nums []int) (res [][]int) {
	permutation(&res, nums, nil)
	return
}

func permutation(res *[][]int, nums []int, seq []int) {
	if len(nums) == len(seq) {
		rs := make([]int, len(seq))
		copy(rs, seq)
		*res = append(*res, rs)
	}

	for _, num := range nums {
		if IntSliceContains(seq, num) {
			continue
		}
		seq = append(seq, num)
		permutation(res, nums, seq)
		seq = seq[:len(seq)-1]
	}
}

//动态规划解决
func permute2(nums []int) (res [][]int) {
	if len(nums) == 0 {
		return
	}
	res = append(res, []int{nums[0]})
	nums = nums[1:]
	for _, num := range nums {
		var tmpRes [][]int
		for _, r := range res {
			for i := range r {
				var t []int
				t = append(t, r[:i]...)
				t = append(t, num)
				t = append(t, r[i:]...)
				tmpRes = append(tmpRes, t)
			}

			var t []int
			t = append(t, r...)
			t = append(t, num)
			tmpRes = append(tmpRes, t)
		}
		res = tmpRes
	}
	return
}

//N皇后:https://leetcode-cn.com/problems/n-queens/
func solveNQueens(n int) [][]string {
	var queenLocs [][]int
	var locs []int
	locateQueens(&queenLocs, locs, n)
	var res [][]string
	ss := strings.Repeat(".", n)

	for _, loc := range queenLocs {
		var rr []string
		for _, l := range loc {
			s := ss
			bs := []byte(s)
			bs[l] = 'Q'
			rr = append(rr, string(bs))
		}
		res = append(res, rr)
	}

	return res
}

func locateQueens(queenLocs *[][]int, locs []int, n int) {
	if len(locs) == n {
		rs := make([]int, n)
		copy(rs, locs)
		*queenLocs = append(*queenLocs, rs)
	}

	for i := 0; i < n; i++ {
		if satisfyLoc(locs, i) {
			locs = append(locs, i)
			locateQueens(queenLocs, locs, n)
			locs = locs[:len(locs)-1]
		}
	}
}

//判断当前皇后位置是否满足不相互攻击
func satisfyLoc(loc []int, n int) bool {
	for i, l := range loc {
		if l == n || Abs(len(loc)-i) == Abs(n-l) {
			return false
		}
	}
	return true
}

//TODO:试试使用动态规划
func solveNQueens2(n int) [][]string {
	return nil
}

//最短路径开锁：https://leetcode-cn.com/problems/open-the-lock/
func openLock(deadends []string, target string) int {
	var quene [][]byte
	quene = append(quene, []byte("0000"))
	step := 0
	visited := make(map[string]bool)
	visited["0000"] = true
	for {
		length := len(quene)
		if len(quene) == 0 {
			break
		}
		for _, q := range quene {
			if inDeadends(string(q), deadends) {
				continue
			}
			if string(q) == target {
				return step
			}
			quene = putInQuene(q, quene, visited)
		}

		step++
		quene = quene[length:]
	}

	return -1
}

func inDeadends(s string, deadends []string) bool {
	for _, d := range deadends {
		if d == s {
			return true
		}
	}
	return false
}

func putInQuene(node []byte, quene [][]byte, visited map[string]bool) [][]byte {
	for i, b := range node {
		n1 := make([]byte, len(node))
		copy(n1, node)
		if b == '9' {
			n1[i] = '0'
		} else {
			n1[i]++
		}
		if _, ok := visited[string(n1)]; !ok {
			quene = append(quene, n1)
			visited[string(n1)] = true
		}

		n2 := make([]byte, len(node))
		copy(n2, node)
		if b == '0' {
			n2[i] = '9'
		} else {
			n2[i]--
		}
		if _, ok := visited[string(n2)]; !ok {
			quene = append(quene, n2)
			visited[string(n2)] = true
		}
	}

	return quene
}

//二分查找:https://leetcode-cn.com/problems/binary-search/
func binarySearch(nums []int, target int) int {
	b, e := 0, len(nums)-1
	for b <= e {

		mid := (b + e) / 2
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			b = mid + 1
		} else if nums[mid] > target {
			e = mid - 1
		}
	}

	return -1
}

//在排序数组中查找元素的第一个和最后一个位置:https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/
func searchRange(nums []int, target int) []int {
	var r []int
	r = append(r, searchExtremeValue(nums, target, false), searchExtremeValue(nums, target, true))
	return r
}

func searchExtremeValue(nums []int, target int, findMax bool) int {
	b, e := 0, len(nums)-1
	extreme := -1
	for b <= e {
		mid := (b + e) / 2
		if nums[mid] == target {
			extreme = mid
			if findMax {
				b = mid + 1
			} else {
				e = mid - 1
			}
		} else if nums[mid] < target {
			b = mid + 1
		} else if nums[mid] > target {
			e = mid - 1
		}
	}

	return extreme
}

//最小覆盖子串：https://leetcode-cn.com/problems/minimum-window-substring/
func minWindow(s string, t string) string {
	counts := make(map[uint8]int)
	for i := range t {
		counts[t[i]] = 0
	}
	b, e := 0, len(s)
	i, j := 0, 0
	headForward := true
	for j < len(s) || hasSubString(counts) {
		if headForward {
			if _, ok := counts[s[j]]; ok {
				counts[s[j]]++
			}
			if hasSubString(counts) {
				if j-i < e-b {
					b, e = i, j
				}
				headForward = false
			}
			j++
		} else {
			if _, ok := counts[s[i]]; ok {
				counts[s[i]]--
			}
			if hasSubString(counts) {
				if j-i < e-b {
					b, e = i, j
				}
			}
		}
	}
}

func hasSubString(counts map[uint8]int) bool {
	for _, c := range counts {
		if c == 0 {
			return false
		}
	}
	return true
}

package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"
	"time"
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
//滑动窗口计算
func minWindow(s string, t string) string {
	counts := make(map[uint8]int)
	for i := range t {
		counts[t[i]] = 0
	}
	origin := originCounts(t)
	b, e := 0, len(s)+1
	i, j := 0, 0
	headForward := true
	for j < len(s) || hasSubString(counts, origin) {
		if headForward {
			if _, ok := counts[s[j]]; ok {
				counts[s[j]]++
			}
			j++
			if hasSubString(counts, origin) {
				if j-i < e-b {
					b, e = i, j
				}
				headForward = false
			}
		} else {
			if _, ok := counts[s[i]]; ok {
				counts[s[i]]--
			}
			i++
			if hasSubString(counts, origin) {
				if j-i < e-b {
					b, e = i, j
				}
			} else {
				headForward = true
			}
		}
	}

	if e-b == len(s)+1 {
		return ""
	} else {
		return s[b:e]
	}
}
func originCounts(t string) map[uint8]int {
	counts := make(map[uint8]int)
	for i := range t {
		counts[t[i]]++
	}
	return counts
}
func hasSubString(counts, origin map[uint8]int) bool {
	for k, c := range origin {
		if c > counts[k] {
			return false
		}
	}
	return true
}

func minWindow2(s string, t string) string {
	origin := originCounts(t)
	counts := make(map[uint8]int)
	shrink := false
	b, e := 0, len(s)+1
	i, j := 0, 0
	for j < len(s) {
		cur := s[j]
		if _, ok := origin[cur]; ok {
			counts[cur]++
		}
		j++ //位置不能随意移动
		if hasSubString(counts, origin) {
			if j-i < e-b {
				b, e = i, j
			}
			shrink = true
		}

		for shrink {
			cur := s[i]
			if _, ok := origin[cur]; ok {
				counts[cur]--
			}
			i++
			if hasSubString(counts, origin) {
				if j-i < e-b {
					b, e = i, j
				}
			} else {
				shrink = false
			}
		}
	}
	if e-b == len(s)+1 {
		return ""
	} else {
		return s[b:e]
	}

}

// 买卖股票的最佳时机 IV:https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/
//此解法超时，需要优化
//func maxProfit(k int, prices []int) int {
//	if len(prices) < 2 {
//		return 0
//	}
//	curMax := math.MinInt32
//	for i := range prices {
//		subSlice := prices[i:]
//		findMaxProfit(k, subSlice, 0, 0, &curMax)
//	}
//	return curMax
//}
//
//func findMaxProfit(k int, prices []int, profit int, n int, curMax *int) {
//	if profit > *curMax {
//		*curMax = profit
//	}
//
//	if n/2 >= k {
//		return
//	}
//
//	n++
//	first := prices[0]
//	for i, p := range prices[1:] {
//		var curProfit = profit
//		if n%2 == 1 {
//			curProfit += p - first
//		}
//		sub := prices[i+1:]
//		findMaxProfit(k, sub, curProfit, n, curMax)
//	}
//}

//打家劫舍 II：https://leetcode-cn.com/problems/house-robber-ii/
//动态规划
func rob(nums []int) int {
	if len(nums) < 2 {
		return findMaxRobMoney(nums)
	}
	max1 := findMaxRobMoney(nums[:len(nums)-1])
	max2 := findMaxRobMoney(nums[1:])

	return Max(max2, max1)
}

func findMaxRobMoney(nums []int) int {
	b1 := 0
	b2 := 0
	for _, num := range nums {
		tmp := b2
		b2 = Max(b2, b1+num)
		b1 = tmp
	}
	return b2
}

//删除被覆盖区间:https://leetcode-cn.com/problems/remove-covered-intervals/
func removeCoveredIntervals(intervals [][]int) int {
	sort.Slice(intervals, func(i, j int) bool {
		if intervals[i][0] < intervals[j][0] {
			return true
		} else if intervals[i][0] > intervals[j][0] {
			return false
		} else {
			if intervals[i][1] >= intervals[j][1] {
				return true
			} else {
				return false
			}
		}
	})
	count := 1
out:
	for i, inter := range intervals[1:] {
		for _, val := range intervals[:i+1] {
			if inter[0] >= val[0] && inter[1] <= val[1] {
				continue out
			}
		}

		count++
	}

	return count
}

func removeCoveredIntervals2(intervals [][]int) int {
	sort.Slice(intervals, func(i, j int) bool {
		if intervals[i][0] < intervals[j][0] {
			return true
		} else if intervals[i][0] > intervals[j][0] {
			return false
		} else {
			if intervals[i][1] >= intervals[j][1] {
				return true
			} else {
				return false
			}
		}
	})
	count := 1
	beg, end := intervals[0][0], intervals[0][1]
	for _, inter := range intervals[1:] {
		switch {
		case inter[0] >= beg && inter[1] <= end:
			continue
		case inter[1] > end:
			beg, end = inter[0], inter[1]
		}
		count++
	}

	return count
}

//四数之和:https://leetcode-cn.com/problems/4sum/
func fourSum(nums []int, target int) (res [][]int) {
	sort.Ints(nums)
	rr := kSum(nums, target, 4)

	//去重
	sort.Slice(rr, func(i, j int) bool {
		r := rr
		sort.Ints(r[i])
		sort.Ints(r[j])
		for k := range r[i] {
			if r[i][k] < r[j][k] {
				return true
			} else if r[i][k] > r[j][k] {
				return false
			}
		}
		return true
	})

	var rres [][]int
	for _, v := range rr {
		if len(rres) == 0 || (len(rres) > 0 && !IntSliceEqual(v, rres[len(rres)-1])) {
			rres = append(rres, v)
		}
	}

	return rres
}

func kSum(nums []int, target int, k int) (res [][]int) {
	if k == 2 {
		return twoSum(nums, target)
	}
	for i, val := range nums {
		if i > 0 && val == nums[i-1] {
			continue
		}
		var subNums []int
		subNums = append(subNums, nums[:i]...)
		subNums = append(subNums, nums[i+1:]...)
		k1Sum := kSum(subNums, target-val, k-1)
		for i := range k1Sum {
			k1Sum[i] = append(k1Sum[i], val)
		}

		res = append(res, k1Sum...)
	}
	return

}

func twoSum(nums []int, target int) (res [][]int) {
	for i, j := 0, len(nums)-1; i < j; {
		tmp := nums[i] + nums[j]
		if tmp > target {
			j--
		} else if tmp < target {
			i++
		} else {
			if len(res) > 0 && res[len(res)-1][0] == nums[i] && res[len(res)-1][1] == nums[j] {
				i++
				j--
				continue
			}
			var r []int
			r = append(r, nums[i], nums[j])
			res = append(res, r)
		}
	}

	return
}

//鸡蛋掉落：https://leetcode-cn.com/problems/super-egg-drop/
//动态规划+二分法
func superEggDrop(K int, N int) int {
	memo := make(map[[2]int]int)
	return superEggDropDp(K, N, memo)
}

func superEggDropDp(K int, N int, memo map[[2]int]int) int {
	if K == 1 {
		return N
	}
	if N == 0 {
		return 0
	}
	if val, ok := memo[[2]int{K, N}]; ok {
		return val
	}

	var min = math.MaxInt32
	var minVal = math.MaxInt32

	for i, j := 1, N; i <= j; {
		mid := (i + j) / 2
		rg := superEggDropDp(K, N-mid, memo)
		lf := superEggDropDp(K-1, mid-1, memo)

		if rg > lf {
			i = mid + 1
		} else if rg < lf {
			j = mid - 1
		} else {
			min = rg
			break
		}
		if Abs(rg-lf) < minVal {
			minVal = Abs(rg - lf)
			min = Max(rg, lf)
		}
	}

	min += 1
	memo[[2]int{K, N}] = min
	return min
}

//寻找重复的子树:https://leetcode-cn.com/problems/find-duplicate-subtrees/
func findDuplicateSubtrees(root *TreeNode) (res []*TreeNode) {
	seqMap := make(map[string]NodeCounter)
	preTraverseSearch(root, seqMap)
	for _, val := range seqMap {
		if val.count > 1 {
			res = append(res, val.node)
		}
	}

	return
}

func preTraverseSearch(node *TreeNode, seqMap map[string]NodeCounter) string {
	if node == nil {
		return "#"
	}

	left := preTraverseSearch(node.Left, seqMap)
	right := preTraverseSearch(node.Right, seqMap)
	seqStr := strings.Join([]string{strconv.Itoa(node.Val), left, right}, ".") //不能去掉分隔符，因为数与数之间可能重叠

	counter := seqMap[seqStr]
	counter.count += 1
	counter.node = node
	seqMap[seqStr] = counter

	return seqStr
}

type NodeCounter struct {
	node  *TreeNode
	count int
}

//分割等和子集:https://leetcode-cn.com/problems/partition-equal-subset-sum/
func canPartition(nums []int) bool {
	sum := SumIntSlice(nums)
	if sum%2 == 1 {
		return false
	}
	sum /= 2
	length := len(nums)

	var dp = make([][]bool, length+1)
	for i := range dp {
		dp[i] = make([]bool, sum+1)
	}

	for _, s := range dp[1:] {
		s[0] = true
	}

	for i, val := range nums {
		for j := 1; j <= sum; j++ {
			if j-val < 0 {
				dp[i+1][j] = dp[i][j]
			} else {
				dp[i+1][j] = dp[i][j] || dp[i][j-val]
			}
		}
	}

	return dp[length][sum]
}

//零钱兑换 II:https://leetcode-cn.com/problems/coin-change-2/
func change(amount int, coins []int) int {
	dp := make([][]int, len(coins)+1)
	for i := range dp {
		dp[i] = make([]int, amount+1)
		dp[i][0] = 1
	}

	for i, value := range coins {
		for j := 1; j <= amount; j++ {
			if j >= value {
				//for n := j; n >= 0; n -= value {
				//	dp[i+1][j] += dp[i][n]
				//}
				dp[i+1][j] = dp[i][j] + dp[i+1][j-value]
			} else {
				dp[i+1][j] += dp[i][j]
			}
		}
	}

	return dp[len(coins)][amount]
}

//反转链表 II:https://leetcode-cn.com/problems/reverse-linked-list-ii/
func reverseBetween(head *ListNode, m int, n int) *ListNode {
	var p, h *ListNode = nil, head
	for i := 0; i < m-1; i++ {
		if i == m-2 {
			p = h
		}
		h = h.Next
	}
	h1 := reverseList(h, n-m)
	if p != nil {
		p.Next = h1
	} else {
		head = h1 //说明翻转整个链表
	}
	return head
}

func reverseList(head *ListNode, n int) *ListNode {
	if n == 0 {
		return head
	}
	last := reverseList(head.Next, n-1)
	tmp := head.Next.Next
	head.Next.Next = head
	head.Next = tmp
	return last
}

//for循环翻转单个链表：https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/
func reverseList2(head *ListNode) *ListNode {
	var hh *ListNode = nil
	for h := head; h != nil; {
		tmp := h.Next
		h.Next = hh
		hh = h
		h = tmp
	}
	return hh
}

//回文链表：https://leetcode-cn.com/problems/palindrome-linked-list/
func isPalindrome(head *ListNode) bool {
	if head == nil {
		return true
	}
	mid, isEven := findMidNode(head)
	h := reverseListFromMid(head, mid)

	if !isEven {
		mid = mid.Next
	}
	for m, p := mid, h; m != nil; m, p = m.Next, p.Next {
		if m.Val != p.Val {
			return false
		}
	}

	return true
}

func findMidNode(head *ListNode) (*ListNode, bool) {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
	}
	if fast == nil {
		return slow, true
	}
	return slow, false
}

func reverseListFromMid(head *ListNode, mid *ListNode) *ListNode {
	h := mid
	for hh := head; hh != mid; {
		tmp := hh.Next
		hh.Next = h
		h = hh
		hh = tmp
	}
	return h
}

//翻转二叉树:https://leetcode-cn.com/problems/invert-binary-tree/
func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return root
	}
	invertTree(root.Left)
	invertTree(root.Right)
	root.Left, root.Right = root.Right, root.Left
	return root
}

type Node struct {
	Val   int
	Left  *Node
	Right *Node
	Next  *Node
}

//填充每个节点的下一个右侧节点指针：https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node/
func connect(root *Node) *Node {
	if root == nil {
		return root
	}
	for l, r := root.Left, root.Right; l != nil && r != nil; l, r = l.Right, r.Left {
		l.Next = r
	}

	connect(root.Left)
	connect(root.Right)
	return root
}

//二叉树展开为链表:https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/
func flatten(root *TreeNode) {
	if root == nil {
		return
	}
	flatten(root.Left)
	flatten(root.Right)
	tmp := root.Right
	root.Right = root.Left
	r := root
	for ; r.Right != nil; r = r.Right {
	}
	r.Right = tmp
	root.Left = nil
}

//最大二叉树：https://leetcode-cn.com/problems/maximum-binary-tree/
func constructMaximumBinaryTree(nums []int) *TreeNode {
	if len(nums) == 0 {
		return nil
	}

	var index, max = math.MinInt32, math.MinInt32
	for i, val := range nums {
		if val > max {
			max = val
			index = i
		}
	}

	left := constructMaximumBinaryTree(nums[:index])
	right := constructMaximumBinaryTree(nums[index+1:])

	return &TreeNode{Val: max, Left: left, Right: right}
}

//从前序与中序遍历序列构造二叉树:https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 {
		return nil
	}

	index := FindIntSliceIndex(inorder, preorder[0])
	left := buildTree(preorder[1:index+1], inorder[:index])
	right := buildTree(preorder[index+1:], inorder[index+1:])

	return &TreeNode{
		Val:   preorder[0],
		Left:  left,
		Right: right,
	}
}

//二叉搜索树中第K小的元素:https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/
func kthSmallest(root *TreeNode, k int) int {
	seq := inorderTraversal(root)
	return seq[k-1]
}

//把二叉搜索树转换为累加树:https://leetcode-cn.com/problems/convert-bst-to-greater-tree/
func convertBST(root *TreeNode) *TreeNode {
	seq := inorderTraversal(root)
	var sum int
	valToSum := make(map[int]int)
	for i := len(seq) - 1; i >= 0; i-- {
		sum += seq[i]
		valToSum[seq[i]] = sum
	}
	doConvertBST(root, valToSum)
	return root
}

func doConvertBST(root *TreeNode, valToSum map[int]int) {
	if root == nil {
		return
	}
	root.Val = valToSum[root.Val]
	doConvertBST(root.Left, valToSum)
	doConvertBST(root.Right, valToSum)
}

func convertBST2(root *TreeNode) *TreeNode {
	doConvertBST2(root, 0)
	return root
}

func doConvertBST2(root *TreeNode, sum int) int {
	if root == nil {
		return sum
	}

	root.Val += doConvertBST2(root.Right, sum)
	return doConvertBST2(root.Left, root.Val)
}

//删除二叉搜索树中的节点:https://leetcode-cn.com/problems/delete-node-in-a-bst/
func deleteNode(root *TreeNode, key int) *TreeNode {
	pre, n, isLeft := findBSTNode(root, key)
	if n == nil {
		return root
	}
	var son *TreeNode

	if n.Right == nil {
		son = n.Left
	} else if n.Right.Left == nil {
		son = n.Right
		son.Left = n.Left
	} else {
		left := n.Right
		leftMost := n.Right.Left
		for leftMost.Left != nil {
			leftMost = leftMost.Left
			left = left.Left
		}

		left.Left = leftMost.Right
		leftMost.Left, leftMost.Right = n.Left, n.Right
		son = leftMost
	}

	if pre != nil {
		if isLeft {
			pre.Left = son
		} else {
			pre.Right = son
		}
	} else {
		root = son //如果查找恰好是根节点，那么根节点需要改变
	}

	return root
}

func findBSTNode(root *TreeNode, key int) (*TreeNode, *TreeNode, bool) {
	var pre *TreeNode
	var isLeft bool
	for root != nil {
		if root.Val == key {
			return pre, root, isLeft
		} else if root.Val > key {
			pre = root
			isLeft = true
			root = root.Left
		} else if root.Val < key {
			pre = root
			isLeft = false
			root = root.Right
		}
	}
	return nil, nil, isLeft
}

//二叉搜索树中的插入操作:https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/
func insertIntoBST(root *TreeNode, val int) *TreeNode {
	if root == nil {
		return &TreeNode{Val: val}
	}
	if root.Val < val {
		root.Right = insertIntoBST(root.Right, val)
	} else {
		root.Left = insertIntoBST(root.Left, val)
	}
	return root
}

func insertIntoBST2(root *TreeNode, val int) *TreeNode {
	if root == nil {
		return &TreeNode{Val: val}
	}
	if root.Left == nil && root.Val > val {
		root.Left = &TreeNode{Val: val}
		return root
	}
	if root.Right == nil && root.Val < val {
		root.Right = &TreeNode{Val: val}
		return root
	}
	if root.Val < val {
		insertIntoBST2(root.Right, val)
	} else {
		insertIntoBST2(root.Left, val)
	}
	return root
}

//扁平化嵌套列表迭代器:https://leetcode-cn.com/problems/flatten-nested-list-iterator/
//递归解法，比较慢
//type NestedIterator struct {
//	list []int
//	loc  int
//}
//
//func iteration(nestedList []*NestedInteger) []int {
//	var res []int
//	for _, l := range nestedList {
//		if l.IsInteger() {
//			res = append(res, l.GetInteger())
//			continue
//		}
//		res = append(res, iteration(l.GetList())...)
//	}
//	return res
//}
//
//func Constructor(nestedList []*NestedInteger) *NestedIterator {
//	return &NestedIterator{list: iteration(nestedList)}
//}
//
//func (this *NestedIterator) Next() int {
//	if this.HasNext() {
//		this.loc++
//		return this.list[this.loc-1]
//	}
//	return -1
//}
//
//func (this *NestedIterator) HasNext() bool {
//	if this.loc < len(this.list) {
//		return true
//	}
//
//	return false
//}

//惰性取数据
//type NestedIterator struct {
//	list []*NestedInteger
//}
//
//func Constructor(nestedList []*NestedInteger) *NestedIterator {
//	return &NestedIterator{list: nestedList}
//}
//
//func (this *NestedIterator) Next() int {
//	res := this.list[0].GetInteger()
//	this.list = this.list[1:]
//	return res
//}
//
//func (this *NestedIterator) HasNext() bool {
//	for len(this.list) > 0 && !this.list[0].IsInteger() {
//		ll := this.list[0].GetList()
//		this.list = append(ll, this.list[1:]...)
//	}
//
//	return len(this.list) != 0
//}

//二叉树的最近公共祖先:https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}

	left := lowestCommonAncestor(root.Left, p, q)
	right := lowestCommonAncestor(root.Right, p, q)

	if left != nil && right != nil {
		return root
	}
	if root.Val == p.Val || root.Val == q.Val {
		return root
	}
	if left != nil {
		return left
	}

	if right != nil {
		return right
	}

	return nil
}

//完全二叉树的节点个数：https://leetcode-cn.com/problems/count-complete-tree-nodes/
func countNodes(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return countNodes(root.Left) + countNodes(root.Right) + 1
}

//LRU缓存机制：https://leetcode-cn.com/problems/lru-cache/
type LRUCache struct {
	cache      map[int]*LRUnode
	capacity   int
	head, tail *LRUnode
}
type LRUnode struct {
	val  int
	key  int
	pre  *LRUnode
	next *LRUnode
}

func Constructor(capacity int) LRUCache {
	cache := make(map[int]*LRUnode)
	return LRUCache{
		cache:    cache,
		capacity: capacity,
	}
}

func (this *LRUCache) Get(key int) int {
	n := this.cache[key]
	if n == nil {
		return -1
	}

	this.delOneNode(n)
	this.insertNode(n, nil)
	return n.val

}

func (this *LRUCache) Put(key int, value int) {
	if n, ok := this.cache[key]; ok {
		n.val = value
		this.delOneNode(n)
		this.insertNode(n, nil)
		return
	}
	if len(this.cache) >= this.capacity {
		n := this.delOneNode(this.head)
		delete(this.cache, n.key)
	}

	nn := &LRUnode{
		val: value,
		key: key,
	}
	this.insertNode(nn, nil)
	this.cache[key] = nn
}

//双端链表删除一个节点,不删除key,返回被删除的node
func (this *LRUCache) delOneNode(n *LRUnode) *LRUnode {
	if n.pre == nil {
		this.head = n.next
	} else {
		n.pre.next = n.next
	}

	if n.next == nil {
		this.tail = n.pre
	} else {
		n.next.pre = n.pre
	}

	n.pre, n.next = nil, nil
	return n
}

///双端链表某个node前插入一个节点，loc为nil则表示尾部插入
func (this *LRUCache) insertNode(n *LRUnode, loc *LRUnode) {
	if loc == nil {
		if this.tail != nil {
			this.tail.next = n
			n.pre = this.tail
		}
		this.tail = n
		if this.head == nil {
			this.head = n
		}
	} else {
		if loc.pre != nil {
			loc.pre.next = n
			n.pre = loc.pre
		} else {
			this.head = n
		}
		loc.pre = n
		n.next = loc
	}
}

//LFU缓存机制：https://leetcode-cn.com/problems/lfu-cache/
type LFUCache struct {
	cache      map[int]*LFUnode
	capacity   int
	head, tail *LFUnode
}
type LFUnode struct {
	val   int
	key   int
	count int //访问次数
	pre   *LFUnode
	next  *LFUnode
}

func LFUConstructor(capacity int) LFUCache {
	cache := make(map[int]*LFUnode)
	return LFUCache{
		cache:    cache,
		capacity: capacity,
	}
}

func (this *LFUCache) Get(key int) int {
	n := this.cache[key]
	if n == nil {
		return -1
	}
	this.reSort(n)
	return n.val

}

func (this *LFUCache) reSort(n *LFUnode) {
	n.count++

	var end *LFUnode
	for end = n.next; end != nil && end.count <= n.count; end = end.next {
	}

	this.delOneNode(n)
	this.insertNode(n, end)
}

func (this *LFUCache) Put(key int, value int) {
	if n, ok := this.cache[key]; ok {
		n.val = value
		this.reSort(n)
		return
	}
	if this.capacity == 0 {
		return
	}
	if len(this.cache) >= this.capacity {
		n := this.delOneNode(this.head)
		delete(this.cache, n.key)
	}

	nn := &LFUnode{
		val: value,
		key: key,
	}
	this.insertNode(nn, this.head)
	this.reSort(nn)
	this.cache[key] = nn
}

//双端链表删除一个节点,不删除key,返回被删除的node
func (this *LFUCache) delOneNode(n *LFUnode) *LFUnode {
	if n.pre == nil {
		this.head = n.next
	} else {
		n.pre.next = n.next
	}

	if n.next == nil {
		this.tail = n.pre
	} else {
		n.next.pre = n.pre
	}

	n.pre, n.next = nil, nil
	return n
}

///双端链表某个node前插入一个节点，loc为nil则表示尾部插入
func (this *LFUCache) insertNode(n *LFUnode, loc *LFUnode) {
	if loc == nil {
		if this.tail != nil {
			this.tail.next = n
			n.pre = this.tail
		}
		this.tail = n
		if this.head == nil {
			this.head = n
		}
	} else {
		if loc.pre != nil {
			loc.pre.next = n
			n.pre = loc.pre
		} else {
			this.head = n
		}
		loc.pre = n
		n.next = loc
	}
}

//被围绕的区域:https://leetcode-cn.com/problems/surrounded-regions/
func solve(board [][]byte) {
	for i, b := range board {
		for j := range b {
			if i == 0 || i == len(board)-1 || j == 0 || j == len(board[0])-1 {
				DFSBoundary(board, i, j)
			}
		}
	}

	for i, b := range board {
		for j := range b {
			if board[i][j] == 'O' {
				board[i][j] = 'X'
			}
			if board[i][j] == '-' {
				board[i][j] = 'O'
			}
		}
	}
}

func beyondBoundary(board [][]byte, i, j int) bool {
	if i < 0 || i >= len(board) || j < 0 || j >= len(board[0]) {
		return true
	}
	return false
}

func DFSBoundary(board [][]byte, i, j int) {
	if beyondBoundary(board, i, j) || board[i][j] != 'O' {
		return
	}

	board[i][j] = '-'
	DFSBoundary(board, i+1, j)
	DFSBoundary(board, i-1, j)
	DFSBoundary(board, i, j+1)
	DFSBoundary(board, i, j-1)
}

//数据流的中位数:https://leetcode-cn.com/problems/find-median-from-data-stream/
type PriorityQueue struct {
	list      []int
	isBigHeap bool
}

func (p *PriorityQueue) length() int {
	return len(p.list)
}
func (p *PriorityQueue) addNum(num int) {
	p.list = append(p.list, num)
	var k, preK = len(p.list)/2 - 1, len(p.list) - 1
	for ; k >= 0; k, preK = (k+1)/2-1, k {
		if p.isBigHeap {
			if p.list[k] < num {
				p.list[preK] = p.list[k]
				continue
			}
		} else {
			if p.list[k] > num {
				p.list[preK] = p.list[k]
				continue

			}
		}
		break
	}
	p.list[preK] = num
}
func (p *PriorityQueue) getHead() int {
	if len(p.list) == 0 {
		return -1
	}
	return p.list[0]
}

func (p *PriorityQueue) fetchHead() int {
	if len(p.list) == 0 {
		return -1
	}
	tmp := p.list[0]
	p.list[0] = p.list[len(p.list)-1]
	p.list = p.list[:len(p.list)-1]
	p.adjustHead()
	return tmp
}

func (p *PriorityQueue) adjustHead() {
	length := len(p.list)
	if length == 0 {
		return
	}

	tmp := p.list[0]
	var k, preK = 1, 0
	for ; k < length; k, preK = 2*k+1, k {
		if p.isBigHeap {
			if k+1 < length && p.list[k] < p.list[k+1] {
				k++
			}
			if p.list[k] > tmp {
				p.list[preK] = p.list[k]
				continue
			}
		} else {
			if k+1 < length && p.list[k] > p.list[k+1] {
				k++
			}
			if p.list[k] < tmp {
				p.list[preK] = p.list[k]
				continue
			}
		}
		break
	}
	p.list[preK] = tmp
}

type MedianFinder struct {
	smallQ, biggerQ PriorityQueue
}

/** initialize your data structure here. */
func MedianFinderConstructor() MedianFinder {
	return MedianFinder{
		smallQ:  PriorityQueue{},
		biggerQ: PriorityQueue{isBigHeap: true},
	}
}

func (this *MedianFinder) AddNum(num int) {
	if this.smallQ.length() < this.biggerQ.length() {
		this.biggerQ.addNum(num)
		this.smallQ.addNum(this.biggerQ.fetchHead())
	} else {
		this.smallQ.addNum(num)
		this.biggerQ.addNum(this.smallQ.fetchHead())
	}
}

func (this *MedianFinder) FindMedian() float64 {
	if this.biggerQ.length() > this.smallQ.length() {
		return float64(this.biggerQ.getHead())
	} else {
		return float64(this.biggerQ.getHead()+this.smallQ.getHead()) / 2
	}
}

//设计推特:https://leetcode-cn.com/problems/design-twitter/
type Twitter struct {
	followMap  map[int]map[int]bool
	twitterMap map[int][]Tweet
	count      int64
}

type Tweet struct {
	id        int
	timeStamp int64
}

/** Initialize your data structure here. */
func TwitterConstructor() Twitter {
	return Twitter{
		followMap:  make(map[int]map[int]bool),
		twitterMap: make(map[int][]Tweet),
	}
}

/** Compose a new tweet. */
func (this *Twitter) PostTweet(userId int, tweetId int) {
	tweets := this.twitterMap[userId]
	tweets = append(tweets, Tweet{tweetId, this.count})
	this.count++
	this.twitterMap[userId] = tweets
}

/** Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent. */
func (this *Twitter) GetNewsFeed(userId int) []int {
	var tweets []Tweet
	tt := this.twitterMap[userId]
	if len(tt) > 10 {
		tt = tt[len(tt)-10:]
	}
	tweets = append(tweets, tt...)

	for uid := range this.followMap[userId] {
		tt := this.twitterMap[uid]
		if len(tt) > 10 {
			tt = tt[len(tt)-10:]
		}
		tweets = append(tweets, tt...)
	}

	sort.Slice(tweets, func(i, j int) bool {
		return tweets[i].timeStamp > tweets[j].timeStamp
	})

	if len(tweets) > 10 {
		tweets = tweets[:10]
	}

	var resId []int
	for _, tt := range tweets {
		resId = append(resId, tt.id)
	}

	return resId

}

/** Follower follows a followee. If the operation is invalid, it should be a no-op. */
func (this *Twitter) Follow(followerId int, followeeId int) {
	if followeeId == followerId {
		return
	}
	follower := this.followMap[followerId]
	if follower == nil {
		this.followMap[followerId] = make(map[int]bool)
		follower = this.followMap[followerId]
	}
	follower[followeeId] = true
}

/** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
func (this *Twitter) Unfollow(followerId int, followeeId int) {
	follower := this.followMap[followerId]
	if follower == nil {
		return
	}

	delete(follower, followeeId)
}

//下一个更大元素 I:https://leetcode-cn.com/problems/next-greater-element-i/
func nextGreaterElement(nums1 []int, nums2 []int) []int {
	var numToRes = make(map[int]int)
	for _, n := range nums1 {
		numToRes[n] = math.MinInt32
	}

	for _, val := range nums2 {
		if _, ok := numToRes[val]; ok {
			numToRes[val] = -1
		}

		for key, rr := range numToRes {
			if rr == math.MinInt32 {
				continue
			}
			if rr == -1 && val > key {
				numToRes[key] = val
			}
		}

	}

	var res []int
	for _, val := range nums1 {
		res = append(res, numToRes[val])
	}

	return res
}

//下一个更大元素 II:https://leetcode-cn.com/problems/next-greater-element-ii/
//使用单调栈
func nextGreaterElements(nums []int) []int {
	var res = make([]int, len(nums))
	var stack []int
	length := len(nums)
	for k := 2*len(nums) - 1; k >= 0; k-- {
		i := k % length
		num := nums[i]
		for len(stack) > 0 && num >= stack[len(stack)-1] {
			stack = stack[:len(stack)-1]
		}

		if len(stack) == 0 {
			res[i] = -1
		} else {
			res[i] = stack[len(stack)-1]
		}

		stack = append(stack, num)
	}
	return res
}

//滑动窗口最大值:https://leetcode-cn.com/problems/sliding-window-maximum/
//使用单调队列

type IncreasingQueue struct {
	head   *Qnode
	tail   *Qnode
	k      int
	length int
}

type Qnode struct {
	val  int
	next *Qnode
	pre  *Qnode
}

func (q *IncreasingQueue) add(val int, first int) {
	if q.length == q.k && q.get() == first {
		q.poll()
	}
	m := q.head
	for ; m != nil && m.val < val; m = m.next { //不能用<=
	}
	tmp := Qnode{
		val:  val,
		next: m,
	}
	if m != nil {
		m.pre = &tmp
	} else {
		q.tail = &tmp
	}

	q.head = &tmp
	if q.length < q.k {
		q.length++
	}
}

func (q *IncreasingQueue) get() int {
	if q.tail != nil {
		return q.tail.val
	} else {
		return -1
	}
}

func (q *IncreasingQueue) poll() int {
	if q.tail != nil {
		tmp := q.tail
		if q.tail.pre != nil {
			q.tail.pre.next = nil
		}
		q.tail = q.tail.pre
		if q.tail == nil {
			q.head = nil
		}
		q.length--
		return tmp.val
	}
	return -1
}
func maxSlidingWindow(nums []int, k int) []int {
	quene := IncreasingQueue{k: k}
	for _, val := range nums[:k-1] {
		quene.add(val, val)
	}
	var res []int
	for i, value := range nums[k-1:] {
		if i-1 < 0 {
			quene.add(value, 0)
		} else {
			quene.add(value, nums[i-1])
		}
		res = append(res, quene.get())
	}
	return res
}

//用栈实现队列:https://leetcode-cn.com/problems/implement-queue-using-stacks/
type MyQueue struct {
	fStack, sStack []int
}

/** Initialize your data structure here. */
func MyQueueConstructor() MyQueue {
	return MyQueue{}
}

/** Push element x to the back of queue. */
func (this *MyQueue) Push(x int) {
	this.fStack = append(this.fStack, x)
}

/** Removes the element from in front of queue and returns that element. */
func (this *MyQueue) Pop() int {
	if len(this.sStack) == 0 {
		this.copyStack()
	}
	tmp := this.sStack[len(this.sStack)-1]
	this.sStack = this.sStack[:len(this.sStack)-1]
	return tmp
}

func (this *MyQueue) copyStack() {
	for i := 0; i < len(this.fStack)/2; i++ {
		this.fStack[i], this.fStack[len(this.fStack)-1-i] = this.fStack[len(this.fStack)-1-i], this.fStack[i]
	}
	this.fStack, this.sStack = this.sStack, this.fStack
}

/** Get the front element. */
func (this *MyQueue) Peek() int {
	if len(this.sStack) == 0 {
		this.copyStack()
	}
	return this.sStack[len(this.sStack)-1]
}

/** Returns whether the queue is empty. */
func (this *MyQueue) Empty() bool {
	return len(this.sStack) == 0 && len(this.fStack) == 0
}

//爱吃香蕉的珂珂:https://leetcode-cn.com/problems/koko-eating-bananas/
func minEatingSpeed(piles []int, H int) int {
	max := MaxSliceValue(piles)

	var i, j = 1, max
	for ; i < j-1; {
		mid := (i + j) / 2
		res := calHours(piles, mid)
		fmt.Println(i, j, mid, res)
		if res > H {
			i = mid + 1
		} else if res <= H {
			j = mid
		}
	}

	if calHours(piles, i) <= H {
		return i
	} else {
		return j
	}
}

func calHours(piles []int, speed int) (H int) {
	for _, p := range piles {
		if p%speed == 0 {
			H += p / speed
		} else {
			H += p/speed + 1
		}
	}
	return
}

//环形链表:https://leetcode-cn.com/problems/linked-list-cycle/
func hasCycle(head *ListNode) bool {
	var slow, fast = head, head
	var hasCycel bool
	for fast != nil {
		if fast.Next != nil {
			fast = fast.Next.Next
		} else {
			break
		}
		slow = slow.Next
		if fast == slow {
			hasCycel = true
			break
		}
	}
	return hasCycel
}

//环形链表 II:https://leetcode-cn.com/problems/linked-list-cycle-ii/
func detectCycle(head *ListNode) *ListNode {
	n := meetNode(head)
	if nil == n {
		return nil
	}

	for h := head; h != n; h, n = h.Next, n.Next {
	}
	return n
}

func meetNode(head *ListNode) *ListNode {
	var slow, fast = head, head
	var hasCycel bool
	for fast != nil {
		if fast.Next != nil {
			fast = fast.Next.Next
		} else {
			break
		}
		slow = slow.Next
		if fast == slow {
			hasCycel = true
			break
		}
	}
	if hasCycel {
		return slow
	} else {
		return nil
	}
}

//删除链表的倒数第N个节点：https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	var h, e = head, head
	i := 0
	for ; i < n; i++ {
		e = e.Next
	}
	if e == nil {
		return head.Next
	}

	for ; e.Next != nil; h, e = h.Next, e.Next {
	}
	h.Next = h.Next.Next
	return head
}

//两数之和 II - 输入有序数组：https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/
func calTwoSum(numbers []int, target int) (res []int) {
	var m, n = 0, len(numbers) - 1
	for m < n {
		sum := numbers[m] + numbers[n]
		if sum > target {
			n--
		} else if sum < target {
			m++
		} else {
			break
		}

	}
	res = append(res, m+1, n+1)
	return
}

//字符串的排列：https://leetcode-cn.com/problems/permutation-in-string/
func checkInclusion(s1 string, s2 string) bool {
	if len(s1) > len(s2) {
		return false
	}
	oriChToNum := make(map[uint8]int)
	resChToNum := make(map[uint8]int)

	for i := range s1 {
		oriChToNum[s1[i]]++
	}

	for i := range s2 {
		if i < len(s1)-1 {
			resChToNum[s2[i]]++
		} else {
			resChToNum[s2[i]]++
			if i != len(s1)-1 {
				resChToNum[s2[i-len(s1)]]--
				if resChToNum[s2[i-len(s1)]] == 0 {
					delete(resChToNum, s2[i-len(s1)])
				}
			}

			if _, ok := oriChToNum[s2[i]]; ok && isInclude(oriChToNum, resChToNum) {
				return true
			}

		}
	}

	return false
}

func isInclude(ori, res map[uint8]int) bool {
	for key, val := range ori {
		if res[key] != val {
			return false
		}
	}
	return true
}

//无重复字符的最长子串：https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/
func lengthOfLongestSubstring2(s string) int {
	if len(s) <= 0 {
		return 0
	}
	var max int
	chToNum := make(map[uint8]int)

	var i, j = 0, 0
	for {
		for ; j < len(s); j++ {
			chToNum[s[j]]++
			if chToNum[s[j]] > 1 {
				if len(chToNum) > max {
					max = len(chToNum)
				}
				j++
				break
			}
		}

		if j >= len(s) {
			break
		}

		for ; i < j; i++ {
			chToNum[s[i]]--
			if chToNum[s[i]] == 0 {
				delete(chToNum, s[i])
			} else {
				i++
				break
			}
		}
	}

	if len(chToNum) > max {
		max = len(chToNum)
	}
	return max
}

//常数时间插入、删除和获取随机元素：https://leetcode-cn.com/problems/insert-delete-getrandom-o1/
type RandomizedSet struct {
	valMap   map[int]*setNode
	nodeList []*setNode
}

type setNode struct {
	val   int
	index int
}

/** Initialize your data structure here. */
func RandomizedSetConstructor() RandomizedSet {
	return RandomizedSet{
		valMap: make(map[int]*setNode),
	}
}

/** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
func (this *RandomizedSet) Insert(val int) bool {
	if _, ok := this.valMap[val]; ok {
		return false
	}
	n := setNode{
		val:   val,
		index: len(this.nodeList),
	}
	this.valMap[val] = &n
	this.nodeList = append(this.nodeList, &n)
	return true
}

/** Removes a value from the set. Returns true if the set contained the specified element. */
func (this *RandomizedSet) Remove(val int) bool {
	if _, ok := this.valMap[val]; !ok {
		return false
	}
	index := this.valMap[val].index
	delete(this.valMap, val)
	this.nodeList[index] = this.nodeList[len(this.nodeList)-1]
	this.nodeList[index].index = index
	this.nodeList = this.nodeList[:len(this.nodeList)-1]
	return true
}

/** Get a random element from the set. */
func (this *RandomizedSet) GetRandom() int {
	rand.Seed(time.Now().UnixNano())
	return this.nodeList[rand.Intn(len(this.nodeList))].val
}

//黑名单中的随机数:https://leetcode-cn.com/problems/random-pick-with-blacklist/
type Solution struct {
	N          int
	blacklist  []int
	blackToVal map[int]int
}

func SolutionConstructor(N int, blacklist []int) Solution {
	blackToVal := make(map[int]int)
	for _, black := range blacklist {
		blackToVal[black] = black
	}
	var j, sz = N - 1, N - len(blacklist)

	for _, black := range blacklist {
		if black >= sz {
			continue
		}

		for {
			if _, ok := blackToVal[j]; !ok {
				break
			}
			j--
		}
		blackToVal[black] = j
		j--
	}

	return Solution{
		N:          N,
		blacklist:  blacklist,
		blackToVal: blackToVal,
	}
}

func (this *Solution) Pick() int {
	rand.Seed(time.Now().UnixNano())
	val := rand.Intn(this.N - len(this.blacklist))
	if _, ok := this.blackToVal[val]; ok {
		return this.blackToVal[val]
	} else {
		return val
	}
}

//删除排序数组中的重复项：https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/
func removeDuplicates(nums []int) int {
	sort.Ints(nums)
	var count int
	for i := 0; i < len(nums); i++ {
		if i+1 < len(nums) && nums[i] == nums[i+1] {
			count++
			nums[i] = math.MinInt32
		}
	}

	flag := true
	for i, j := 0, len(nums)-1; i <= j; {
		if flag {
			if nums[j] != math.MinInt32 {
				flag = false
			}
			j--
		} else {
			if nums[i] == math.MinInt32 {
				nums[i], nums[j+1] = nums[j+1], nums[i]
				flag = true
			}
			i++
		}
	}
	sort.Ints(nums[:len(nums)-count])
	return len(nums) - count
}

func removeElement(nums []int, val int) int {
	var count int
	for i, j := 0, 0; j < len(nums); {
		if nums[j] != val {
			nums[i] = nums[j]
			i++
			count++
		}
		j++
	}
	return count
}

//编辑距离：https://leetcode-cn.com/problems/edit-distance/
func minDistance(word1 string, word2 string) int {
	memo := make([][]int, len(word1))
	for i := range memo {
		memo[i] = make([]int, len(word2))
		for j := range memo[i] {
			memo[i][j] = -1
		}
	}
	return minDistanceDp(word1, word2, memo, len(word1)-1, len(word2)-1)
}

func minDistanceDp(word1, world2 string, memo [][]int, i, j int) int {
	if i == -1 {
		return j + 1
	} else if j == -1 {
		return i + 1
	}
	if memo[i][j] > -1 {
		return memo[i][j]
	}

	if word1[i] == world2[j] {
		return minDistanceDp(word1, world2, memo, i-1, j-1)
	} else {
		v1 := minDistanceDp(word1, world2, memo, i, j-1) + 1
		v2 := minDistanceDp(word1, world2, memo, i-1, j) + 1
		v3 := minDistanceDp(word1, world2, memo, i-1, j-1) + 1
		vv := Min(v1, v2)
		vv = Min(vv, v3)
		memo[i][j] = vv
		return vv
	}
}

//目标和：https://leetcode-cn.com/problems/target-sum/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china
func findTargetSumWays(nums []int, S int) (res int) {
	if len(nums) == 0 {
		if S == 0 {
			return 1
		} else {
			return 0
		}
	}

	length := len(nums)
	res = findTargetSumWays(nums[:length-1], S+nums[length-1])
	res += findTargetSumWays(nums[:length-1], S-nums[length-1])
	return res
}

//俄罗斯套娃信封问题:https://leetcode-cn.com/problems/russian-doll-envelopes/
//超时了
func maxEnvelopes(envelopes [][]int) int {
	if len(envelopes) == 0 {
		return 0
	}

	sort.Slice(envelopes, func(i, j int) bool {
		if envelopes[i][0] != envelopes[j][0] {
			return envelopes[i][0] < envelopes[j][0]
		} else {
			return envelopes[i][1] > envelopes[j][1]
		}
	})
	var list []int
	for _, n := range envelopes {
		list = append(list, n[1])
	}

	return lengthOfLIS(list)
}

//最长上升子序列:https://leetcode-cn.com/problems/longest-increasing-subsequence/
func lengthOfLIS(nums []int) int {
	dp := make([]int, len(nums))
	dp[0] = 1
	for i, n := range nums[1:] {
		m := i + 1
		var max = 1
		for j, k := range nums[:m] {
			if n > k {
				max = Max(max, dp[j]+1)
			}
		}
		dp[m] = max
	}

	return MaxSliceValue(dp)
}

//最大子序和:https://leetcode-cn.com/problems/maximum-subarray/
func maxSubArray(nums []int) int {
	var dp = make([]int, len(nums))
	dp[0] = nums[0]
	for i, v := range nums[1:] {
		k := i + 1
		if dp[k-1]+v > v {
			dp[k] = dp[k-1] + v
		} else {
			dp[k] = v
		}
	}
	return MaxSliceValue(dp)
}

//最长公共子序列:https://leetcode-cn.com/problems/longest-common-subsequence/
func longestCommonSubsequence(text1 string, text2 string) int {
	dp := make([][]int, len(text1)+1)
	for i := range dp {
		dp[i] = make([]int, len(text2)+1)
	}

	for i := 0; i < len(text1); i++ {
		for j := 0; j < len(text2); j++ {
			if text1[i] == text2[j] {
				dp[i+1][j+1] = dp[i][j] + 1
			} else {
				dp[i+1][j+1] = Max(dp[i][j+1], dp[i+1][j])
			}
		}
	}

	return dp[len(text1)][len(text2)]
}

//两个字符串的删除操作:https://leetcode-cn.com/problems/delete-operation-for-two-strings/
func minStringDistance(word1 string, word2 string) int {
	dp := make([][]int, len(word1)+1)
	for i := range dp {
		dp[i] = make([]int, len(word2)+1)
		dp[i][0] = i
	}
	for i := range dp[0] {
		dp[0][i] = i
	}

	for i := 1; i <= len(word1); i++ {
		for j := 1; j <= len(word2); j++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				dp[i][j] = Min(dp[i-1][j]+1, dp[i][j-1]+1)
			}
		}
	}

	return dp[len(word1)][len(word2)]
}

//两个字符串的最小ASCII删除和：https://leetcode-cn.com/problems/minimum-ascii-delete-sum-for-two-strings/
func minimumDeleteSum(word1 string, word2 string) int {
	dp := make([][]int, len(word1)+1)
	for i := range dp {
		dp[i] = make([]int, len(word2)+1)
		dp[i][0] = SumAscii(word1[:i])
	}
	for i := range dp[0] {
		dp[0][i] = SumAscii(word2[:i])
	}

	for i := 1; i <= len(word1); i++ {
		for j := 1; j <= len(word2); j++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				dp[i][j] = Min(dp[i-1][j]+int(word1[i-1]), dp[i][j-1]+int(word2[j-1]))
			}
		}
	}

	return dp[len(word1)][len(word2)]
}

func SumAscii(word1 string) int {
	sum := 0
	for _, val := range word1 {
		sum += int(val)
	}
	return sum
}

//无重叠区间:https://leetcode-cn.com/problems/non-overlapping-intervals/
func eraseOverlapIntervals(intervals [][]int) int {
	sort.Slice(intervals, func(i, j int) bool {
		if intervals[i][0] != intervals[j][0] {
			return intervals[i][0] < intervals[j][0]
		} else {
			return intervals[i][1] < intervals[j][1]
		}
	})

	dp := make([]int, len(intervals)+1)
	dp[0] = 0

	for i := 1; i <= len(intervals); i++ {
		max := 1
		for j := 1; j < i; j++ {
			if intervals[i-1][0] >= intervals[j-1][1] {
				max = Max(dp[j]+1, max)
			}
		}
		dp[i] = max
	}

	return len(intervals) - MaxSliceValue(dp)
}

func eraseOverlapIntervals2(intervals [][]int) int {
	if len(intervals) == 0 {
		return 0
	}
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][1] < intervals[j][1]
	})
	first := intervals[0]
	count := 0

	for i := 1; i < len(intervals); i++ {
		if intervals[i][0] >= first[1] {
			first = intervals[i]
		} else {
			count++
		}
	}

	return count
}

//用最少数量的箭引爆气球:https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons/
func findMinArrowShots(points [][]int) int {
	if len(points) == 0 {
		return 0
	}
	sort.Slice(points, func(i, j int) bool {
		return points[i][1] < points[j][1]
	})

	var count int = 1
	first := points[0]

	for i := 1; i < len(points); i++ {
		if points[i][0] <= first[1] {
			continue
		} else {
			first = points[i]
			count++
		}
	}

	return count
}

//跳跃游戏：https://leetcode-cn.com/problems/jump-game/
func canJump(nums []int) bool {
	memo := make([]int, len(nums))
	for i := 0; i < len(nums); i++ {
		memo[i] = -1
	}

	return checkCanJump(nums, 0, memo)
}

func checkCanJump(nums []int, start int, memo []int) bool {
	if start == len(nums)-1 {
		return true
	} else if start >= len(nums)-1 {
		return false
	}
	if memo[start] == 0 {
		return false
	}

	for i := 1; i <= nums[start]; i++ {
		if checkCanJump(nums, start+i, memo) {
			return true
		}
	}
	memo[start] = 0
	return false
}

//跳跃游戏 II：https://leetcode-cn.com/problems/jump-game-ii/
func jump(nums []int) int {
	dp := make([]int, len(nums))

	dp[0] = 0

	for i := 1; i < len(nums); i++ {

		var min = math.MaxInt32
		for j := 0; j < i; j++ {
			if j+nums[j] >= i {
				min = Min(min, dp[j]+1)
			}
			dp[i] = min
		}
	}

	return dp[len(nums)-1]
}

//预测赢家:https://leetcode-cn.com/problems/predict-the-winner/
func PredictTheWinner(nums []int) bool {
	dp := make([][][2]int, len(nums))

	for i := 0; i < len(nums); i++ {
		dp[i] = make([][2]int, len(nums))
		dp[i][i] = [2]int{nums[i], 0}
	}

	for i := 1; i < len(nums); i++ {
		for k, j := 0, i; k < len(nums) && j < len(nums); k, j = k+1, j+1 {
			left := nums[k] + dp[k+1][j][1]
			right := nums[j] + dp[k][j-1][1]
			if left > right {
				dp[k][j][0] = left
				dp[k][j][1] = dp[k+1][j][0]
			} else {
				dp[k][j][0] = right
				dp[k][j][1] = dp[k][j-1][0]
			}

		}
	}
	fmt.Println(dp)
	res := dp[0][len(nums)-1]
	return res[0] >= res[1]
}

//让字符串成为回文串的最少插入次数:https://leetcode-cn.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/
func minInsertions(s string) int {
	dp := make([][]int, len(s))

	for i := 0; i < len(s); i++ {
		dp[i] = make([]int, len(s))
		for j := 0; j < len(s); j++ {
			dp[i][j] = -1
		}
		dp[i][i] = 0
	}

	return minInsertionsDp(s, 0, len(s)-1, dp)
}

func minInsertionsDp(s string, i, j int, dp [][]int) int {
	if i >= j {
		return 0
	}
	if dp[i][j] >= 0 {
		return dp[i][j]
	}

	var min int
	if s[i] == s[j] {
		min = minInsertionsDp(s, i+1, j-1, dp)
	} else {
		min = Min(minInsertionsDp(s, i, j-1, dp)+1, minInsertionsDp(s, i+1, j, dp)+1)
	}
	dp[i][j] = min

	return min
}

//子集:https://leetcode-cn.com/problems/subsets/
func subsets(nums []int) [][]int {
	memo := make(map[int][][]int)

	res := [][]int{{}}
	for i := 0; i < len(nums); i++ {
		res = append(res, subsetsDp(nums[i:], i, memo)...)
	}

	return res
}

func subsetsDp(nums []int, index int, memo map[int][][]int) [][]int {
	if val, ok := memo[index]; ok {
		return val
	}
	var res [][]int
	first := nums[0]
	res = append(res, []int{first})

	for j := 1; j < len(nums); j++ {
		rr := subsetsDp(nums[j:], index+j, memo)
		for i := range rr {
			rrr := append([]int{first}, rr[i]...)
			res = append(res, rrr)
		}
	}

	memo[index] = res
	return res
}

//组合：https://leetcode-cn.com/problems/combinations/
func combine(n int, k int) [][]int {
	var res [][]int

	findCombine(1, n, k, []int{}, &res)
	return res
}

func findCombine(e, n, k int, trace []int, res *[][]int) {
	if len(trace) == k {
		rr := append([]int{}, trace...)
		*res = append(*res, rr)
	}
	if e > n {
		return
	}

	for i := e; i <= n; i++ {
		trace = append(trace, i)
		findCombine(i+1, n, k, trace, res)
		trace = trace[:len(trace)-1]
	}

	return
}

//解数独：https://leetcode-cn.com/problems/sudoku-solver/
func solveSudoku(board [][]byte) {
	solveSudokuDp(board, 0, 0)
}

func solveSudokuDp(board [][]byte, i, j int) bool {
	for {
		if j == len(board[0]) {
			j = 0
			i++
		}
		if i >= len(board) {
			return true
		}
		if board[i][j] == '.' {
			break
		}
		j++
	}

	for _, avail := range availNum(board, i, j) {
		board[i][j] = avail
		if solveSudokuDp(board, i, j+1) {
			return true
		}
		board[i][j] = '.'
	}

	return false
}

func availNum(board [][]byte, i, j int) []byte {
	rr1 := notIn1_9(board[i])

	var rr2 []byte
	for k := 0; k < len(board); k++ {
		rr2 = append(rr2, board[k][j])
	}

	rr2 = notIn1_9(rr2)

	var rr3 [] byte

	m, n := i/3*3, j/3*3
	rr3 = append(rr3, board[m][n:n+3]...)
	rr3 = append(rr3, board[m+1][n:n+3]...)
	rr3 = append(rr3, board[m+2][n:n+3]...)

	rr3 = notIn1_9(rr3)

	return intersection(rr3, intersection(rr1, rr2))

}

func intersection(a, b []byte) []byte {
	var m = make(map[byte]bool)
	for _, n := range a {
		m[n] = true
	}
	var res []byte
	for _, n := range b {
		if _, ok := m[n]; ok {
			res = append(res, n)
		}
	}

	return res
}

func notIn1_9(nums []byte) []byte {

	var s = []byte{'1', '2', '3', '4', '5', '6', '7', '8', '9'}

	var m = make(map[byte]bool)
	for _, n := range nums {
		m[n] = true
	}

	var res []byte
	for _, ss := range s {
		if _, ok := m[ss]; !ok {
			res = append(res, ss)
		}
	}

	return res
}

//括号生成：https://leetcode-cn.com/problems/generate-parentheses/
func generateParenthesis(n int) []string {
	var res []string
	generateParenthesisDp(n, n, nil, &res)
	return res
}

func generateParenthesisDp(i, j int, rr []byte, res *[]string) {
	if i == 0 && j == 0 {
		*res = append(*res, string(rr))
		return
	}

	if i == j {
		rr = append(rr, '(')
		generateParenthesisDp(i-1, j, rr, res)
	} else if i > 0 {
		rr = append(rr, '(')
		generateParenthesisDp(i-1, j, rr, res)

		rr[len(rr)-1] = ')'
		generateParenthesisDp(i, j-1, rr, res)
	} else {
		rr = append(rr, ')')
		generateParenthesisDp(i, j-1, rr, res)
	}
}

//航班预订统计：https://leetcode-cn.com/problems/corporate-flight-bookings/
func corpFlightBookings(bookings [][]int, n int) []int {
	res := make([]int, n)

	for _, bb := range bookings {
		for i, j := bb[0], bb[1]; i <= j; i++ {
			if i <= n {
				res[i-1] += bb[2]
			}
		}
	}

	return res
}

//和为K的子数组:https://leetcode-cn.com/problems/subarray-sum-equals-k/
func subarraySum(nums []int, k int) int {
	var res int
	var sum = make([]int, len(nums))
	sum[0] = nums[0]
	for i := 1; i < len(sum); i++ {
		sum[i] = nums[i] + sum[i-1]
	}

	for j := len(nums) - 1; j >= 0; j-- {
		if sum[j] == k {
			res++
		}

		for i := 0; i < j; i++ {
			if sum[j]-sum[i] == k {
				res++
			}
		}
	}

	return res
}

func quickSort(nums []int) {
	if len(nums) <= 0 {
		return
	}
	tmp := nums[0]
	var backward = true
	i, j := 0, len(nums)-1
	for i < j {
		if backward {
			if nums[j] < tmp {
				nums[i] = nums[j]
				backward = false
				continue
			}
			j--
		} else {
			if nums[i] > tmp {
				nums[j] = nums[i]
				backward = true
				continue
			}
			i++
		}
	}

	nums[i] = tmp

	quickSort(nums[:i])
	quickSort(nums[i+1:])
}

//数组中的第K个最大元素：https://leetcode-cn.com/problems/kth-largest-element-in-an-array/
func findKthLargest(nums []int, k int) int {
	var kNums []int
	for _, n := range nums[:k] {
		kNums = append(kNums, n)
		shiftUp(kNums)
	}

	for _, n := range nums[k:] {
		if kNums[0] < n {
			kNums[0] = n
			shiftDown(kNums)
		}
	}

	return kNums[0]
}

func shiftUp(num []int) {
	tmp := num[len(num)-1]
	i := len(num) - 1
	for ; i > 0; {
		var next int
		if i%2 == 0 {
			next = i/2 - 1
		} else {
			next = i / 2
		}

		if num[next] > tmp {
			num[i] = num[next]
		} else {
			break
		}
		i = next
	}

	num[i] = tmp
}

func shiftDown(num []int) {
	tmp := num[0]
	i := 0
	for ; 2*i+1 < len(num); {
		j := 2*i + 1
		if j+1 < len(num) && num[j+1] < num[j] {
			j++
		}

		if num[j] < tmp {
			num[i] = num[j]
		} else {
			break
		}
		i = j
	}

	num[i] = tmp
}
//type Node struct {
//	Val    int
//	Next   *Node
//	Random *Node
//}
//
////复杂链表的复制:https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/
//func copyRandomList(head *Node) *Node {
//	pToOrder := make(map[*Node]int)
//	for i, h := 0, head; h != nil; i, h = i+1, h.Next {
//		pToOrder[h] = i
//	}
//
//	randonToOrder := make(map[int]int)
//	for i, h := 0, head; h != nil; i, h = i+1, h.Next {
//		if h.Random != nil {
//			randonToOrder[i] = pToOrder[h.Random]
//		}
//	}
//
//	var hhead, tail *Node
//	orderToP := make(map[int]*Node)
//	for i, h := 0, head; h != nil; i, h = i+1, h.Next {
//		tmp := *h
//		if i == 0 {
//			tail = &tmp
//			hhead = tail
//		} else {
//			tail.Next = &tmp
//			tail = &tmp
//		}
//		orderToP[i] = &tmp
//	}
//
//	for i, h := 0, hhead; h != nil; i, h = i+1, h.Next {
//		if val, ok := randonToOrder[i]; ok {
//			h.Random = orderToP[val]
//		}
//	}
//
//	return hhead
//
//}
//
////深度优先遍历
//func copyRandomList2(head *Node) *Node {
//	visit := make(map[*Node]*Node)
//	return copyRandomList2DFS(head, visit)
//}
//
//func copyRandomList2DFS(head *Node, visit map[*Node]*Node) (n *Node) {
//	if head == nil {
//		return nil
//	}
//
//	if visit[head] != nil {
//		return visit[head]
//	}
//	n = &Node{
//		Val: head.Val,
//	}
//	visit[head] = n
//	n.Next = copyRandomList2DFS(head.Next, visit)
//	n.Random = copyRandomList2DFS(head.Random, visit)
//
//	return n
//}

//二叉搜索树与双向链表:https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/
//无法在leetcode验证
func treeToDoublyList(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	var head **TreeNode
	tail := treeToDoublyListDp(root, 0, head)
	tail.Right = *head
	(*head).Left = tail
	return *head
}

func treeToDoublyListDp(root *TreeNode, count int, head **TreeNode) *TreeNode {
	if root.Left == nil && root.Right == nil {
		count++
		if count == 1 {
			head = &root
		}
		return root
	}

	lNode := treeToDoublyListDp(root.Left, count, head)
	rNode := treeToDoublyListDp(root.Right, count, head)

	root.Left = lNode
	lNode.Right = root
	return rNode
}

func permutationStr(s string) []string {
	res := make([]string, 0)
	visit := make(map[byte]int)
	for i := 0; i < len(s); i++ {
		visit[s[i]]++
	}
	permutationDP(s, nil, visit, &res)
	return res
}

func permutationDP(s string, subStr []byte, visit map[byte]int, res *[]string) {
	if len(s) == 0 {
		return
	}
	if len(subStr) == len(s) {
		*res = append(*res, string(subStr))
	}

	var dup = make(map[byte]bool)
	for i := 0; i < len(s); i++ {
		if visit[s[i]] > 0 && !dup[s[i]] {
			subStr = append(subStr, s[i])
			visit[s[i]]--
			permutationDP(s, subStr, visit, res)
			subStr = subStr[:len(subStr)-1]
			visit[s[i]]++
			dup[s[i]] = true
		}
	}
}

//数字序列中某一位的数字：https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/
func findNthDigit(n int) int {

	digit := 1
	totalBit := 0

	for {
		intervalBit := int(9*math.Pow10(digit-1)) * digit
		if totalBit <= n && totalBit+int(intervalBit) >= n {
			break
		}
		totalBit += intervalBit
		digit++
	}

	begNum := int(math.Pow10(digit-1)) - 1
	num := (n - totalBit) / digit
	remain := (n - totalBit) % digit

	if remain == 0 {
		lastNum := begNum + num
		ss := strconv.Itoa(lastNum)
		rr, _ := strconv.Atoi(string(ss[len(ss)-1]))
		return rr
	} else {
		lastNum := begNum + num + 1
		ss := strconv.Itoa(lastNum)
		rr, _ := strconv.Atoi(string(ss[remain-1]))
		return rr
	}

}

//把数组排成最小的数:https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/
func minNumber(nums []int) string {
	var numsStr []string

	for _, num := range nums {
		s := strconv.Itoa(num)
		numsStr = append(numsStr, s)
	}
	sort.Slice(numsStr, func(i, j int) bool {
		s1, s2 := numsStr[i]+numsStr[j], numsStr[j]+numsStr[i]
		if s1 <= s2 {
			return true
		} else {
			return false
		}
	})

	res := strings.Join(numsStr, "")
	return res
}

//把数字翻译成字符串:https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/
func translateNum(num int) int {
	numStr := strconv.Itoa(num)
	var dp = make([]int, len(numStr)+1)
	dp[1] = 1
	dp[0] = 1

	for i := 2; i < len(numStr)+1; i++ {
		dp[i] += dp[i-1]

		tmp, _ := strconv.Atoi(numStr[i-2 : i])
		if numStr[i-2] != '0' && tmp <= 25 {
			dp[i] += dp[i-2]
		}
	}

	return dp[len(dp)-1]
}

//礼物的最大价值:https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof/
func maxValue(grid [][]int) int {
	//max := math.MinInt32
	//maxValueTranverse(grid, 0, 0, 0, &max)
	//return max

	var dp = make([][]int, len(grid))

	tmp := 0
	for i := range dp {
		dp[i] = make([]int, len(grid[0]))
		tmp += grid[i][0]
		dp[i][0] = tmp
	}

	tmp = 0
	for i, val := range grid[0] {
		tmp += val
		dp[0][i] = tmp
	}

	for i := 1; i < len(grid); i++ {
		for j := 1; j < len(grid[0]); j++ {
			var max = dp[i][j-1]
			if dp[i-1][j] > dp[i][j-1] {
				max = dp[i-1][j]
			}

			dp[i][j] = max + grid[i][j]
		}
	}

	return dp[len(grid)-1][len(grid[0])-1]
}

//最长不含重复字符的子字符串:https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/
func lengthOfLongestSubstring(s string) int {
	if len(s)==0{
		return 0
	}
	var dp = make([]int, len(s))
	dp[0] = 1

	for i := 1; i < len(s); i++ {
		var tmp = make(map[byte]int)
		var needAdd bool = true
		for j := i - dp[i-1]; j <= i; j++ {
			tmp[s[j]]++
			if tmp[s[j]] > 1 {
				needAdd = false
				break
			}
		}

		dp[i] = 1
		if needAdd {
			dp[i] = dp[i-1] + 1
		} else if dp[i-1] > 1 {
			dp[i] = dp[i-1]
		}
	}

	return dp[len(dp)-1]
}

//使用快排
func findKthLargest2(nums []int, k int) int {
	var i, j = 0, len(nums)
	k = len(nums) - k
	for i < j {
		m := partition(nums[i:j])
		m = i + m

		if m > k {
			j = m
		} else if m < k {
			i = m + 1
		} else {
			return nums[m]
		}
	}

	return 0
}

func partition(nums []int) int {
	tmp := nums[0]
	var i, j = 0, len(nums) - 1
	var backward = true

	for i < j {
		if backward {
			if nums[j] < tmp {
				nums[i] = nums[j]
				backward = false
				continue
			}
			j--
		} else {
			if nums[i] > tmp {
				nums[j] = nums[i]
				backward = true
				continue
			}
			i++
		}
	}

	nums[i] = tmp
	return i
}

//堆排序
func heapSort(nums []int) {

	////建堆方式1
	//e := len(nums)/2 - 1
	//for i := e; i >= 0; i-- {
	//	heapSortShiftDown(nums, i)
	//}

	//建堆方式2
	for i := 1; i < len(nums); i++ {
		heapSortShiftUp(nums[:i+1], i)
	}

	for i := len(nums) - 1; i > 0; i-- {
		nums[0], nums[i] = nums[i], nums[0]
		heapSortShiftDown(nums[:i], 0)
	}

	for i := 0; i < len(nums)/2; i++ {
		nums[i], nums[len(nums)-1-i] = nums[len(nums)-1-i], nums[i]
	}
}

func heapSortShiftDown(nums []int, index int) {
	tmp := nums[index]
	var i, j int
	for i, j = index, 2*index+1; j < len(nums); i, j = j, 2*j+1 {
		if j+1 < len(nums) && nums[j+1] < nums[j] {
			j++
		}

		if nums[j] < tmp {
			nums[i] = nums[j]
		} else {
			break
		}
	}

	nums[i] = tmp
}

func heapSortShiftUp(nums []int, index int) {
	tmp := nums[index]

	var i, j int
	for i, j = index, index/2; i > 0; i, j = j, j/2 {
		if i%2 == 0 {
			j--
		}

		if nums[j] > tmp {
			nums[i] = nums[j]
		} else {
			break
		}
	}

	nums[i] = tmp
}

//为运算表达式设计优先级:https://leetcode-cn.com/problems/different-ways-to-add-parentheses/
func diffWaysToCompute(input string) []int {
	if isNum(input) {
		iin, _ := strconv.Atoi(input)
		return []int{iin}
	}

	var res []int
	for i := 0; i < len(input); i++ {
		if input[i] == '-' || input[i] == '+' || input[i] == '*' {
			a := diffWaysToCompute(input[:i])
			b := diffWaysToCompute(input[i+1:])

			for _, v1 := range a {
				for _, v2 := range b {
					res = append(res, calResult(v1, v2, input[i]))
				}
			}
		}
	}

	return res
}

func calResult(a, b int, symbol uint8) int {
	if symbol == '-' {
		return a - b
	} else if symbol == '+' {
		return a + b
	} else {
		return a * b
	}
}

func isNum(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

//阶乘后的零：https://leetcode-cn.com/problems/factorial-trailing-zeroes/
func trailingZeroes(n int) int {
	var count int

	for i := n / 5; i > 0; i = i / 5 {
		count += i
	}

	return count
}

//计数质数：https://leetcode-cn.com/problems/count-primes/
//超时
func countPrimes(n int) int {
	if n <= 2 {
		return 0
	} else if n <= 3 {
		return 1
	}
	var res []int
	res = append(res, 2)

outer:
	for i := 3; i < n; i++ {
		if i%2 == 1 {
			for _, v := range res {
				if i%v == 0 {
					continue outer
				}
			}
			res = append(res, i)
		}
	}

	return len(res)
}

//优化算法
func countPrimes2(n int) int {
	notPrime := make([]bool, n)

	for i := 2; i*i < n; i++ {
		if !notPrime[i] {
			for j := 2 * i; j < n; j += i {
				notPrime[j] = true
			}
		}
	}

	var count int
	for i := 2; i < n; i++ {
		if !notPrime[i] {
			count++
		}
	}
	return count
}

//基本计算器 II：https://leetcode-cn.com/problems/basic-calculator-ii/
func calculate(s string) int {
	var sign = byte('+')
	var stack []int
	var st int
	var beg int
	for i := 0; i <= len(s); i++ {
		ff := func() {
			val := stringToInt(s[beg : beg+st])

			switch sign {
			case '+':
				stack = append(stack, val)
			case '-':
				stack = append(stack, -val)
			case '*':
				stack[len(stack)-1] *= val
			case '/':
				stack[len(stack)-1] /= val
			}
		}

		if i == len(s) {
			ff()
			break
		}
		switch s[i] {
		case ' ':
			continue
		case '+', '-', '*', '/':
			ff()
			sign = s[i]
			st = 0
		default:
			if st == 0 {
				beg = i
			}
			st++
		}
	}

	var rr int
	for _, val := range stack {
		rr += val
	}

	return rr
}

func stringToInt(ss string) int {
	var res int
	for i := 0; i < len(ss); i++ {
		res = res*10 + int(ss[i]-'0')
	}
	return res
}

//最长回文子串：https://leetcode-cn.com/problems/longest-palindromic-substring/
func longestPalindrome(s string) string {
	var res string
	for i := 0; i < len(s); i++ {
		r := searchPalindrome(s, i, false)
		r1 := searchPalindrome(s, i, true)
		if len(r) < len(r1) {
			r = r1
		}
		if len(r) > len(res) {
			res = r
		}
	}

	return res
}

func searchPalindrome(s string, i int, both bool) string {
	var b, e = i, i
	if both {
		if i+1 >= len(s) || s[i] != s[i+1] {
			return ""
		}
		e = i + 1
	}

	for ; b >= 0 && e < len(s) && s[b] == s[e]; b, e = b-1, e+1 {
	}

	return s[b+1 : e]

}

//有效的括号：https://leetcode-cn.com/problems/valid-parentheses/
func isValid(s string) bool {
	var stack []byte
	var mm = map[byte]byte{
		'(': ')',
		'{': '}',
		'[': ']',
	}

	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '(', '{', '[':
			stack = append(stack, s[i])
		case ')', '}', ']':
			if len(stack) == 0 {
				return false
			}
			if mm[stack[len(stack)-1]] != s[i] {
				return false
			}
			stack = stack[:len(stack)-1]
		}

	}
	if len(stack) != 0 {
		return false
	}

	return true
}

//判断子序列:https://leetcode-cn.com/problems/is-subsequence/
func isSubsequence(s string, t string) bool {
	if len(s) == 0 {
		return true
	}

	p := 0

	for i := 0; i < len(t); i++ {
		if t[i] == s[p] {
			p++
			if p >= len(s) {
				return true
			}
		}
	}
	return false
}

//连续子数组的最大和:https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/
func maxSubArray2(nums []int) int {

	res := make([]int, len(nums))
	res[0] = nums[0]
	var max = nums[0]
	for i := 1; i < len(nums); i++ {
		if res[i-1]+nums[i] > nums[i] {
			res[i] = res[i-1] + nums[i]
		} else {
			res[i] = nums[i]
		}

		if res[i] > max {
			max = res[i]
		}
	}

	return max
}

//二叉搜索树的第k大节点:https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/
func kthLargest(root *TreeNode, k int) int {
	return kthLargestTraverse(root, &k)
}

func kthLargestTraverse(root *TreeNode, k *int) int {
	if root == nil {
		return -1
	}

	if r := kthLargestTraverse(root.Right, k); r != -1 {
		return r
	}

	*k--
	if *k == 0 {
		return root.Val
	}

	if r := kthLargestTraverse(root.Left, k); r != -1 {
		return r
	}

	return -1
}

////队列的最大值：https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/
//type MaxQueue struct {
//	max        int
//	length     int
//	head, tail *queueNode
//}
//
//type queueNode struct {
//	val  int
//	next *queueNode
//}
//
//func MaxQueueConstructor() MaxQueue {
//
//}
//
//func (this *MaxQueue) Max_value() int {
//	if this.length == 0 {
//		return -1
//	}
//	return this.max
//}
//
//func (this *MaxQueue) Push_back(value int) {
//	if this.head == nil {
//		this.head = &queueNode{
//			val: value,
//		}
//		this.tail = this.head
//	} else {
//		n := &queueNode{
//			val: value,
//		}
//		this.tail.next = n
//		this.tail = n
//	}
//
//	this.length++
//	if value > this.max {
//		this.max = value
//	}
//}
//
//func (this *MaxQueue) Pop_front() int {
//	if this.head == nil {
//		return -1
//	}
//
//	val := this.head.val
//	this.head = this.head.next
//	if this.head == nil {
//		this.tail = this.head
//	}
//	this.length--
//
//}

//二叉搜索树的最近公共祖先：https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
func lowestCommonAncestor2(root, p, q *TreeNode) *TreeNode {
	if root == q {
		return q
	}
	if root == p {
		return p
	}
	if root == nil {
		return nil
	}

	r1 := lowestCommonAncestor(root.Left, p, q)
	r2 := lowestCommonAncestor(root.Right, p, q)

	if r1 != nil && r2 != nil {
		return root
	} else if r1 != nil {
		return r1
	} else if r2 != nil {
		return r2
	}
	return nil
}

//n个骰子的点数:https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/
//超时
func dicesProbability(n int) []float64 {
	countMap := make(map[int]int)
	countDicesNum(n, 0, countMap)

	var res [][2]int
	for k, v := range countMap {
		res = append(res, [2]int{k, v})
	}

	sort.Slice(res, func(i, j int) bool {
		return res[i][0] < res[j][0]
	})

	var rres []float64
	p := math.Pow(1.0/6, float64(n))
	for _, r := range res {
		rres = append(rres, float64(r[1])*p)
	}
	return rres
}

func countDicesNum(n, sum int, countMap map[int]int) {
	if n == 0 {
		countMap[sum]++
		return
	}
	for i := 1; i <= 6; i++ {
		countDicesNum(n-1, sum+i, countMap)
	}
}

func dicesProbability2(n int) []float64 {
	var dp = make([][]int, n+1)

	for i := 1; i <= n; i++ {
		dp[i] = make([]int, n*6+1)
		if 1 == i {
			for j := 1; j <= 6; j++ {
				dp[i][j] = 1
			}
		}
	}

	for i := 2; i <= n; i++ {
		for j := 1; j <= n*6; j++ {
			for k := 1; k <= 6; k++ {
				if j-k > 0 {
					dp[i][j] += dp[i-1][j-k]
				}
			}
		}
	}

	p := math.Pow(1.0/6, float64(n))

	var res []float64
	for _, val := range dp[n] {
		if val != 0 {
			res = append(res, float64(val)*p)
		}
	}

	return res
}

//和为s的连续正数序列：https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/
func findContinuousSequence(target int) [][]int {
	var res [][]int
	for i := 1; i <= target/2; i++ {
		var tmp []int
		sum := 0
		for j := i; sum < target; j++ {
			sum += j
			tmp = append(tmp, j)
		}
		if sum == target {
			res = append(res, tmp)
		}
	}

	return res
}

//翻转单词顺序:https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof/
func reverseWords(s string) string {
	var ss []string

	for i := 0; i < len(s); {
		if s[i] == ' ' {
			i++
			continue
		}

		var bs []byte
		j := i
		for ; j < len(s); j++ {
			if s[j] != ' ' {
				bs = append(bs, s[j])
			} else {
				break
			}
		}
		ss = append(ss, string(bs))
		i = j
	}

	for i := 0; i < len(ss)/2; i++ {
		ss[i], ss[len(ss)-1-i] = ss[len(ss)-1-i], ss[i]
	}

	return strings.Join(ss, " ")
}

func isBalanced(root *TreeNode) bool {
	_, res := checkIfBalanced(root)
	return res
}

func checkIfBalanced(root *TreeNode) (n int, isb bool) {
	if root == nil {
		return 0, true
	}
	n2, isb2 := checkIfBalanced(root.Left)
	n1, isb1 := checkIfBalanced(root.Right)

	if n2 > n1 {
		n = n2
	} else {
		n = n1
	}

	if isb1 && isb2 && n-n1 <= 1 && n-n2 <= 1 {
		return n + 1, isb1
	} else {
		return n + 1, false
	}
}

//二维数组中的查找：https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/
func findNumberIn2DArray(matrix [][]int, target int) bool {
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return false
	}
	for k := 0; k < len(matrix[0]) && matrix[0][k] <= target; k++ {
		var ss []int
		for i := 0; i < len(matrix); i++ {
			ss = append(ss, matrix[i][k])
		}

		_, exist := binarySearchInFindNumberIn2DArray(ss, target)
		if exist {
			return true
		}
	}

	return false
}

func binarySearchInFindNumberIn2DArray(arr []int, num int) (int, bool) {
	var i, j = 0, len(arr) - 1

	for i <= j {
		m := (i + j) / 2
		if arr[m] > num {
			if i == j {
				return i - 1, false
			}
			j = m - 1
		} else if arr[m] < num {
			if i == j {
				return i, false
			}
			i = m + 1
		} else {
			return m, true
		}
	}

	return 0, false
}

// 数组中数字出现的次数：https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/
func singleNumbers(nums []int) []int {
	hash := make(map[int]bool)

	for _, num := range nums {
		if _, ok := hash[num]; ok {
			delete(hash, num)
		} else {
			hash[num] = true
		}
	}

	var res []int
	for k := range hash {
		res = append(res, k)
	}

	return res
}

//两个链表的第一个公共节点:https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	var stack1, stack2 []*ListNode
	for p := headA; p != nil; p = p.Next {
		stack1 = append(stack1, p)
	}

	for p := headB; p != nil; p = p.Next {
		stack2 = append(stack2, p)
	}

	if len(stack1) == 0 || len(stack2) == 0 {
		return nil
	}

	i := 0
	for ; len(stack1)-1-i >= 0 && len(stack2)-1-i >= 0; i++ {
		if stack1[len(stack1)-1-i] != stack2[len(stack2)-1-i] {
			break
		}
	}

	if i == 0 {
		return nil
	} else {
		return stack1[len(stack1)-i]
	}

}

//从上到下打印二叉树 II:https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/
func levelOrder(root *TreeNode) [][]int {
	if root == nil {
		return nil
	}
	var res [][]int

	var quene []*TreeNode

	quene = append(quene, root)
	k, count := 1, 0
	for len(quene) != 0 {
		count = 0
		var tmp []int
		for _, node := range quene[:k] {
			if node.Left != nil {
				quene = append(quene, node.Left)
				count++
			}
			if node.Right != nil {
				quene = append(quene, node.Right)
				count++
			}
			tmp = append(tmp, node.Val)
		}
		if len(tmp) > 0 {
			res = append(res, tmp)
		}
		quene = quene[k:]
		//fmt.Println(quene)
		k = count
	}

	return res
}

//树的子结构:https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/
func isSubStructure(A *TreeNode, B *TreeNode) bool {
	if A == nil || B == nil {
		return false
	}
	if A.Val == B.Val {
		if checkIsSubStructure(A, B) {
			return true
		}
	}

	if isSubStructure(A.Right, B) {
		return true
	} else if isSubStructure(A.Left, B) {
		return true
	} else {
		return false
	}

}

func checkIsSubStructure(A *TreeNode, B *TreeNode) bool {
	if B == nil {
		return true
	} else if A == nil && B != nil {
		return false
	}

	if A.Val == B.Val {
		if !checkIsSubStructure(A.Left, B.Left) {
			return false
		}
		if !checkIsSubStructure(A.Right, B.Right) {
			return false
		}
		return true
	}
	return false
}

//矩阵中的路径：https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/
func exist(board [][]byte, word string) bool {
	if word == "" {
		return true
	}

	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[0]); j++ {
			if board[i][j] == word[0] {
				occupy := make(map[[2]int]bool)
				occupy[[2]int{i, j}] = true
				if existPath(board, word, 1, i, j, occupy) {
					return true
				}
			}
		}
	}

	return false
}

func existPath(board [][]byte, word string, c int, i, j int, occupy map[[2]int]bool) bool {
	if c >= len(word) {
		return true
	}

	if i+1 < len(board) && word[c] == board[i+1][j] && !occupy[[2]int{i + 1, j}] {
		occupy[[2]int{i + 1, j}] = true
		if existPath(board, word, c+1, i+1, j, occupy) {
			return true
		}
		delete(occupy, [2]int{i + 1, j})
	}

	if i-1 >= 0 && word[c] == board[i-1][j] && !occupy[[2]int{i - 1, j}] {
		occupy[[2]int{i - 1, j}] = true
		if existPath(board, word, c+1, i-1, j, occupy) {
			return true
		}
		delete(occupy, [2]int{i - 1, j})

	}

	if j-1 >= 0 && word[c] == board[i][j-1] && !occupy[[2]int{i, j - 1}] {
		occupy[[2]int{i, j - 1}] = true
		if existPath(board, word, c+1, i, j-1, occupy) {
			return true
		}
		delete(occupy, [2]int{i, j - 1})

	}

	if j+1 < len(board[0]) && word[c] == board[i][j+1] && !occupy[[2]int{i, j + 1}] {
		occupy[[2]int{i, j + 1}] = true
		if existPath(board, word, c+1, i, j+1, occupy) {
			return true
		}
		delete(occupy, [2]int{i, j + 1})
	}

	return false
}

//剪绳子：https://leetcode-cn.com/problems/jian-sheng-zi-lcof/
func cuttingRope(n int) int {
	if n <= 1 {
		return 0
	}
	var max int

	for i := 2; i <= n; i++ {
		m := n / i
		k := n % i
		var tmp = 1
		for j := 1; j <= i; j++ {
			if k > 0 {
				tmp *= m + 1
				k--
			} else {
				tmp *= m
			}
		}

		if tmp > max {
			max = tmp
		}

	}

	return max
}

//数值的整数次方：https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/
func myPow(x float64, n int) float64 {
	if n == 0 {
		return 1
	}
	if n == 1 {
		return x
	}
	if n < 0 {
		x, n = 1/x, -n
	}
	res := 1.0
	if n%2 == 1 {
		res *= x
	}
	rr := myPow(x, n/2)
	res *= rr * rr
	return res
}

//调整数组顺序使奇数位于偶数前面:https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/
func exchange(nums []int) []int {
	flag := true
	for i, j := 0, len(nums)-1; i < j; {
		if flag {
			if nums[i]%2 == 0 {
				flag = false
				continue
			}
			i++
		} else {
			if nums[j]%2 == 1 {
				nums[i], nums[j] = nums[j], nums[i]
				flag = true
			}
			j--
		}
	}
	return nums
}

//链表中倒数第k个节点：https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/
func getKthFromEnd(head *ListNode, k int) *ListNode {
	var h = head
	for i := 0; i < k; i++ {
		h = h.Next
	}

	var s = head

	for ; h != nil; h = h.Next {
		s = s.Next
	}
	return s
}

//合并两个排序的链表：https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil {
		return l2
	} else if l2 == nil {
		return l1
	}

	var mTmp, nTmp *ListNode

	for m, n := l1, l2; m != nil; {
		mTmp = m.Next

		for ; n != nil; nTmp, n = n, n.Next {
			if n.Val > m.Val {
				break
			}
		}

		if n != nil {
			if nTmp == nil {
				m.Next = l2
				l2 = m
			} else {
				nTmp.Next = m
				m.Next = n
			}
		} else {
			nTmp.Next = m
			m.Next = nil
		}
		n = m
		m = mTmp
	}

	return l2
}

//顺时针打印矩阵：https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/
func spiralOrder(matrix [][]int) []int {
	var res []int
	flag := 1
	num := math.MaxInt32
	for i, j := 0, 0; i < len(matrix) && i >= 0 && j < len(matrix[0]) && j >= 0 && matrix[i][j] != num; {
		res = append(res, matrix[i][j])
		matrix[i][j] = num
		switch flag {
		case 1:
			if (j+1) >= len(matrix[0]) || matrix[i][j+1] == num {
				flag = 2
				i++
				continue
			}
			j++

		case 2:
			if i+1 >= len(matrix) || matrix[i+1][j] == num {
				flag = 3
				j--
				continue
			}
			i++
		case 3:
			if j-1 < 0 || matrix[i][j-1] == num {
				flag = 4
				i--
				continue
			}
			j--
		case 4:
			if i-1 < 0 || matrix[i-1][j] == num {
				flag = 1
				j++
				continue
			}
			i--
		}
	}

	return res

}

//栈的压入、弹出序列：https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/
func validateStackSequences(pushed []int, popped []int) bool {
	var stack []int
	var i, j int
	for i, j = 0, 0; i < len(pushed) && j < len(popped); {
		if len(stack) > 0 && popped[j] == stack[len(stack)-1] {
			stack = stack[:len(stack)-1]
			j++
		} else {
			stack = append(stack, pushed[i])
			i++
		}
	}

	for ; j < len(popped); {
		if len(stack) > 0 && popped[j] == stack[len(stack)-1] {
			stack = stack[:len(stack)-1]
			j++
		} else {
			break
		}
	}

	if i == len(pushed) && j == len(popped) {
		return true
	}

	return false
}

//从上到下打印二叉树 III：https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/
func levelOrder2(root *TreeNode) [][]int {
	if root == nil {
		return nil
	}
	var res [][]int
	var quene []*TreeNode
	quene = append(quene, root)
	length := 1
	var reverse = false
	for len(quene) != 0 {
		var rr []int
		if !reverse { //也可以用判断奇偶的方法，这样可以减少代码
			var tmp int
			for i := 0; i < length; i++ {
				tmp += addNode(&quene, quene[i])
				rr = append(rr, quene[i].Val)
			}
			quene = quene[length:]
			length = tmp
			reverse = true
		} else {
			var tmp int
			for i := 0; i < length; i++ {
				tmp += addNode(&quene, quene[i])
				rr = append(rr, quene[length-1-i].Val)
			}
			quene = quene[length:]
			length = tmp
			reverse = false
		}
		res = append(res, rr)
	}
	return res
}

func addNode(quene *[]*TreeNode, node *TreeNode) int {
	var tmp int
	if node.Left != nil {
		*quene = append(*quene, node.Left)
		tmp++
	}
	if node.Right != nil {
		*quene = append(*quene, node.Right)
		tmp++
	}
	return tmp
}

// 二叉搜索树的后序遍历序列:https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/
func verifyPostorder(postorder []int) bool {
	if len(postorder) == 0 {
		return true
	}
	last := postorder[len(postorder)-1]

	var i int
	for i = len(postorder) - 1; i >= 0 && postorder[i] >= last; i-- {
	}
	for j := 0; j < i; j++ {
		if postorder[j] > last {
			return false
		}
	}

	if verifyPostorder(postorder[:i+1]) && verifyPostorder(postorder[i+1:len(postorder)-1]) {
		return true
	}
	return false
}

//二叉树中和为某一值的路径：https://leetcode-cn.com/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/
func pathSum(root *TreeNode, sum int) [][]int {
	var res [][]int
	pathSumDp(root, sum, []int{}, 0, &res)
	return res
}

func pathSumDp(root *TreeNode, sum int, tmp []int, total int, res *[][]int) {
	if root == nil {
		return
	}

	tmp = append(tmp, root.Val)
	total += root.Val
	if root.Right == nil && root.Left == nil && total == sum {
		tt := append([]int{}, tmp...)
		*res = append(*res, tt)
		return
	}

	pathSumDp(root.Left, sum, tmp, total, res)
	pathSumDp(root.Right, sum, tmp, total, res)
	return
}

package main

import (
	"math"
	"sort"
	"strconv"
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


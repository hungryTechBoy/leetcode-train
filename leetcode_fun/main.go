// These examples demonstrate more intricate uses of the flag package.
package main

import "fmt"

func main() {
	//coins := {}int{1, 2, 5}
	//amount := 11
	//fmt.Println(coinChange3(coins, amount))

	//t1 := TreeNode{
	//	Val:   1,
	//	Left:  nil,
	//	Right: nil,
	//}
	//t2 := TreeNode{
	//	Val:   4,
	//	Left:  nil,
	//	Right: nil,
	//}
	//t3 := TreeNode{
	//	Val:   3,
	//	Left:  nil,
	//	Right: nil,
	//}
	//t1.Right = &t2
	//t2.Left = &t3
	//fmt.Println(preorderTraversal2(&t1))
	//d := {}string{"8887","8889","8878","8898","8788","8988","7888","9888"}
	//t := "8888"
	//prices := {}int{48,12,60,93,97,42,25,64,17,56,85,93,9,48,52,42,58,85,81,84,69,36,1,54,23,15,72,15,11,94}
	//fmt.Println(maxProfit(7, prices))
	s := []int{1, 2, 5}
	fmt.Println(change(5, s))
}

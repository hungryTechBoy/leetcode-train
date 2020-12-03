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
	//s := []int{3,9,20,15,7}
	//ss := []int{9,3,15,20,7}
	//fmt.Println(buildTree(s,ss))

	//lRUCache := Constructor(2)
	//lRUCache.Put(1, 1)           // 缓存是 {1=1}
	//lRUCache.Put(2, 2)           // 缓存是 {1=1, 2=2}
	//fmt.Println(lRUCache.Get(1)) // 返回 1
	//lRUCache.Put(3, 3)          // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
	//fmt.Println(lRUCache.Get(2)) // 返回 -1 (未找到)
	//lRUCache.Put(4, 4)           // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
	//fmt.Println(lRUCache.Get(1)) // 返回 -1 (未找到)
	//fmt.Println(lRUCache.Get(3)) // 返回 3
	//fmt.Println(lRUCache.Get(4)) // 返回 4
	//
	//ttt := TwitterConstructor()
	//ttt.PostTweet(1,5)
	//ttt.PostTweet(1,3)
	//ttt.PostTweet(1,101)
	//ttt.PostTweet(1,102)
	//ttt.PostTweet(1,103)
	//ttt.PostTweet(1,104)
	//ttt.PostTweet(1,105)
	//ttt.PostTweet(1,106)
	//ttt.PostTweet(1,107)
	//ttt.PostTweet(1,108)
	//ttt.PostTweet(1,109)
	//fmt.Println(ttt.GetNewsFeed(1))
	//b := []int{0,1}
	//s := SolutionConstructor(4, b)
	//for i := 0; i < 10000; i++ {
	//	fmt.Println(s.Pick())
	//}

	//ss:=[][]int{{1,3},{3,5},{6,7},{6,8},{8,4},{9,5}}
	s := []int{1,2}
	fmt.Println(findKthLargest(s, 2))

}

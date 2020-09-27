// These examples demonstrate more intricate uses of the flag package.
package main

import "fmt"

func main() {
	coins := []int{492,364,366,144,492,316,221,326,16,166,353,5253}
	amount := 5253
	fmt.Println(coinChange(coins, amount))
}


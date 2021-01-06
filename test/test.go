package main

import "fmt"

func main() {
	var i = 5
	for i, j := 0, 0; i < 3; i++ {
		fmt.Println(j)
	}

	fmt.Println(i)
}

type tettest struct {
	val int
}

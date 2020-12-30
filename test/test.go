package main

import "fmt"

func main() {
	for i:=0;i<3;i++{
		if i==0{
			i++
			continue
		}
		fmt.Println(i)
	}
}

type tettest struct {
	val int
}

package main

import "fmt"

func main() {
	ss := make(map[string]tettest)
	tt := tettest{}
	ss["s"] = tt
	tt.val=5
	fmt.Println(ss)

}

type tettest struct {
	val int
}

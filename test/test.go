package main

import (
   "fmt"
)

func main() {
	//const nihongo = "\xbd\xb2\x3d\xbc\x20\xe2\x8c\x98"
    //for index, runeValue := range nihongo {
    //	fmt.Println(index, runeValue)
    //    //fmt.Printf("%#U starts at byte position %d\n", runeValue, index)
    //}

   //str := "adv他"
   //for _,s :=range str {
   //		fmt.Println(byte(s))
   //		fmt.Println(s)
   //}

   	const INT_MAX_STR = "2147483647"

   fmt.Println((int(1)-int(INT_MAX_STR[1])))
}


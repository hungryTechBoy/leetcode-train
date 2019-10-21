package main

import (
	"time"
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


   	go func() {
   		for {
   			if 5> 9 {
   				break
			}
		}
	}()

   	time.Sleep(1* time.Second)
   	panic("error")
   	//fmt.Println("test")

   	//for i := range ch_t {
   	//	fmt.Println(i)
	//}

}


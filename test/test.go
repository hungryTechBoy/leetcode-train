package main

import (
	"fmt"
	"unicode/utf8"
)

func main() {
	//const nihongo = "\xbd\xb2\x3d\xbc\x20\xe2\x8c\x98"
    //for index, runeValue := range nihongo {
    //	fmt.Println(index, runeValue)
    //    //fmt.Printf("%#U starts at byte position %d\n", runeValue, index)
    //}

    const nihongo = "\xbd\xb2\x3d\xbc\x20\xe2\x8c\x98"
    for i, w := 0, 0; i < len(nihongo); i += w {
        runeValue, width := utf8.DecodeRuneInString(nihongo[i:])
        fmt.Printf("%#U starts at byte position %d\n", runeValue, i)
        w = width
    }
}


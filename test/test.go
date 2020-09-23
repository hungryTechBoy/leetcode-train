package main

import (
	"fmt"
)

func main() {
	var t interface{}
	t = 6
	switch t := t.(type) {
	default:
		fmt.Printf("unexpected type %T\n", t) // %T prints whatever type t has
	case bool:
		fmt.Printf("boolean %t\n", t) // t has type bool
	case int:
		fmt.Printf("integer %d\n", t) // t has type int
	case *bool:
		fmt.Printf("pointer to boolean %t\n", *t) // t has type *bool
	case *int:
		fmt.Printf("pointer to integer %d\n", *t) // t has type *int
	}

}

type (
	namespace struct {
		NamespaceName  string `json:"namespaceName"`
		releaseKey     string `json:"-"` //最近可用version
		latestKey      string `json:"-"` //最新的version
		NotificationId int64  `json:"notificationId"`
	}
)

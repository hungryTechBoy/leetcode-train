package main

import (
	"fmt"
)

func main() {
	s := make([]int, 2, 3)
	s[0]=1
	s[1]=2

	for _, i := range s {
		fmt.Println(i)

		for j := 0; j < 1; j++ {
			s = append(s, j)
		}
		s[1] = 111111
		fmt.Println(s)
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

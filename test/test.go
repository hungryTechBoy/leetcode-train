package main

import "fmt"

func main() {
	var s = []int{1, 2}
	s1 := &s
	*s1 = append(*s1, 5)
	fmt.Println(s)
	fmt.Println(s1)

}

type (
	namespace struct {
		NamespaceName  string `json:"namespaceName"`
		releaseKey     string `json:"-"` //最近可用version
		latestKey      string `json:"-"` //最新的version
		NotificationId int64  `json:"notificationId"`
	}
)

package main

func MakeRange(beg int, end int, step int) []int {
	res := make([]int, (end-beg)/step+1)

	for i := range res {
		res[i] = beg + i*step
	}
	return res
}

func Max(n1 int, n2 int) int {
	if n1 >= n2 {
		return n1
	} else {
		return n2
	}

}

func Min(n1 int, n2 int) int {
	if n1 >= n2 {
		return n2
	} else {
		return n1
	}
}

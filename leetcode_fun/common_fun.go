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

func IntSliceContains(is []int, i int) bool {
	for _, s := range is {
		if s == i {
			return true
		}
	}
	return false
}

func Abs(i int) int {
	if i < 0 {
		return -i
	}
	return i
}

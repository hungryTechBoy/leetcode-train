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

func MaxSliceValue(s []int) int {
	var max int
	for _, n := range s {
		if n > max {
			max = n
		}
	}
	return max
}

func SumIntSlice(s []int) (res int) {
	for _, ss := range s {
		res += ss
	}
	return
}

func IntSliceEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}

	for i, v := range a {
		if v != b[i] {
			return false
		}
	}

	return true
}

func FindIntSliceIndex(list []int, val int) int {
	for i, l := range list {
		if val == l {
			return i
		}
	}
	return -1
}

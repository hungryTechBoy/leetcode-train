package main

import (
	"fmt"
	"image/color"
	"math"
)

func main() {

	//f := func(test int) {
	//	fmt.Println(1)
	//}
	//
	//fmt.Println(reflect.TypeOf(f))
	//
	//var t HandlerFunc
	//t = f
	//fmt.Println(t)
	//fmt.Println(reflect.TypeOf(t))
	//t.test()
	for i, j := 0, 7; i < 2; {
		fmt.Println(j)
		i++
	}
}

type Test interface {
	ServeHTTPs(test int) int
}

func haha(t Test) {

}

type HandlerFunc func(test int)

func (HandlerFunc) test() {
	fmt.Println("test")
}

type myint int

func (myint) test() {
	fmt.Println("test")

}

type Handler interface {
	ServeHTTP(test int) int
}

func (*Point) ServeHTTPs(test int) int {
	fmt.Println("test")
	return 1
}

type Point struct{ X, Y float64 }

type ColoredPoint struct {
	Point
	Color color.RGBA
}

//!-decl

func (p Point) Distance(q Point) float64 {
	dX := q.X - p.X
	dY := q.Y - p.Y
	return math.Sqrt(dX*dX + dY*dY)
}

func (p *Point) ScaleBy(factor float64) {
	p.X *= factor
	p.Y *= factor
}

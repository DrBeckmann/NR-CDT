module TestImages

using Images
using Luxor
    
abstract type AbstractShape end
abstract type AbstractBaseShape <: AbstractShape end
abstract type AbstractComposedShape <: AbstractShape end

struct Circle <: AbstractBaseShape
    scale::Real
end

struct Polygon <: AbstractBaseShape
    edges::Integer
    rotation::Real
    scale::Real
end

struct Star <: AbstractBaseShape
    rays::Integer
    rotation::Real
    scale::Real
end

struct Empty <: AbstractBaseShape end

struct OrbAndCross <: AbstractComposedShape
    orb::AbstractBaseShape
    cross::AbstractBaseShape
    scale::Real
end

struct Shield <: AbstractComposedShape
    emblem::AbstractBaseShape
    scale::Real
end

function generate(shape::AbstractShape; size=(128, 128), width=3)
    initiate_drawing(size, width)
    draw(shape)
    return extract_drawing()
end

function initiate_drawing(size, width)
    (x, y) = size
    Drawing(x, y, :png)
	origin()
	background("black")
	sethue("white")
	setline(width)
	scale(x / 2, y / 2)
end

function draw(circle::Circle)
	sethue("black")
	Luxor.circle(Point(0, 0), 0.75, action=:fill)
	sethue("white")
	Luxor.circle(Point(0, 0), 0.75, action=:stroke)
end

function draw(polygon::Polygon)
    rotate(-polygon.rotation)
	sethue("black")
	Luxor.ngon(Point(0, 0), 0.75, polygon.edges, action=:fill)
	sethue("white")
	Luxor.ngon(Point(0, 0), 0.75, polygon.edges, action=:stroke)
    rotate(polygon.rotation)
end

function draw(star::Star)
    rotate(-star.rotation)
	sethue("white")
    for k in 1:star.rays
        move(Point(0, 0))
	    line(Point(0.75sin(k * 2π / star.rays), -0.75cos(k * 2π / star.rays)))
    end
    strokepath()
    rotate(star.rotation)
end

function draw(::Empty) end

function draw(orbandcross::OrbAndCross)
    move(Point(0, -0.5))
    line(Point(0, 0.5))
    strokepath()
    translate(0, 0.5)
    scale(0.5)
    draw(orbandcross.orb)
    scale(2)
    translate(0, -1)
    scale(0.5)
    draw(orbandcross.cross)
    scale(2)
end

function draw(shield::Shield)
    move(Point(-0.75, -0.75))
    line(Point(-0.75, -0.5))
    curve(Point(-0.75, 0.5), Point(0, 0.75), Point(0, 0.75))
    curve(Point(0, 0.75), Point(0.75, 0.5), Point(0.75, -0.5))
    line(Point(0.75, -0.75))
    closepath()
    strokepath()
    translate(0, -0.125)
    scale(0.5)
    draw(shield.emblem)
    scale(2)
end

function extract_drawing()
    image = image_as_matrix()
    finish()
    return float(Gray.(image))
end

end
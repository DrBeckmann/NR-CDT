module TestImages

using Images
using Luxor
    
abstract type AbstractShape end
abstract type AbstractBaseShape <: AbstractShape end
abstract type AbstractComposedShape <: AbstractShape end

struct Circle <: AbstractBaseShape
    rotation::Real
    scale::Tuple{Real, Real}
end

struct Polygon <: AbstractBaseShape
    edges::Integer
    rotation::Real
    scale::Tuple{Real, Real}
end

struct Star <: AbstractBaseShape
    rays::Integer
    rotation::Real
    scale::Tuple{Real, Real}
end

struct Empty <: AbstractBaseShape end

struct OrbAndCross <: AbstractComposedShape
    orb::AbstractBaseShape
    cross::AbstractBaseShape
    rotation::Real
    scale::Tuple{Real, Real}
end

struct Shield <: AbstractComposedShape
    emblem::AbstractBaseShape
    rotation::Real
    scale::Tuple{Real, Real}
end

function generate(shape::AbstractShape; size::Tuple{Integer, Integer}=(128, 128), width::Real=3)
    initiate_luxor_drawing(size, width)
    luxor_draw(shape)
    return extract_luxor_drawing()
end

function initiate_luxor_drawing(size::Tuple{Integer, Integer}, width::Real)
    (x, y) = size
    Drawing(x, y, :png)
	origin()
	background("black")
	setline(width)
	scale(x / 2, y / 2)
end

function luxor_draw(::Empty) end

function luxor_draw(circle::Circle)
    apply_local_transform(circle)
    luxor_draw_circle()
    annul_local_transform(circle)
end

function luxor_draw_circle()
    sethue("black")
    circle(Point(0, 0), 0.75, action=:fill)
    sethue("white")
    circle(Point(0, 0), 0.75, action=:stroke)
end

function luxor_draw(polygon::Polygon)
    apply_local_transform(polygon)
    luxor_draw_polygon(polygon.edges)
	annul_local_transform(polygon)
end

function luxor_draw_polygon(edges::Integer)
    if edges > 0
        sethue("black")
        ngon(Point(0, 0), 0.75, edges, action=:fill)
        sethue("white")
        ngon(Point(0, 0), 0.75, edges, action=:stroke)
    end
end

function luxor_draw(star::Star)
    apply_local_transform(star)
    luxor_draw_star(star.rays)
    annul_local_transform(star)
end

function luxor_draw_star(rays::Integer)
    if rays > 0
        sethue("white")
        for k in 1:rays
            move(Point(0, 0))
            line(Point(0.75sin(k * 2π / rays), -0.75cos(k * 2π / rays)))
        end
        strokepath()
    end
end

function luxor_draw(orbandcross::OrbAndCross)
    apply_local_transform(orbandcross)
    luxor_draw_crosspiece()
    luxor_draw_orb(orbandcross)
    luxor_draw_cross(orbandcross)
    annul_local_transform(orbandcross)
end

function luxor_draw_crosspiece()
    sethue("white")
    move(Point(0, -0.5))
    line(Point(0, 0.5))
    strokepath()
end

function luxor_draw_orb(orbandcross)
    translate(0, 0.5)
    scale(0.5)
    luxor_draw(orbandcross.orb)
    scale(2)
    translate(0, -0.5)
end

function luxor_draw_cross(orbandcross)
    translate(0, -0.5)
    scale(0.5)
    luxor_draw(orbandcross.cross)
    scale(2)
    translate(0, 0.5)
end

function luxor_draw(shield::Shield)
    sethue("white")
    move(Point(-0.75, -0.75))
    line(Point(-0.75, -0.5))
    curve(Point(-0.75, 0.5), Point(0, 0.75), Point(0, 0.75))
    curve(Point(0, 0.75), Point(0.75, 0.5), Point(0.75, -0.5))
    line(Point(0.75, -0.75))
    closepath()
    strokepath()
    translate(0, -0.125)
    scale(0.5)
    luxor_draw(shield.emblem)
    scale(2)
end

function apply_local_transform(shape::AbstractShape)
    (x, y) = shape.scale
    ϕ = shape.rotation
    scale(x, y)
    rotate(-ϕ)
end

function annul_local_transform(shape::AbstractShape)
    (x, y) = shape.scale
    ϕ = shape.rotation
    scale(1 / x, 1 / y)
    rotate(ϕ)
end

function extract_luxor_drawing()
    image = image_as_matrix()
    finish()
    return float(Gray.(image))
end

end
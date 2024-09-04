module TestImages

using Images
using Luxor
using FFTW

export Circle, Empty, Polygon, Square, Star, Triangle
export OrbAndCross, Shield
export render

abstract type AbstractShape end
abstract type AbstractBaseShape <: AbstractShape end
abstract type AbstractComposedShape <: AbstractShape end

struct Circle <: AbstractBaseShape
    rotation::Real
    scale::Tuple{Real,Real}
end

Circle(; rotation::Real=0, scale::Tuple{Real,Real}=(1, 1)) = Circle(rotation, scale)

struct Polygon <: AbstractBaseShape
    edges::Integer
    rotation::Real
    scale::Tuple{Real,Real}
    Polygon(n, r, s) = n <= 0 ? error("negative edges") : new(n, r, s)
end

function Polygon(edges; rotation::Real=0, scale::Tuple{Real,Real}=(1, 1))
    return Polygon(edges, rotation, scale)
end

function Triangle(; rotation::Real=0, scale::Tuple{Real,Real}=(1, 1))
    return Polygon(3, rotation + π / 2, scale)
end

function Square(; rotation::Real=0, scale::Tuple{Real,Real}=(1, 1))
    return Polygon(4, rotation + π / 4, scale)
end

struct Star <: AbstractBaseShape
    rays::Integer
    rotation::Real
    scale::Tuple{Real,Real}
    Star(n, r, s) = n <= 0 ? error("negative rays") : new(n, r, s)
end

function Star(rays; rotation::Real=0, scale::Tuple{Real,Real}=(1, 1))
    return Star(rays, rotation, scale)
end

struct Empty <: AbstractBaseShape end

struct OrbAndCross <: AbstractComposedShape
    orb::AbstractBaseShape
    cross::AbstractBaseShape
    rotation::Real
    scale::Tuple{Real,Real}
end

function OrbAndCross(orb, cross; rotation::Real=0, scale::Tuple{Real,Real}=(1, 1))
    return OrbAndCross(orb, cross, rotation, scale)
end

struct Shield <: AbstractComposedShape
    emblem::AbstractBaseShape
    rotation::Real
    scale::Tuple{Real,Real}
end

function Shield(emblem; rotation::Real=0, scale::Tuple{Real,Real}=(1, 1))
    return Shield(emblem, rotation, scale)
end

function render(
    shape::AbstractShape; size::Tuple{Integer,Integer}=(128, 128), width::Real=2
)
    initiate_luxor_drawing(size, width)
    luxor_draw(shape)
    return extract_luxor_drawing()
end

function initiate_luxor_drawing(size::Tuple{Integer,Integer}, width::Real)
    (x, y) = size
    Drawing(x, y, :png)
    origin()
    background("black")
    setline(width)
    scale(x / 2, y / 2)
    return nothing
end

function luxor_draw(::Empty) end

function luxor_draw(circle::Circle)
    apply_local_transform(circle)
    luxor_draw_circle()
    annul_local_transform(circle)
    return nothing
end

function luxor_draw_circle()
    sethue("black")
    circle(Point(0, 0), 0.75; action=:fill)
    sethue("white")
    circle(Point(0, 0), 0.75; action=:stroke)
    return nothing
end

function luxor_draw(polygon::Polygon)
    apply_local_transform(polygon)
    luxor_draw_polygon(polygon.edges)
    annul_local_transform(polygon)
    return nothing
end

function luxor_draw_polygon(edges::Integer)
    sethue("black")
    ngon(Point(0, 0), 0.75, edges; action=:fill)
    sethue("white")
    ngon(Point(0, 0), 0.75, edges; action=:stroke)
    return nothing
end

function luxor_draw(star::Star)
    apply_local_transform(star)
    luxor_draw_star(star.rays)
    annul_local_transform(star)
    return nothing
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
    
    return nothing
end

function luxor_draw(orbandcross::OrbAndCross)
    apply_local_transform(orbandcross)
    luxor_draw_crosspiece()
    luxor_draw_orb(orbandcross)
    luxor_draw_cross(orbandcross)
    annul_local_transform(orbandcross)
    return nothing
end

function luxor_draw_crosspiece()
    sethue("white")
    move(Point(0, -0.5))
    line(Point(0, 0.5))
    strokepath()
    return nothing
end

function luxor_draw_orb(orbandcross::OrbAndCross)
    translate(0, 0.5)
    scale(0.5)
    luxor_draw(orbandcross.orb)
    scale(2)
    translate(0, -0.5)
    return nothing
end

function luxor_draw_cross(orbandcross::OrbAndCross)
    translate(0, -0.5)
    scale(0.5)
    luxor_draw(orbandcross.cross)
    scale(2)
    translate(0, 0.5)
    return nothing
end

function luxor_draw(shield::Shield)
    apply_local_transform(shield)
    luxor_draw_shield()
    luxor_draw_emblem(shield)
    annul_local_transform(shield)
    return nothing
end

function luxor_draw_shield()
    sethue("white")
    move(Point(-0.75, -0.75))
    line(Point(-0.75, -0.5))
    curve(Point(-0.75, 0.5), Point(0, 0.75), Point(0, 0.75))
    curve(Point(0, 0.75), Point(0.75, 0.5), Point(0.75, -0.5))
    line(Point(0.75, -0.75))
    closepath()
    strokepath()
    return nothing
end

function luxor_draw_emblem(shield::Shield)
    translate(0, -0.125)
    scale(0.5)
    luxor_draw(shield.emblem)
    scale(2)
    translate(0, 0.125)
    return nothing
end

function apply_local_transform(shape::AbstractShape)
    (x, y) = shape.scale
    ϕ = shape.rotation
    rotate(-ϕ)
    scale(x, y)
    return nothing
end

function annul_local_transform(shape::AbstractShape)
    (x, y) = shape.scale
    ϕ = shape.rotation
    scale(1 / x, 1 / y)
    rotate(ϕ)
    return nothing
end

function extract_luxor_drawing()
    image = image_as_matrix()
    finish()
    return Gray.(image)
end

end
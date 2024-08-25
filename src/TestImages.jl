module TestImages

using Luxor
    
abstract type AbstractShape end
abstract type AbstractBaseShape <: AbstractShape end
abstract type AbstractComposedShape <: AbstractShape end

struct Circle <: AbstractBaseShape
    scale::Real
end

struct Polygon <: AbstractBaseShape
    scale::Real
    rotation::Real
    edges::Unsigned
end

struct Star <: AbstractBaseShape
    scale::Real
    rotation::Real
    rays::Unsigned
end

struct Empty <: AbstractBaseShape end

struct OrbAndCross <: AbstractComposedShape
    scale::Real
    orb::AbstractBaseShape
    cross::AbstractBaseShape
end

struct Shield <: AbstractComposedShape
    scale::Real
    emblem::AbstractBaseShape
end


end
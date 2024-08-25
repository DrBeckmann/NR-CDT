module TestImages

using Luxor
    
abstract type AbstractShape end
abstract type AbstractBaseShape <: AbstractShape end
abstract type AbstractComposedShape <: AbstractShape end

struct Circle <: AbstractBaseShape
    scale::Real
end

struct Polygon <: AbstractBaseShape
    edges::Unsigned
    rotation::Real
    scale::Real
end

struct Star <: AbstractBaseShape
    rays::Unsigned
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


end
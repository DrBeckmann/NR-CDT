struct RadonTransform
    radii::Int64
    angles::Int64
    width::Float64
    function RadonTransform(radii, angles, width)
        if radii <= 0
            error("non-positive number of radii")
        elseif angles <= 0
            error("non-positive number of angles")
        elseif width < 0.0
            error("negative width")
        end 
        return new(radii, angles, width)
    end
end

struct Phantom
    data::Matrix{Float64}
    pixel_size::Float64
    dim_x::Int64
    dim_y::Int64
end

function Phantom(image::AbstractMatrix)
    pixel_size = sqrt(2) / max(size(image)...)
    return Phantom(Float64.(image), pixel_size, size(image, 2), size(image, 1))    
end

struct Ray
    t::Float64
    θ::Float64
    w::Float64
end

function (R::RadonTransform)(image::AbstractMatrix)
    P = Phantom(image)
    S = zeros(R.radii, R.angles)
    for j in axes(S, 1), k in axes(S, 2)
        r = determine_ray(j, k, R)
        S[j, k] = integrate(P, r)
    end
    return S
end

function determine_ray(j::Int64, k::Int64, R::RadonTransform)
    t = 2 * (j - 1) / (R.radii - 1) - 1
    θ = π * (k - 1) / R.angles
    return Ray(t, θ, R.width)
end

function integrate(P::Phantom, r::Ray)
    if π / 4 <= r.θ < 3 * π / 4
        return integrate_horizontal_ray(P, r)
    else
        return integrate_vertical_ray(P, r)
    end
end 

function integrate_vertical_ray(P::Phantom, r::Ray)
    S = 0.0
    for j in axes(P.data, 1)
        for k in horizontal_pixels(j, P, r)
            S += integrate_pixel(j, k, P, r)
        end
    end
    return S
end

function horizontal_pixels(j::Int64, P::Phantom, r::Ray)
    (cos_θ, sin_θ) = (cos(r.θ), sin(r.θ))
    y = index_to_y_coordinate(j, P)
    τ = (r.t * sin_θ - y) / cos_θ
    x = r.t * cos_θ + τ * sin_θ
    x_width = cos_θ * r.w
    first_index = Int64(floor(x_coordinate_to_index(x - x_width, P)))
    last_index = Int64(ceil(x_coordinate_to_index(x + x_width, P)))
    return max(first_index, 1):min(last_index, P.dim_x)
end

function integrate_horizontal_ray(P::Phantom, r::Ray)
    S = 0.0
    for k in axes(P.data, 2)
        for j in vertical_pixels(k, P, r)
            S += integrate_pixel(j, k, P, r)
        end
    end
    return S
end

function vertical_pixels(k::Int64, P::Phantom, r::Ray)
    (cos_θ, sin_θ) = (cos(r.θ), sin(r.θ))
    x = index_to_x_coordinate(k, P)
    τ = (x - r.t * cos_θ) / sin_θ
    y = r.t * sin_θ - τ * cos_θ
    y_width = sin_θ * r.w
    first_index = Int64(floor(y_coordinate_to_index(y - y_width, P)))
    last_index = Int64(ceil(y_coordinate_to_index(y + y_width, P)))
    return max(first_index, 1):min(last_index, P.dim_y)
end

function index_to_x_coordinate(k::Int64, P::Phantom)
    return (k - (P.dim_x + 1) / 2) * P.pixel_size
end

function x_coordinate_to_index(x::Float64, P::Phantom)
    return x / P.pixel_size + (P.dim_x + 1) / 2
end

function index_to_y_coordinate(j::Int64, P::Phantom)
    return ((P.dim_y + 1) / 2 - j) * P.pixel_size
end

function y_coordinate_to_index(y::Float64, P::Phantom)
    return (P.dim_y + 1) / 2 - y / P.pixel_size
end

function integrate_pixel(j::Int64, k::Int64, P::Phantom, r::Ray)
    unit_r = select_unit_ray(j, k, P, r)
    weight = integrate_unit_pixel(unit_r) 
    return weight * P.pixel_size * P.data[j, k]
end

function select_unit_ray(j::Int64, k::Int64, P::Phantom, r::Ray)
    t_pixel = t_coordinate(j, k, P, r)
    unit_t = (r.t - t_pixel) / P.pixel_size
    unit_θ = mod(r.θ, π / 2)
    unit_θ = (0 <= unit_θ < π/4) ? unit_θ : π / 2 - unit_θ
    unit_w = r.w / P.pixel_size
    return Ray(unit_t, unit_θ, unit_w)
end

function t_coordinate(j::Int64, k::Int64, P::Phantom, r::Ray)
    x = index_to_x_coordinate(k, P)
    y = index_to_y_coordinate(j, P)
    return x * cos(r.θ) + y * sin(r.θ)
end


function integrate_unit_pixel(r::Ray)
    if r.w == 0.0
        return line_integral_unit_pixel(r.t, r.θ)
    else
        # Area integral
    end
end

function line_integral_unit_pixel(t::Float64, θ::Float64)
    line = 1 / cos(θ)
    t1 = -(cos(θ) + sin(θ)) / 2
    t2 = (sin(θ) - cos(θ)) / 2
    t3 = -t2
    t4 = -t1
    if t <= t1
        return 0.0
    elseif t <= t2
        return line * (t - t1) / (t2 - t1)
    elseif t <= t3
        return line
    elseif t <= t4
        return line * (t4 - t) / (t4 - t3)
    else
        return 0.0
    end
end












function integrate_along_area_ray(I::Phantom, radius::Float64, angle::Float64, width::Float64)
    if 0 <= angle < π / 4 || 3 * π / 4 <= angle < π
        return integrate_horizontal_area_first(I, radius, angle, width)
    elseif π / 4 <= angle < π / 2 || π / 2 <= angle < 3 * π / 4
        return integrate_vertical_area_first(I, radius, angle, width)
    end
end

function integrate_horizontal_area_first(I::Phantom, radius::Float64, angle::Float64, width::Float64)
    wscale = (width/2) / cos(π - angle) / I.pixel_size
    wscale = abs(wscale)
    P = 0.0
    (nax1, nax2) = size(I.data)
    for j in axes(I.data, 2)
        x = (j - nax2 / 2 - 0.5) * I.pixel_size
        yₒ, yᵤ = intersection_y(x, angle, radius, I.pixel_size, wscale, nax1)
        for i in range(max(yᵤ,1), min(yₒ,nax1))
            y = (i - nax1 / 2 - 0.5) * I.pixel_size
            xyᵤ, xyₒ = itegration_bound(x, y, angle, radius, I.pixel_size, width)
            P += compute_partial_area(I.data[i, j], xyᵤ, xyₒ, angle, I.pixel_size)
        end
    end
    return P ./ width
end


function integrate_vertical_area_first(I::Phantom, radius::Float64, angle::Float64, width::Float64)
    wscale = (width/2) / sin(angle) / I.pixel_size
    wscale = abs(wscale)
    P = 0.0
    (nax1, nax2) = size(I.data)
    for i in axes(I.data, 1)
        y = (i - nax1 / 2 - 0.5) * I.pixel_size
        xₒ, xᵤ = intersection_x(y, angle, radius, I.pixel_size, wscale, nax1)
        for j in range(max(xᵤ,1), min(xₒ,nax2))
            x = (j - nax2 / 2 - 0.5) * I.pixel_size
            xyᵤ, xyₒ = itegration_bound(x, y, angle, radius, I.pixel_size, width)
            P += compute_partial_area(I.data[i, j], xyᵤ, xyₒ, angle, I.pixel_size)
        end
    end
    return P ./ width
end


function intersection_x(y::Float64, ψ::Float64, l::Float64, scale::Float64, wscale::Float64, nax2::Int64)
    η = (y - l * cos(ψ)) / sin(ψ)
    x = (-l * sin(ψ) + η * cos(ψ)) / scale + nax2 / 2
    xₒ = Int(floor(x + wscale + nax2))
    xᵤ = Int(ceil(x - wscale - nax2))
    return xₒ, xᵤ
end

function itegration_bound(x::Float64, y::Float64, ψ::Float64, l::Float64, scale::Float64, width::Float64)
    xyₜ = -x * sin(ψ) + y * cos(ψ)
    xyᵤ, xyₒ = (l - xyₜ - width / 2) / scale, (l - xyₜ + width / 2) / scale
    return xyᵤ, xyₒ
end

function compute_partial_area(I::Float64, xyᵤ::Float64, xyₒ::Float64, ψ::Float64, scale::Float64)
    area = (compute_unit_pixel_area(xyₒ, ψ) - compute_unit_pixel_area(xyᵤ, ψ)) * scale^2 * I
    return area
end

function integrate_along_line_ray(I::Phantom, radius::Float64, angle::Float64)
    if 0 <= angle < π / 4 || 3 * π / 4 <= angle < π
        return integrate_vertical_ray(I, radius, angle)
    elseif π / 4 <= angle < π / 2 || π / 2 <= angle < 3 * π / 4
        return integrate_horizontal_ray(I, radius, angle)
    end
end



function compute_unit_pixel_area(t::Float64, ψ::Float64)
    ψ = mod(ψ, π / 2)
    if 0 <= ψ < π/4
        return compute_unit_pixel_area_octant(t, ψ)
    elseif π / 4 <= ψ < π / 2
        return compute_unit_pixel_area_octant(t, π / 2 - ψ)
    end    
end

function compute_unit_pixel_area_octant(t::Float64, ψ::Float64)
    area_triangle = tan(ψ) / 2
    area_parallogram = 1 - tan(ψ)
    t1 = -(cos(ψ) + sin(ψ)) / 2
    t2 = (sin(ψ) - cos(ψ)) / 2
    t3 = -t2
    t4 = -t1
    if t <= t1
        return 0.0
    elseif t <= t2
        return area_triangle * ((t - t1) / (t2 - t1))^2
    elseif t <= t3
        return area_triangle + area_parallogram * (t - t2) / (t3 - t2)
    elseif t <= t4
        return 1 - area_triangle * ((t - t4) / (t4 - t3))^2
    else
        return 1.0
    end
end


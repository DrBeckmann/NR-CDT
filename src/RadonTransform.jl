module RadonTransform

export radon

struct Intensity
    data::Matrix{Float64}
    pixel_size::Float64
end


function Intensity(I::AbstractMatrix)
    pixel_size = sqrt(2) / max(size(I)...)
    return Intensity(I, pixel_size)    
end

function radon(image::Matrix{Float64}, radii::Int64, angles::Real, width::Float64)
    t = collect(LinRange(-1, 1, radii))
    θ = (angles == 0) ? [0.0] : ((mod(angles,1) != 0) ? [Float64(angles)] : ((angles == 1) ? [0.0] : collect(LinRange(0.0, π, angles))))
    return radon_compute(image, t, θ, width)
end

function radon_compute(image::Matrix{Float64}, radii::Vector{Float64}, angles::Vector{Float64}, width::Float64)
    I = Intensity(Float64.(image))
    S = zeros((length(radii), length(angles)))
    for ℓ in eachindex(radii), k in eachindex(angles)
        S[ℓ, k] = integrate_along_ray(I, radii[ℓ], angles[k], width)
    end
    return S
end

function integrate_along_ray(I::Intensity, radius::Float64, angle::Float64, width::Float64)
    if width > 0
        integrate_along_area_ray(I::Intensity, radius::Float64, angle::Float64, width::Float64)
    else 
        integrate_along_line_ray(I::Intensity, radius::Float64, angle::Float64)
    end
end 

function integrate_along_area_ray(I::Intensity, radius::Float64, angle::Float64, width::Float64)
    if 0 <= angle < π / 4 || 3 * π / 4 <= angle < π
        return integrate_horizontal_area_first(I, radius, angle, width)
    elseif π / 4 <= angle < π / 2 || π / 2 <= angle < 3 * π / 4
        return integrate_vertical_area_first(I, radius, angle, width)
    end
end

function integrate_horizontal_area_first(I::Intensity, radius::Float64, angle::Float64, width::Float64)
    wscale = (width/2) / cos(π - angle) / I.pixel_size
    wscale = abs(wscale)
    P = 0.0
    (nax1, nax2) = size(I.data)
    for j in axes(I.data, 2)
        x = (j - nax2 / 2) * I.pixel_size
        yₒ, yᵤ = intersection_y(x, angle, radius, I.pixel_size, wscale, nax1)
        for i in range(max(yᵤ,1), min(yₒ,nax1))
            y = (i - nax1 / 2) * I.pixel_size
            xyᵤ, xyₒ = itegration_bound(x, y, angle, radius, I.pixel_size, width)
            P += compute_partial_area(I.data[i, j], xyᵤ, xyₒ, angle, I.pixel_size)
        end
    end
    return P ./ width
end


function integrate_vertical_area_first(I::Intensity, radius::Float64, angle::Float64, width::Float64)
    wscale = (width/2) / sin(angle) / I.pixel_size
    wscale = abs(wscale)
    P = 0.0
    (nax1, nax2) = size(I.data)
    for i in axes(I.data, 1)
        y = (i - nax1 / 2) * I.pixel_size
        xₒ, xᵤ = intersection_x(y, angle, radius, I.pixel_size, wscale, nax1)
        for j in range(max(xᵤ,1), min(xₒ,nax2))
            x = (j - nax2 / 2) * I.pixel_size
            xyᵤ, xyₒ = itegration_bound(x, y, angle, radius, I.pixel_size, width)
            P += compute_partial_area(I.data[i, j], xyᵤ, xyₒ, angle, I.pixel_size)
        end
    end
    return P ./ width
end

function intersection_y(x::Float64, ψ::Float64, l::Float64, scale::Float64, wscale::Float64, nax1::Int64)
    η = (x + l * sin(ψ)) / cos(ψ)
    y = (l * cos(ψ) + η * sin(ψ)) / scale + nax1 / 2
    yₒ = Int(floor(y + wscale + 0.5))
    yᵤ = Int(ceil(y - wscale - 0.5))
    return yₒ, yᵤ
end

function intersection_x(y::Float64, ψ::Float64, l::Float64, scale::Float64, wscale::Float64, nax2::Int64)
    η = (y - l * cos(ψ)) / sin(ψ)
    x = (-l * sin(ψ) + η * cos(ψ)) / scale + nax2 / 2
    xₒ = Int(floor(x + wscale + 0.5))
    xᵤ = Int(ceil(x - wscale - 0.5))
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

function integrate_along_line_ray(I::Intensity, radius::Float64, angle::Float64)
    if 0 <= angle < π / 4 || 3 * π / 4 <= angle < π
        return integrate_horizontal_line_first(I, radius, angle)
    elseif π / 4 <= angle < π / 2 || π / 2 <= angle < 3 * π / 4
        return integrate_vertical_line_first(I, radius, angle)
    end
end

function integrate_horizontal_line_first(I::Intensity, radius::Float64, angle::Float64)
    P = 0.0
    (nax1, nax2) = size(I.data)
    for j in axes(I.data, 2)
        x = (j - nax2 / 2) * I.pixel_size
        yₒ, yᵤ = intersection_y(x, angle, radius, I.pixel_size, 1.0, nax1)
        for i in range(max(yᵤ,1), min(yₒ,nax1))
            y = (i - nax1 / 2) * I.pixel_size
            xyₜ = intersection_t(x, y, angle)
            P += compute_partial_line(I.data[i, j], radius, xyₜ, angle, I.pixel_size)
        end
    end
    return P
end

function integrate_vertical_line_first(I::Intensity, radius::Float64, angle::Float64)
    P = 0.0
    (nax1, nax2) = size(I.data)
    for i in axes(I.data, 1)
        y = (i - nax1 / 2) * I.pixel_size
        xₒ, xᵤ = intersection_x(y, angle, radius, I.pixel_size, 1.0, nax1)
        for j in range(max(xᵤ,1), min(xₒ,nax2))
            x = (j - nax2 / 2) * I.pixel_size
            xyₜ = intersection_t(x, y, angle)
            P += compute_partial_line(I.data[i, j], radius, xyₜ, angle, I.pixel_size)
        end
    end
    return P
end

function intersection_t(x::Float64, y::Float64, ψ::Float64)
    xyₜ = -x * sin(ψ) + y * cos(ψ)
    return xyₜ
end

function compute_partial_line(I::Float64, l::Float64, xyₜ::Float64, ψ::Float64, scale::Float64)
    line = compute_unit_pixel_line((l - xyₜ) / scale, ψ) * scale * I
    return line
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

function compute_unit_pixel_line(t::Float64, ψ::Float64)
    ψ = mod(ψ, π / 2)
    if 0 <= ψ < π/4
        return compute_unit_pixel_line_octant(t, ψ)
    elseif π / 4 <= ψ < π / 2
        return compute_unit_pixel_line_octant(t, π / 2 - ψ)
    end    
end

function compute_unit_pixel_line_octant(t::Float64, ψ::Float64)
    line = 1 / cos(ψ)
    t1 = -(cos(ψ) + sin(ψ)) / 2
    t2 = (sin(ψ) - cos(ψ)) / 2
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

end
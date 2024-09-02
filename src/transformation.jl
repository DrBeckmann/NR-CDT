
function radon(image::AbstractMatrix, radii::Integer, angles::Integer; width::Real=0)
    ψ = LinRange(0, π, angles)
    t = LinRange(-1, 1, radii)
    if width > 1e-8
        return radon_area_fast(Float64.(image), ψ, t, width)
    else
        return radon_line(Float64.(image), ψ, t)
    end
end

function radon_area(I::AbstractMatrix, θ::AbstractRange, t::AbstractRange, width::Real)
    P = zeros(eltype(I), length(t), length(θ))
    ax1, ax2 = axes(I)

    nax1, nax2 = length(ax1), length(ax2)
    scale = sqrt(2) / max(nax1, nax2)
    for j in ax2, i in ax1
        x = (j - nax2 / 2) * scale
        y = (i - nax1 / 2) * scale
        for (k, θₖ) in enumerate(θ)
            for (ℓ, tₗ) in enumerate(t)
                xyt = -x * sin(θₖ) + y * cos(θₖ)
                P[ℓ, k] += (compute_unit_pixel_area((tₗ - xyt + width / 2) / scale, θₖ) - compute_unit_pixel_area((tₗ - xyt - width / 2) / scale, θₖ)) * scale^2 * I[i, j]
            end
        end
    end

    return P ./ width
end

function radon_area_fast(I::AbstractMatrix, θ::AbstractRange, t::AbstractRange, width::Real)
    P = zeros(eltype(I), length(t), length(θ))
    ax1, ax2 = axes(I)

    nax1, nax2 = length(ax1), length(ax2)
    scale = sqrt(2) / max(nax1, nax2)
    for (k, θₖ) in enumerate(θ)
        for (ℓ, tₗ) in enumerate(t)
            # tscale = tₗ / scale 
            # wscale = (tₗ - width/2) / cos(θₖ)
            if 0 <= mod(θₖ , π / 2) < π / 4
                for i in ax1
                    η = (tₗ * sin(θₖ) / scale - i) / cos(θₖ)
                    xᵢ = tₗ * cos(θₖ) / scale + η * sin(θₖ)
                    xₒ = Int(floor(xᵢ + width / 2 / scale + 0.5))
                    xᵤ = Int(ceil(xᵢ - width / 2 / scale - 0.5))
                    for j in range(xᵤ, xₒ)
                        x = (j - nax2 / 2) * scale
                        y = (i - nax1 / 2) * scale
                        xyt = -x * sin(θₖ) + y * cos(θₖ)
                        P[ℓ, k] += (compute_unit_pixel_area((tₗ - xyt + width / 2) / scale, θₖ) - compute_unit_pixel_area((tₗ - xyt - width / 2) / scale, θₖ)) * scale^2 * I[i, j]
                    end
                end
            elseif π / 4 <= mod(θₖ , π / 2) < π / 2
                for j in ax2
                    η = - (tₗ * cos(θₖ) / scale - i) / sin(θₖ)
                    yⱼ = tₗ * sin(θₖ) / scale - η * sin(θₖ)
                    yₒ = Int(floor(yᵢ + width / 2 / scale + 0.5))
                    yᵤ = Int(ceil(yᵢ - width / 2 / scale - 0.5))
                    for i in range(yᵤ, yₒ)
                        x = (j - nax2 / 2) * scale
                        y = (i - nax1 / 2) * scale
                        xyt = -x * sin(θₖ) + y * cos(θₖ)
                        P[ℓ, k] += (compute_unit_pixel_area((tₗ - xyt + width / 2) / scale, θₖ) - compute_unit_pixel_area((tₗ - xyt - width / 2) / scale, θₖ)) * scale^2 * I[i, j]
                    end
                end
            end
        end
    end

    return P ./ width
end


function radon_line(I::AbstractMatrix, θ::AbstractRange, t::AbstractRange)
    P = zeros(eltype(I), length(t), length(θ))
    ax1, ax2 = axes(I)

    nax1, nax2 = length(ax1), length(ax2)
    scale = sqrt(2) / max(nax1, nax2)
    for j in ax2, i in ax1
        x = (j - nax2 / 2) * scale
        y = (i - nax1 / 2) * scale
        for (k, θₖ) in enumerate(θ)
            for (ℓ, tₗ) in enumerate(t)
                xyt = -x * sin(θₖ) + y * cos(θₖ)
                P[ℓ, k] += compute_unit_pixel_line((tₗ - xyt) / scale, θₖ) * scale * I[i, j]
            end
        end
    end

    return P
end

function compute_unit_pixel_area(t::Real, ψ::Real)
    ψ = mod(ψ, π / 2)
    if 0 <= ψ < π/4
        return compute_unit_pixel_area_octant(t, ψ)
    elseif π / 4 <= ψ < π / 2
        return compute_unit_pixel_area_octant(t, π / 2 - ψ)
    end    
end

function compute_unit_pixel_area_octant(t::Real, ψ::Real)
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

function compute_unit_pixel_line(t::Real, ψ::Real)
    ψ = mod(ψ, π / 2)
    if 0 <= ψ < π/4
        return compute_unit_pixel_line_octant(t, ψ)
    elseif π / 4 <= ψ < π / 2
        return compute_unit_pixel_line_octant(t, π / 2 - ψ)
    end    
end

function compute_unit_pixel_line_octant(t::Real, ψ::Real)
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
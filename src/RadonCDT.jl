struct RadonCDT 
    quantiles::Int64
    Radon::RadonTransform
    signal_raising::Float64
    function RadonCDT(q, R, sr)
        if q <= 0
            error("non-positive number of quantiles") 
        elseif sr < 0.0
            error("negative signal raising")
        end
        return new(q, R, sr)
    end
end

RadonCDT(q::Int64, R::RadonTransform) = RadonCDT(q, R, 1e-13)

function (RCDT::RadonCDT)(image::AbstractMatrix)
    S = RCDT.Radon(image)
    Q = zeros(RCDT.quantiles, RCDT.Radon.angles)
    for j in axes(Q, 1)
        Q[j, :] = cdt(S[j, :], RCDT)
    end
    return Q
end

function cdt(s::Vector{Float64}, RCDT::RadonCDT)
    p = signal_to_density(s, RCDT.signal_raising)
    P = cumsum(p)
    s_grid = collect(LinRange(-1, 1, length(s)))
    q_grid = collect(LinRange(0, 1, RCDT.quantiles +2))[2:end - 1]
    q = LinearInterpolation(P, s_grid)
    return q(q_grid)
end

function signal_to_density(s::Vector{Float64}, ϵ::Float64)
    p = s .+ ϵ
    return p ./= sum(p)
end
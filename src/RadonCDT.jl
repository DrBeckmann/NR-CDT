struct RadonCDT 
    quantiles::Int64
    Radon::RadonTransform
    function RadonCDT(q, R)
        if q <= 0
            error("non-positive number of quantiles")
        end
        return new(q, R)
    end
end

function (RCDT::RadonCDT)(image::AbstractMatrix)
    S = RCDT.Radon(image)
    Q = zeros(RCDT.quantiles, RCDT.Radon.angles)
    for k in axes(Q, 2)
        Q[:, k] = cdt(S[:, k], RCDT)
    end
    return Q
end

function cdt(s::Vector{Float64}, RCDT::RadonCDT)
    p = signal_to_density(s)
    P = cumsum(p)
    s_grid = collect(LinRange(-1, 1, length(s)))
    q_grid = collect(LinRange(0, 1, RCDT.quantiles +2))[2:end - 1]
    q = LinearInterpolation(P, s_grid)
    return q(q_grid)
end

function signal_to_density(s::Vector{Float64})
    return s ./= sum(s)
end

struct NormRadonCDT
    RCDT::RadonCDT
end

function (NRCDT::NormRadonCDT)(image::AbstractMatrix)
    rcdt = NRCDT.RCDT(image)
    nrcdt = (rcdt .- mean(rcdt, dims=1)) ./ std(rcdt, dims=1)
    return nrcdt
end

struct MaxNormRadonCDT
    RCDT::RadonCDT
end

function (mNRCDT::MaxNormRadonCDT)(image::AbstractMatrix)
    NRCDT = NormRadonCDT(mNRCDT.RCDT)
    nrcdt = NRCDT(image)
    mnrcdt = dropdims(maximum(nrcdt, dims=2), dims=2)
    return mnrcdt
end

struct MeanNormRadonCDT
    RCDT::RadonCDT
end

function (aNRCDT::MeanNormRadonCDT)(image::AbstractMatrix)
    NRCDT = NormRadonCDT(aNRCDT.RCDT)
    nrcdt = NRCDT(image)
    anrcdt = dropdims(mean(nrcdt, dims=2), dims=2)
    return anrcdt
end
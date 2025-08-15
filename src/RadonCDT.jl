struct RadonCDT 
    quantiles::Int64
    Radon::RadonTransform
    signal_raising::Float64
    function RadonCDT(q, R, ϵ)
        if q <= 0
            error("non-positive number of quantiles")
        elseif ϵ <= 0
            error("non-positive signal raising")
        end
        return new(q, R, ϵ)
    end
end

RadonCDT(q, R) = RadonCDT(q, R, 1e-10)

function (RCDT::RadonCDT)(image::AbstractMatrix)
    S = RCDT.Radon(image)
    Q = zeros(RCDT.quantiles, RCDT.Radon.angles)
    for k in axes(Q, 2)
        Q[:, k] = cdt(S[:, k], RCDT)
    end
    return Q
end

function cdt(s::Vector{Float64}, RCDT::RadonCDT)
    p = signal_to_density(s, RCDT.signal_raising)
    P = cumsum(p)
    s_grid = collect(LinRange(-1 - RCDT.Radon.width / 2, 1 + RCDT.Radon.width / 2, length(s)))
    q_grid = collect(LinRange(0, 1, RCDT.quantiles + 2))[2:end - 1]
    q = LinearInterpolation(P, s_grid)
    return q(q_grid)
end

function signal_to_density(s::Vector{Float64}, ϵ::Float64)
    p = s .+ ϵ
    return p ./= sum(p)
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

struct MinNormRadonCDT
    RCDT::RadonCDT
end

function (iNRCDT::MinNormRadonCDT)(image::AbstractMatrix)
    NRCDT = NormRadonCDT(iNRCDT.RCDT)
    nrcdt = NRCDT(image)
    inrcdt = dropdims(minimum(nrcdt, dims=2), dims=2)
    return inrcdt
end

struct MaxMinAbsNormRadonCDT
    RCDT::RadonCDT
end

function (miabsNRCDT::MaxMinAbsNormRadonCDT)(image::AbstractMatrix)
    NRCDT = NormRadonCDT(miabsNRCDT.RCDT)
    nrcdt = NRCDT(image)
    mnrcdt = dropdims(maximum(abs.(nrcdt), dims=2), dims=2)
    inrcdt = dropdims(minimum(abs.(nrcdt), dims=2), dims=2)
    return mnrcdt - inrcdt
end

struct MaxMinNormRadonCDT
    RCDT::RadonCDT
end

function (miNRCDT::MaxMinNormRadonCDT)(image::AbstractMatrix)
    NRCDT = NormRadonCDT(miNRCDT.RCDT)
    nrcdt = NRCDT(image)
    minrcdt = dropdims(maximum(nrcdt .- dropdims(minimum(nrcdt, dims=2), dims=2), dims=2), dims=2)
    return minrcdt
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

function filter_angles(rcdt::AbstractMatrix, num_angle::Int64, angles::Int64)
    choice_angles = sort(Int64.(angles+1 .- collect(range(0,angles,length=num_angle+1))))[1:num_angle]
    filter_rcdt = rcdt[:,choice_angles]
    return filter_rcdt
end

function normalization(rcdt::AbstractArray)
    nrcdt = (rcdt .- mean(rcdt, dims=1)) ./ std(rcdt, dims=1)
    return nrcdt
end

function min_normalization(rcdt::AbstractArray)
    inrcdt = dropdims(minimum(normalization(rcdt), dims=2), dims=2)
    return inrcdt
end

function max_normalization(rcdt::AbstractArray)
    mnrcdt = dropdims(maximum(normalization(rcdt), dims=2), dims=2)
    return mnrcdt
end

function mean_normalization(rcdt::AbstractArray)
    anrcdt = dropdims(mean(normalization(rcdt), dims=2), dims=2)
    return anrcdt
end

function median_normalization(rcdt::AbstractArray)
    anrcdt = dropdims(median(normalization(rcdt), dims=2), dims=2)
    return anrcdt
end

function tv_normalization(rcdt::AbstractArray)
    mnrcdt = dropdims(sum(abs.(normalization(rcdt) .- circshift(normalization(rcdt), (0,1))), dims=2), dims=2)
    return mnrcdt
end

function mtv_normalization(rcdt::AbstractArray)
    # mnrcdt = dropdims(sum(abs.(diff(normalization(rcdt), dims=2)).^1/4, dims=2).^4, dims=2)
    mnrcdt = dropdims(sum(abs.(diff(normalization(rcdt), dims=2)[1:end-1,:]) .- abs.(diff(normalization(rcdt), dims=2)[2:end,:]), dims=2), dims=2)
    # mnrcdt = dropdims(abs.(diff(normalization(rcdt), dims=2)[1:2,:]) .+ abs.(diff(normalization(rcdt), dims=2)[end-1:end,:]), dims=2)
    return mnrcdt
end

function minabs_normalization(rcdt::AbstractArray)
    inrcdt = dropdims(minimum(abs.(normalization(rcdt)).^1/2, dims=2), dims=2)
    return inrcdt
end

function maxabs_normalization(rcdt::AbstractArray)
    mnrcdt = dropdims(maximum(abs.(normalization(rcdt)), dims=2), dims=2)
    return mnrcdt
end

function maxminabs_normalization(rcdt::AbstractArray)
    mnrcdt = dropdims(maximum(abs.(normalization(rcdt)), dims=2), dims=2)
    inrcdt = dropdims(minimum(abs.(normalization(rcdt)), dims=2), dims=2)
    return mnrcdt - inrcdt
end

function maxmin_normalization(rcdt::AbstractArray)
    mnrcdt = dropdims(maximum(normalization(rcdt), dims=2), dims=2)
    inrcdt = dropdims(minimum(normalization(rcdt), dims=2), dims=2)
    return mnrcdt - inrcdt
    #return mnrcdt .+ mnrcdt[end:-1:1]
end

function mink(rcdt::AbstractArray)
    mni = rcdt .- minimum(rcdt, dims=2)
	return mni
end

function maxk(rcdt::AbstractArray)
    mni = rcdt .- maximum(rcdt, dims=2)
	return mni
end

function absm(rcdt::AbstractArray)
    a = abs.(rcdt)
	return a
end
module radon_cdt

using NormalizedRadonCDT: RadonTransform as RT
using NormalizedRadonCDT.RadonTransform: radon as exactradon
using Interpolations: LinearInterpolation as LinInter
using LIBSVM, LIBLINEAR

export rcdt, signal_to_pdf, cdt

function signal_to_pdf(signal::AbstractArray, eps::Real)

    if sum(signal) > 0
        pdf = signal./sum(signal)
    else 
        pdf = signal
    end
    pdf = pdf .+ eps
    pdf = pdf ./ sum(pdf)
    return pdf
end

function cdt(x₀::AbstractArray, s₀::AbstractArray, x₁::AbstractArray, s₁::AbstractArray)
    s₀ = signal_to_pdf(s₀, 1e-15)
    s₁ = signal_to_pdf(s₁, 1e-15)
    r = minimum(abs.(x₀ .- circshift(x₀, 1)))
    cum₀ = cumsum(s₀)
    cum₁ = cumsum(s₁)

    if size(unique(s₀))[1] == 1
        s_hat_inter = LinInter(cum₁, x₁)
        # xnew₀ = ifelse.(x₀ .< minimum(cum₁), minimum(cum₁), ifelse.(x₀ .> maximum(cum₁), maximum(cum₁), x₀))
        # xnew = ifelse.(cum₀ .< minimum(cum₁), minimum(cum₁), ifelse.(cum₀ .> maximum(cum₁[end-1]), maximum(cum₁[end-1]), cum₀))
        xnew = collect(LinRange(0,1,size(cum₀)[1]+2))[2:end-1]
        s_hat = s_hat_inter(xnew)
    else
        s_hat_inter = LinInter(r .* cum₁, x₁)
        s_hat = s_hat_inter(r .* cum₀)
    end 

    return s_hat
end

function rcdt(ref::AbstractMatrix, tar::AbstractMatrix, angles::Real, scale_radii::Integer, width::Real)

    radii = Int(ceil(sqrt(2) * size(tar)[1]))
    
    tar_radon = transpose(exactradon(Float64.(tar); opt=RT.RadonOpt(Int64(radii*scale_radii), Int64(angles), Float64(width))))


    if size(unique(ref))[1] == 1
        ref_radon = ones(size(tar_radon))
    else
        ref_radon = transpose(exactradon(Float64.(ref); opt=RT.RadonOpt(Int64(radii*scale_radii), Int64(angles), Float64(width))))
    end

    tar_rcdt = zeros(size(ref_radon));
    x_ref = collect(LinRange(0, 1, size(ref_radon)[2]))
    x_tar = collect(LinRange(-1, 1, size(tar_radon)[2]))
    
    for i in 1:size(ref_radon)[1]
        tar_rcdt[i, :] = cdt(x_ref, ones(size(ref_radon)[2])/size(ref_radon)[2], x_tar, tar_radon[i, :])
    end

    return tar_rcdt, tar_radon
end

end
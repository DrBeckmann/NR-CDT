module radon_cdt

using NormalizedRadonCDT.transformation: radon as exactradon
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
    s₀ = signal_to_pdf(s₀, 1e-7)
    s₁ = signal_to_pdf(s₁, 1e-7)
    r = minimum(abs.(x₀ .- circshift(x₀, 1)))
    cum₀ = cumsum(s₀)
    cum₁ = cumsum(s₁)

    if size(unique(s₀))[1] == 1
        s_hat_inter = LinInter(cum₁, x₁)
        xnew₀ = ifelse.(x₀ .< minimum(cum₁), minimum(cum₁), ifelse.(x₀ .> maximum(cum₁), maximum(cum₁), x₀))
        s_hat = s_hat_inter(xnew₀)
    else
        s_hat_inter = LinInter(r .* cum₁, x₁)
        s_hat = s_hat_inter(r .* cum₀)
    end 

    return s_hat
end

function rcdt(ref::AbstractMatrix, tar::AbstractMatrix, angles::Integer, width::Real)

    radii = Int(ceil(sqrt(2) * size(tar)[1]))
    
    tar_radon = transpose(exactradon(tar, radii, angles, width))

    if size(unique(ref))[1] == 1
        ref_radon = ones(size(tar_radon))
    else
        ref_radon = transpose(exactradon(ref, radii, angles, width))
    end

    tar_rcdt = zeros(size(ref_radon));
    x_ref = LinRange(0, 1, size(ref_radon)[2])
    x_tar = LinRange(0, 1, size(tar_radon)[2])
    
    for i in 1:size(ref_radon)[1]
        tar_rcdt[i, :] = cdt(x_ref, ref_radon[i, :], x_tar, tar_radon[i, :])
    end

    return tar_rcdt, tar_radon
end

end
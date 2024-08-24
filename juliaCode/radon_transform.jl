function signal_to_pdf(input, epsilon)

    if sum(input) > 0
        pdf = input./sum(input);
    else 
        pdf = input;
    end
    pdf = pdf .+ epsilon;
    pdf = pdf ./sum(pdf);
    return pdf
end;

function cdt(x0, s0, x1, s1)
    s0 = signal_to_pdf(s0, 1e-7);
    s1 = signal_to_pdf(s1, 1e-7);
    r = minimum(abs.(x0 .- circshift(x0, 1)));
    cum0 = cumsum(s0);
    cum1 = cumsum(s1);

    if size(unique(s0))[1] == 1
        s_hat_inter = LinInter(cum1, x1);
        x00 = ifelse.(x0.<minimum(cum1), minimum(cum1), ifelse.(x0.>maximum(cum1), maximum(cum1), x0))
        s_hat = s_hat_inter(x00)
    else
        s_hat_inter = LinInter(r .* cum1, x1);
        s_hat = s_hat_inter(r .* cum0)
    end 

    return s_hat
end;

function rradon(im_size, num_angles, im)
    angles = LinRange(0, pi, num_angles+1)
    I = Int(ceil(sqrt(2) * im_size[1] / 2))
    R = Int(ceil(sqrt(2) * im_size[1]))
    ran = LinRange(-I, I, R+1)

    r_trafo = Radon(im , angles, ran)

    return r_trafo
end;

function rcdt(ref, tar, num_angles)
    
    tar_radon = transpose(rradon(size(tar), num_angles, tar));

    if size(unique(ref))[1] == 1
        ref_radon = ones(size(tar_radon));
    else
        ref_radon = transpose(rradon(size(ref), num_angles, ref));
    end

    tar_rcdt = zeros(size(ref_radon));
    x_ref = LinRange(0, 1, size(ref_radon)[2]);
    x_tar = LinRange(0, 1, size(tar_radon)[2]);
    
    for i in 1:size(ref_radon)[1]
        tar_rcdt[i, :] = cdt(x_ref, ref_radon[i, :], x_tar, tar_radon[i, :]);
    end

    return tar_rcdt, tar_radon
end;
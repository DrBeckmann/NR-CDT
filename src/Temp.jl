using NormalizedRadonCDT.TestImages
using JLD2, Plots

function generator_temp(c::Integer, s::Integer, t::Integer, h::Integer, st::Vector)

    # Image size of the temp 
    image_size = 128;

    dimₒ = c + s + t
    dimᵤ = size(st)[1] + h

    templates = zeros(dimₒ*dimᵤ, image_size, image_size)

    k = 1
    if c == 1
        for l in st
            templates[k,:,:] = render(OrbAndCross(Circle(), Star(l)), width=4.0)
            k += 1
        end
        if h > 0
            templates[k,:,:] = render(Shield(Circle()), width=4.0)
            k += 1
        end
    end
    if s == 1
        for l in st
            templates[k,:,:] = render(OrbAndCross(Square(), Star(l)), width=4.0)
            k += 1
        end
        if h > 0
            templates[k,:,:] = render(Shield(Square()), width=4.0)
            k += 1
        end
    end
    if t == 1
        for l in st
            templates[k,:,:] = render(OrbAndCross(Triangle(), Star(l)), width=4.0)
            k += 1
        end
        if h > 0
            templates[k,:,:] = render(Shield(Triangle()), width=4.0)
            k += 1
        end
    end

    save("temp.jld", "temp", templates)
end 

function view_temp(c::Integer, s::Integer, t::Integer, h::Integer, st::Vector)
    temp = load("temp.jld")["temp"]

    dimₒ = c + s + t
    dimᵤ = size(st)[1] + h

    #plt = plot(layout=(dimₒ,dimᵤ))
    plt = []
    for i in 1:dimₒ*dimᵤ
        push!(plt, heatmap(temp[i,:,:], aspect_ratio=:equal, axis=([], false), cbar=false, c = :grayC))
        #plot!(plt, Gray.(temp[i,:,:]), subplot=i, xaxis=false, yaxis=false, grid=false); # plot each set in a different subplot
    end
    display(plt)
    plot(plt...)
end;
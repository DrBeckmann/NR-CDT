function generator_temp(c::Int, s::Int, t::Int, p::Tuple{Int, AbstractVector}=(0,[]), st::AbstractVector)

    # Image size of the temp 
    image_size = 128;

    dimₒ = c + s + t + p[1] 
    dimᵤ = size(st)

    templates = zeros(dimₒ*dimᵤ, image_size, image_size)

    k = 1
    if c == 1
        for l in st
            templates[k,:,:] = render(OrbAndCross(Circle(), Star(l)))
            k += 1
        end
    elseif s == 1
        for l in st
            templates[k,:,:] = render(OrbAndCross(Square(), Star(l)))
            k += 1
        end
    elseif t == 1
        for l in st
            templates[k,:,:] = render(OrbAndCross(Triangle(), Star(l)))
            k += 1
        end
    elseif p[1] > 0
        for r in polygon[2]
            for l in st
                templates[k,:,:] = render(OrbAndCross(Polygon(r), Star(l)))
                k += 1
            end
        end
    end

    save("temp.jld", "temp", templates)
end 

function view_temp(c::Int, s::Int, t::Int, p::Tuple{Int, AbstractVector}=(0,[]), st::AbstractVector)
    temp = load("temp.jld")["temp"]

    dimₒ = c + s + t + p[1] 
    dimᵤ = size(star)

    plt = plot(layout=(dimₒ,dimᵤ))
    for i in 1:dimₒ*dimᵤ
        plot!(plt, Gray.(temp[i,:,:]), subplot=i, xaxis=false, yaxis=false, grid=false); # plot each set in a different subplot
    end
    display(plt);
end;
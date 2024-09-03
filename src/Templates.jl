function generator_temp(circle::Int, square::int, triangle::Int, polygon::Tuple{Int, AbstractArray}, star::AbstractArray)

    # Image size of the temp 
    image_size = 128;

    dimₒ = circle + square + triangle + polygon[1] 
    dimᵤ = size(star)

    templates = zeros(dimₒ*dimᵤ, image_size, image_size)

    k = 1
    if circle == 1
        for l in star
            templates[k,:,:] = render(OrbAndCross(Circle(), Star(l)))
            k += 1
        end
    elseif square == 1
        for l in star
            templates[k,:,:] = render(OrbAndCross(Square(), Star(l)))
            k += 1
        end
    elseif triangle == 1
        for l in star
            templates[k,:,:] = render(OrbAndCross(Triangle(), Star(l)))
            k += 1
        end
    elseif polygon[1] > 0
        for r in polygon[2]
            for l in star
                templates[k,:,:] = render(OrbAndCross(Polygon(r), Star(l)))
                k += 1
            end
        end
    end

    save("temp.jld", "temp", templates)
end 
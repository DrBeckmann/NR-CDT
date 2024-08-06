function gen_circle(image, seed, size, thick)
    im = Gray.(image);

    draw!(im, Ellipse(CirclePointRadius(seed[:1],seed[:1]+46+size, size; thickness = Int(round(thick/2)), fill = false)))
    return im
end;

function gen_square(image, seed, size, thick)
    im = Gray.(image);

    for i in 0:Int(round(thick/10))-1 
        draw!(im, Polygon(RectanglePoints(seed[:1]-size+i,seed[:2]+i, seed[:1]+size-i,seed[:2]+2*size-i)));
    end 
    return im
end;

function gen_bar(image, seed, length, thick)
    im = Gray.(image);

    d = Int(round(thick/10/2));
    p1 = [seed[:1], seed[:2]];
    p2 = [seed[:1], seed[:2]-length];
    for i in -d:d
        p11 = p1 + [i,0];
        p22 = p2 + [i,0];
        draw!(im, LineSegment(Point(p11[:1], p11[:2]),Point(p22[:1], p22[:2])));
    end
    return im
end;

function gen_cross(image, seed, height, branch, width, thick)
    im = Gray.(image);

    p1 = [seed[:1], seed[:2]];
    p2 = [seed[:1],seed[:2]-height];
    p3 = [branch[:1]-width,branch[:2]];
    p4 = [branch[:1]+width,branch[:2]];
    d = Int(round(thick/10/2));
    for i in -d:d
        p11 = p1 + [i,0];
        p22 = p2 + [i,0];
        p33 = p3 + [0,i];
        p44 = p4 + [0,i];
        draw!(im, LineSegment(Point(p11[:1], p11[:2]),Point(p22[:1], p22[:2])));
        draw!(im, LineSegment(Point(p33[:1], p33[:2]),Point(p44[:1], p44[:2])));
    end
    return im
end;

function gen_triangle(image, seed, size, thick)
    im = Gray.(image);

    p1 = [seed[:1], seed[:2]];
    p2 = [seed[:1]-size,seed[:2]+2*size];
    p3 = [seed[:1]+size,seed[:2]+2*size];
    d = Int(round(thick/10/2));
    for i in -d:d
        p11 = p1 + [i,0];
        p22 = p2 + [0,i];
        p33 = p3 + [i,i];
        draw!(im, LineSegment(Point(p22[:1], p22[:2]),Point(p33[:1], p33[:2])));
        draw!(im, LineSegment(Point(p33[:1], p33[:2]),Point(p11[:1], p11[:2])));
        draw!(im, LineSegment(Point(p11[:1], p11[:2]),Point(p22[:1], p22[:2])));
    end
    return im
end

function gen_star(image, seed, height, branch, width, thick)
    im = Gray.(image);

    p1 = [seed[:1], seed[:2]];
    p2 = [seed[:1], seed[:2]-height]
    p3 = [branch[:1]-width, branch[:2]+width]
    p4 = [branch[:1]-width, branch[:2]-width]
    p5 = [branch[:1]+width, branch[:2]-width]
    p6 = [branch[:1]+width, branch[:2]+width]
    d = Int(round(thick/10/2));
    for i in -d:d
        p11 = p1 + [i,0];
        p22 = p2 + [i,0];
        p33 = p3 + [i,0];
        p44 = p4 + [i,0];
        p55 = p5 + [i,0];
        p66 = p6 + [i,0];
        draw!(im, LineSegment(Point(p33[:1], p33[:2]),Point(p55[:1], p55[:2])));
        draw!(im, LineSegment(Point(p44[:1], p44[:2]),Point(p66[:1], p66[:2])));
    end
    return im
end;

function resize(image, size)
    im = imresize(image, (size,size))
    return im
end;

function viewer_temp()
    temp = load("temp.jld")["temp"]
    plt = plot(layout=(3,3))
    for i in 1:9
        plot!(plt, Gray.(temp[i,:,:]), subplot=i, xaxis=false, yaxis=false, grid=false); # plot each set in a different subplot
    end
    display(plt);
end;

function generator_temp()

    # Black background
    bg = Array{Gray{N0f8},2}(zeros(512,512));

    # Thickness
    thick = 50;

    # Image size of the temp 
    image_size = 128;

    # Parameter
    seed = [256, 304];
    base_size = 80;
    bar_length = 214;
    cross_height = 214;
    cross_branch = [256,150];
    cross_length = 56;
    star_height = 214;
    star_branch = [256,150];
    star_width = 56;

    # Generate the single forms
    circle = gen_circle(bg, seed, base_size, thick);
    rectangle = gen_square(bg, seed, base_size, thick);
    triangle = gen_triangle(bg, seed, base_size, thick);
    bar = gen_bar(bg, seed, bar_length, thick);
    cross = gen_cross(bg, seed, cross_height, cross_branch, cross_length, thick);
    star = gen_star(bg, seed, star_height, star_branch, star_width, thick);

    # Create images
    circle_bar = circle + bar;
    circle_cross = circle + cross;
    circle_star = circle + star + bar;
    rectangle_bar = rectangle + bar;
    rectangle_cross = rectangle + cross;
    rectangle_star = rectangle + star + bar;
    triangle_bar = triangle + bar;
    triangle_cross = triangle + cross;
    triangle_star = triangle + star + bar;

    # Resize images
    circle_bar = resize(circle_bar, image_size);
    circle_cross = resize(circle_cross, image_size);
    circle_star = resize(circle_star, image_size);
    rectangle_bar = resize(rectangle_bar, image_size);
    rectangle_cross = resize(rectangle_cross, image_size);
    rectangle_star = resize(rectangle_star, image_size);
    triangle_bar = resize(triangle_bar, image_size);
    triangle_cross = resize(triangle_cross, image_size);
    triangle_star = resize(triangle_star, image_size);

    # Create template
    templates = zeros(9, image_size, image_size)
    templates[1,:,:] = circle_bar
    templates[2,:,:] = circle_cross
    templates[3,:,:] = circle_star
    templates[4,:,:] = rectangle_bar
    templates[5,:,:] = rectangle_cross
    templates[6,:,:] = rectangle_star
    templates[7,:,:] = triangle_bar
    templates[8,:,:] = triangle_cross
    templates[9,:,:] = triangle_star
    #templates = [circle_bar, circle_cross, circle_star, rectangle_bar, rectangle_cross, rectangle_star, triangle_bar, triangle_cross, triangle_star];
    save("temp.jld", "temp", templates)
end 
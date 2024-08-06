function random_mask(shape, num_pos_min, num_pos_max, width, length_max, seed)
    
    Random.seed!(seed)

    mask = zeros(shape)
    num_pos = rand(num_pos_min: num_pos_max + 1)
    for i in 1:num_pos
        row_pos = rand(10: shape[:1]-10)
        col_pos = rand(10: shape[:2]-10)
        length = rand(1: length_max + 1)
        case = rand(0:4)
        if case == 0
            # vertical
            mask[row_pos: row_pos + length, col_pos: col_pos + width] .= 1
        elseif case == 1
            # horizontal
            mask[row_pos: row_pos + width, col_pos: col_pos + length] .= 1
        elseif case == 2
            # diagonal down
            for i in 1:length 
                try
                    mask[row_pos + i, col_pos + i*Int(round(width/2)): col_pos + i*Int(round(width/2)) + width] .= 1
                catch
                    break
                end
            end
        elseif case == 3
            # diagonal up
            for i in 1:length 
                try
                    mask[row_pos - i, col_pos + i*Int(round(width/2)): col_pos + i*Int(round(width/2)) + width] .= 1
                catch
                    break
                end
            end
        end
    end
    return mask
end;

function random_image_distortion(image, image_size, scale_bounds, angle_bounds, shear_bounds, shift_bounds_x, shift_bounds_y, noise_bounds, seed)

    Random.seed!(seed);

    # Scale
    if scale_bounds[2] > scale_bounds[1]
        scale = round(rand(Uniform(scale_bounds[1], scale_bounds[2])), digits=2);
        img = imresize(image, ratio=scale);
        if scale <= 1.0
            image = zeros((image_size,size(image)[2]));
            a = Int(round((image_size - size(img)[2])/2));
            image[a+1:size(img)[1]+a,a+1:size(img)[2]+a] = img;
        else
            image = zeros(image_size, image_size);
            x = Int(round(size(img)[1]/2));
            y = Int(round(size(img)[2]/2));
            a = Int(round(size(image)[1]/2));
            image = img[x-a:x+a, y-a:y+a];
        end
    end
    # Rotate
    if angle_bounds[2] > angle_bounds[1]
        image = Array{Gray{N0f8},2}(image)
        angle = rand(Uniform(angle_bounds[1], angle_bounds[2]));
        trfm = recenter(RotMatrix(angle), (Int(round(size(image)[1]/2)), Int(round(size(image)[2]/2))));
        image = imresize(warp(image, trfm), (image_size, image_size));
        image = Gray.(image)
    end
    # Shear
    if shear_bounds[2] > shear_bounds[1]
        image = imresize(augment(image, ShearX(shear_bounds[1]:shear_bounds[2])), (image_size, image_size));
        image = imresize(augment(image, ShearY(shear_bounds[1]:shear_bounds[2])), (image_size, image_size));
    end
    # Shift
    if shift_bounds_x[2] > shift_bounds_x[1]
        shift_x = rand(shift_bounds_x[1]: shift_bounds_x[2]);
    else
        shift_x = 0
    end
    if shift_bounds_y[2] > shift_bounds_y[1]
        shift_y = rand(shift_bounds_y[1]: shift_bounds_y[2]);
    else
        shift_y = 0
    end
    image = circshift(image, (shift_x, shift_y));
    image = imresize(image, (image_size,image_size));

    # Noise
    if noise_bounds != 0
        mask = random_mask(size(image), noise_bounds[1], noise_bounds[2], noise_bounds[3], noise_bounds[4], seed);
        image = image + mask
        image = imresize(image, (image_size,image_size));
    end
    return image
end;

function gen_dataset(template, label, image_size, size_data, parameters, seed)

    scale_bounds = parameters[1];
    angle_bounds = parameters[2];
    shear_bounds = parameters[3];
    shift_bounds_x = parameters[4];
    shift_bounds_y = parameters[5];
    if size(parameters)[1] > 5
        noise_bounds = parameters[6];
    else 
        noise_bounds = 0;
    end

    size_temp = size(template)[1]
    dataset = zeros(size_data*size_temp, image_size, image_size);
    labels = zeros(size_data*size_temp);
    for i in 1:size_data
        for j in 1:size_temp
            img = random_image_distortion(template[j,:,:], image_size, scale_bounds, angle_bounds, shear_bounds, shift_bounds_x, shift_bounds_y, noise_bounds, seed+i);
            dataset[j + (i-1)*size_temp,:,:] = img;
            labels[j + (i-1)*size_temp] = label[j];
        end
    end
    
    return dataset, labels
end;

function create_data(samp, image_size, size_data, random_seed)
    templates = load("temp.jld")["temp"];
    label = range(1, size(templates)[1], step=1);

    temp = zeros(size(samp)[1], image_size, image_size)
    lab = zeros(size(samp)[1])
    k = 1
    for i in samp
        temp[k,:,:] = templates[i,:,:]  # convert(Array{Float64}, templates[i])
        lab[k] = label[i]
        k = k+1
    end

    # Translation
    # parameters = [(1,1),(-0.,0.),(-0.,0.),(-10,10),(-25,25),(4,20,2,5)] # with noise
    # parameters = [(1,1),(-0.,0.),(-0.,0.),(-10,10),(-25,25)] # without noise
    # Scaling, translation
    # parameters = [(0.75,1.1),(-0.,0.),(-0.,0.),(-10,10),(-25,25),(4,20,2,5)] # with noise
    # parameters = [(0.75,1.1),(-0.,0.),(-0.,0.),(-10,10),(-25,25)] # without noise
    # Scaling, rotation, translation
    # parameters = [(0.5,1),(-5.,5.),(-0.,0.),(-10,10),(-10,10),(10,25,3,9)] # with noise
    parameters = [[0.5,1],[-5.,5.],[-0,0],[-10,10],[-10,10]]; # without noise
    # Scaling, rotation, shear, translation
    # parameters = [(0.5,1),(-5.,5.),(-0.25,0.25),(-10,10),(-10,10),(10,25,3,9)] # with noise
    # parameters = [(0.5,1),(-5.,5.),(-0.25,0.25),(-10,10),(-10,10)] # without noise
    
    dataset, labels =  gen_dataset(temp, lab, image_size, size_data, parameters, random_seed);

    dataset = round.(5*dataset)
    dataset[dataset .> 1] .= 1

    save("data.jld", "data", dataset)
    save("labels.jld", "labels", labels)
end;

function view_data()
    data = load("data.jld")["data"];

    length = size(data)[1]
    sel = rand(1:length, 9)

    # Plot images
    plt = plot(layout=(3,3))
    for i in 1:9
        # plot each set in a different subplot
        plot!(plt, Gray.(data[sel[i],:,:]), subplot=i, xaxis=false, yaxis=false, grid=false); 
    end
    display(plt);
end;
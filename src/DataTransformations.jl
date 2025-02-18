module DataTransformations

using Random, Images
using Augmentor
# using Statistics, ImageTransformations, Distributions, Rotations, CoordinateTransformations

struct RandomAffineTransformation
    scaling_x::Tuple{Float64, Float64}
    scaling_y::Tuple{Float64, Float64}
    rotating::Tuple{Float64, Float64}
    shearing_x::Tuple{Float64, Float64}
    shearing_y::Tuple{Float64, Float64}
    translating_x::Tuple{Int64, Int64}
    translating_y::Tuple{Int64, Int64}
    function RandomAffineTransformation(scx, scy, rot, shx, shy, tax, tay)
        if !(0 <= scx[1] <= scx[2]) 
            error("inconsistent x scaling")
        elseif !(0 <= scy[1] <= scy[2]) 
            error("inconsistent y scaling")
        elseif !(rot[1] <= rot[2]) 
            error("inconsistent rotating")
        elseif !(shx[1] <= shx[2]) 
            error("inconsistent x shearing")
        elseif !(shy[1] <= shy[2]) 
            error("inconsistent y shearing")
        elseif !(tax[1] <= tax[2]) 
            error("inconsistent x translating")
        elseif !(tay[1] <= tay[2]) 
            error("inconsistent y translating")
        end
        return new(scx, scy, rot, shx, shy, tax, tay)
    end
end

function RandomAffineTransformation(;
    scale_x::Tuple{Float64, Float64}=(1.0, 1.0),
    scale_y::Tuple{Float64, Float64}=(1.0, 1.0),
    rotate::Tuple{Float64, Float64}=(0.0, 0.0),
    shear_x::Tuple{Float64, Float64}=(0.0, 0.0),
    shear_y::Tuple{Float64, Float64}=(0.0, 0.0),
    shift_x::Tuple{Int64, Int64}=(0, 0),
    shift_y::Tuple{Int64, Int64}=(0, 0)
)
    return RandomAffineTransformation(scale_x, scale_y, rotate, shear_x, shear_y, shift_x, shift_y)
end

function (A::RandomAffineTransformation)(image::AbstractMatrix)
    I = Gray{Float64}.(image)
    I = scaling(I, randu(A.scaling_x), randu(A.scaling_y))
    I = rotate(I, randu(A.rotating))
    I = shear(I, randu(A.shearing_x), randu(A.shearing_y))
    I = translate(I, randi(A.translating_x), randi(A.translating_y))
    return I
end

function randu(bounds::Tuple{Float64, Float64})
    (a, b) = bounds
    return (b - a) * rand() + a
end

function randi(bounds::Tuple{Int64, Int64})
    (a, b) = bounds
    return rand(a:b)
end

function scaling(I::Matrix{Gray{Float64}}, sx::Float64, sy::Float64)
    return augment(I, Zoom(sy, sx))
end

function rotate(I::Matrix{Gray{Float64}}, α::Float64)
    s = size(I)
    return augment(I, Rotate(α) |> CropSize(s))
end

function shear(I::Matrix{Gray{Float64}}, α::Float64, β::Float64)
    s = size(I)
    return augment(I, ShearX(α) |> ShearY(β) |> CropSize(s))
end

function translate(I::Matrix{Gray{Float64}}, x::Int64, y::Int64)
    return circshift(I, (-y, x))
end




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


#=
    # Noise
    if noise_bounds != 0
        mask = random_mask(size(image), noise_bounds[1], noise_bounds[2], noise_bounds[3], noise_bounds[4], seed);
        image = image + mask
        image = imresize(image, (image_size,image_size));
    end
    return image
=#

function nonlinear_distortion(img::AbstractMatrix, amplitude::Float64, frequency::Float64)
    rows, cols = size(img)
    distorted_img = zeros(eltype(img), rows, cols)

	amplitude = amplitude*rand(1)[1]
	frequency = frequency*rand(1)[1]

    for i in 1:rows
        for j in 1:cols
            # Compute the new coordinates using a sinusoidal distortion
            new_i = clamp(i + amplitude * sin(2π * frequency * j / cols), 1, rows)
            new_j = clamp(j + amplitude * cos(2π * frequency * i / rows), 1, cols)

            # Perform bilinear interpolation
            top = floor(Int, new_i)
            bottom = ceil(Int, new_i)
            left = floor(Int, new_j)
            right = ceil(Int, new_j)

            weight_tl = (bottom - new_i) * (right - new_j)
            weight_tr = (bottom - new_i) * (new_j - left)
            weight_bl = (new_i - top) * (right - new_j)
            weight_br = (new_i - top) * (new_j - left)

            # Assign interpolated value
            distorted_img[i, j] = 
                weight_tl * img[top, left] +
                weight_tr * img[top, right] +
                weight_bl * img[bottom, left] +
                weight_br * img[bottom, right]
        end
    end

    return distorted_img
end

function random_int_pair(range::UnitRange{Int})
    x = rand(range)
    y = rand(range)
    return (x, y)
end

function impulsive_distortion(img::AbstractMatrix, freg::Float64, amp::Int, num::Int)
    rows, cols = size(img)
    distorted_img = zeros(eltype(img), rows, cols)


	for k in 1:num
		(i,j) = random_int_pair(amp+1:cols-amp-1)
		for k in -amp:amp
			for l in -amp:amp
				# distorted_img[i+k,j+l] += 1/(k^2*freg+l^2*freg+1)
				distorted_img[i+k,j+l] += exp(-(k^2*freg+l^2*freg))
			end
		end
	end

	distorted_img += img

    return distorted_img
end

function temp_distortion(temp, noise)
	nonlinear_noise = noise[1]
	impulsive_noise = noise[2]
    bar_noise = noise[3]

	if nonlinear_noise[1] != 0
		temp = nonlinear_distortion(temp, nonlinear_noise[1], nonlinear_noise[2])
	end

	if impulsive_noise[1] != 0
		temp = impulsive_distortion(temp, impulsive_noise[1], Int(impulsive_noise[2]), Int(impulsive_noise[3]))
	end

    if bar_noise[1] != 0
		temp = bar_distortion(temp)
	end

	return temp
end

function bar_distortion(img::AbstractMatrix)
    size_img = size(img)[1]
    for k in 1:10
        k = rand(10:size_img-10)
        img[:,k:k+3] = zeros(size_img,4)
    end
    return img
end

function gen_dataset(template, label, image_size, size_data, parameters, parameters_non_aff, seed)

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
    for j in 1:size_temp
        for i in 1:size_data
            temp_eps = temp_distortion(template[j,:,:], parameters_non_aff)
            img = random_image_distortion(temp_eps, image_size, scale_bounds, angle_bounds, shear_bounds, shift_bounds_x, shift_bounds_y, noise_bounds, seed+i);
            dataset[i + (j-1)*size_data,:,:] = img;
            labels[i + (j-1)*size_data] = label[j];
        end
    end
    
    return dataset, labels
end;

function gen_dataset_nonaffine(template, label, image_size, size_data, parameters)

    size_temp = size(template)[1]
    dataset = zeros(size_data*size_temp, image_size, image_size);
    labels = zeros(size_data*size_temp);
    for j in 1:size_temp
        for i in 1:size_data
            img = temp_distortion(template[j,:,:], [parameters[1], parameters[2], parameters[3]])
            dataset[i + (j-1)*size_data,:,:] = img;
            labels[i + (j-1)*size_data] = label[j];
        end
    end
    
    return dataset, labels
end;

function gen_dataset_mnist(template, image_size, parameters, parameters_non_aff, seed)

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

    temp_eps = temp_distortion(template, parameters_non_aff)
    noise_data_imag = random_image_distortion(temp_eps, image_size, scale_bounds, angle_bounds, shear_bounds, shift_bounds_x, shift_bounds_y, noise_bounds, seed)
    
    return noise_data_imag
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
    plt = []
    for i in 1:9
        # plot each set in a different subplot
        push!(plt, heatmap(data[sel[i],:,:], aspect_ratio=:equal, axis=([], false), cbar=false))
        # plot!(plt, Gray.(data[sel[i],:,:]), subplot=i, xaxis=false, yaxis=false, grid=false); 
    end
    display(plt)
    plot(plt...)
end;

end
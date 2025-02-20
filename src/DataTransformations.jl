module DataTransformations

using Random, Images, Luxor
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

abstract type AbstractNoise end

struct MikadoNoise <: AbstractNoise
    sticks::Tuple{Int64, Int64}
    length::Tuple{Float64, Float64}
    width::Tuple{Float64, Float64}
    function MikadoNoise(s, l, w)
        if !(0 <= s[1] <= s[2]) 
            error("inconsistent stick number")
        elseif !(0 <= l[1] <= l[2]) 
            error("inconsistent length")
        elseif !(0 <= w[1] <= w[2]) 
            error("inconsistent width")
        end
        return new(s, l, w)
    end
end

function MikadoNoise(;
    sticks::Tuple{Int64, Int64}=(10, 15),
    length::Tuple{Float64, Float64}=(0.125, 0.25),
    width::Tuple{Float64, Float64}=(2.0, 2.0)
)
    return MikadoNoise(sticks, length, width)
end

function (N::MikadoNoise)(image::AbstractMatrix)
    I = Gray{Float64}.(image)
    E = render_noise(N, size(I))
    return max.(I, E)
end

struct SaltNoise <: AbstractNoise
    dots::Tuple{Int64, Int64}
    width::Tuple{Float64, Float64}
    function SaltNoise(d, w)
        if !(0 <= d[1] <= d[2]) 
            error("inconsistent dot number")
        elseif !(0 <= w[1] <= w[2]) 
            error("inconsistent width")
        end
        return new(d, w)
    end
end

function SaltNoise(;
    dots::Tuple{Int64, Int64}=(10, 15),
    width::Tuple{Float64, Float64}=(1e-5, 1e-5)
)
    return SaltNoise(dots, width)
end

function (N::SaltNoise)(image::AbstractMatrix)
    I = Gray{Float64}.(image)
    E = render_noise(N, size(I))
    return max.(I, E)
end

struct BarNoise <: AbstractNoise
    bars::Tuple{Int64, Int64}
    width::Tuple{Float64, Float64}
    function BarNoise(b, w)
        if !(0 <= b[1] <= b[2]) 
            error("inconsistent bar number")
        elseif !(0 <= w[1] <= w[2]) 
            error("inconsistent width")
        end
        return new(b, w)
    end
end

function BarNoise(;
    bars::Tuple{Int64, Int64}=(2, 5),
    width::Tuple{Float64, Float64}=(2.0, 2.0)
)
    return BarNoise(bars, width)
end

function (N::BarNoise)(image::AbstractMatrix)
    I = Gray{Float64}.(image)
    E = 1.0 .- render_noise(N, size(I))
    return min.(I, E)
end

function render_noise(noise::AbstractNoise, size::Tuple{Integer,Integer})
    initiate_luxor_drawing(size)
    luxor_draw(noise)
    return extract_luxor_drawing()
end

function initiate_luxor_drawing(size::Tuple{Integer,Integer})
    (y, x) = size
    Drawing(x, y, :png)
    origin()
    background("black")
    sethue("white")
    Luxor.scale(x / 2, y / 2)
    return nothing
end

function luxor_draw(noise::MikadoNoise)
    for _ in 1:randi(noise.sticks) 
        luxor_draw_stick(noise)
    end
    return nothing
end

function luxor_draw_stick(noise::MikadoNoise)
    direction = rand([:vertical, :horizontal, :diagonal_up, :diagonal_down])
    length = randu(noise.length)
    setline(randu(noise.width))
    stick_path(direction, length)
    strokepath()    
end

function stick_path(direction::Symbol, length::Float64)
    (x, y) = (2 - length) .* (rand(2) .- 0.5) 
    if direction==:vertical
        move(Point(x, y + length / 2))
        line(Point(x, y - length / 2))
    elseif direction==:horizontal
        move(Point(x + length / 2, y))
        line(Point(x - length / 2, y))
    elseif direction==:diagonal_up
        move(Point(x - length / 2sqrt(2), y - length / 2sqrt(2)))
        line(Point(x + length / 2sqrt(2), y + length / 2sqrt(2)))
    elseif direction==:diagonal_down
        move(Point(x - length / 2sqrt(2), y + length / 2sqrt(2)))
        line(Point(x + length / 2sqrt(2), y - length / 2sqrt(2)))
    end   
end

function luxor_draw(noise::SaltNoise)
    for _ in 1:randi(noise.dots) 
        luxor_draw_dot(noise)
    end
    return nothing
end

function luxor_draw_dot(noise::SaltNoise)
    width = randu(noise.width)
    (x, y) = (2 - width) .* (rand(2) .- 0.5) 
    circle(Point(x, y), width / 2; action=:fill)
end

function luxor_draw(noise::BarNoise)
    for _ in 1:randi(noise.bars) 
        luxor_draw_bar(noise)
    end
    return nothing
end

function luxor_draw_bar(noise::BarNoise)
    width = randu(noise.width)
    x = (2 - width) * (rand() - 0.5)
    setline(width)
    move(Point(x, 1.0))
    line(Point(x, -1.0))
    strokepath()
end

function extract_luxor_drawing()
    image = image_as_matrix()
    finish()
    return Gray{Float64}.(image)
end

struct ElasticNoise
    amplitude_x::Tuple{Float64, Float64}
    amplitude_y::Tuple{Float64, Float64}
    frequency_x::Tuple{Float64, Float64}
    frequency_y::Tuple{Float64, Float64}
    phase_x::Tuple{Float64, Float64}
    phase_y::Tuple{Float64, Float64}
    function ElasticNoise(ax, ay, fx, fy, px, py)
        if !(0 <= ax[1] <= ax[2]) 
            error("inconsistent x amplitude")
        elseif !(0 <= ay[1] <= ay[2]) 
            error("inconsistent y amplitude")
        elseif !(0 <= fx[1] <= fx[2]) 
            error("inconsistent x frequency")
        elseif !(0 <= fy[1] <= fy[2]) 
            error("inconsistent y frequency")
        elseif !(px[1] <= px[2]) 
            error("inconsistent x phase")
        elseif !(py[1] <= py[2]) 
            error("inconsistent y phase")
        end
        return new(ax, ay, fx, fy, px, py)
    end
end

function ElasticNoise(;
    amplitude_x::Tuple{Float64, Float64}=(0.0, 0.0),
    amplitude_y::Tuple{Float64, Float64}=(0.0, 0.0),
    frequency_x::Tuple{Float64, Float64}=(1.0, 1.0),
    frequency_y::Tuple{Float64, Float64}=(1.0, 1.0),
    phase_x::Tuple{Float64, Float64}=(0.0, 0.0),
    phase_y::Tuple{Float64, Float64}=(0.0, 0.0),
)
    return ElasticNoise(amplitude_x, amplitude_y, frequency_x, frequency_y, phase_x, phase_y)
end

function (N::ElasticNoise)(image::AbstractMatrix)
    I = Gray{Float64}.(image)
    dim_y, dim_x = size(I)
    J = zero(I)
	ax = randu(N.amplitude_x)
    ay = randu(N.amplitude_y)
	fx = randu(N.frequency_x)
	fy = randu(N.frequency_y)
    px = randu(N.phase_x)
    py = randu(N.phase_y)
    for j in axes(I, 1), k in axes(I, 2)
        x = k + ax * sin(2π * fx * j / dim_y + px)
        y = j + ay * cos(2π * fy * k / dim_x + py)
        J[j, k] = periodic_bilinear_interpolation(x, y, I) 
    end
    return J
end

function periodic_bilinear_interpolation(x::Float64, y::Float64, I::Matrix{Gray{Float64}})
    (dim_y, dim_x) = size(I)
    x₁ = floor(Int64, x)
    x₂ = x₁ + 1
    y₁ = floor(Int64, y)
    y₂ = y₁ + 1    
    z = (x₂ - x) * (y₂ - y) * I[periodic_index(y₁, dim_y), periodic_index(x₁, dim_x)]
    z += (x₂ - x) * (y - y₁) * I[periodic_index(y₂, dim_y), periodic_index(x₁, dim_x)]
    z += (x - x₁) * (y₂ - y) * I[periodic_index(y₁, dim_y), periodic_index(x₂, dim_x)]
    z += (x - x₁) * (y - y₁) * I[periodic_index(y₂, dim_y), periodic_index(x₂, dim_x)]
    return z
end

function periodic_index(j::Int64, dim::Int64)
    return mod(j - 1, dim) + 1
end

function extend_image(image::AbstractMatrix, shape::Tuple{Int64, Int64})
    (dim_y, dim_x) = size(image)
    if dim_y > shape[1] || dim_x > shape[2]
        error("dimension mismatch")
    end
    I = zeros(Gray{Float64}, shape)
    id_x = max((shape[2] - dim_x) ÷ 2, 1)
    id_y = max((shape[1] - dim_y) ÷ 2, 1)
    I[id_y:(id_y + dim_y - 1), id_x:(id_x + dim_x - 1)] .= Gray{Float64}.(image)
    return I
end

function extend_image(image::AbstractMatrix, shape::Int64)
    return extend_image(image, (shape, shape))
end


function generate_academic_classes(images::AbstractArray; class_size::Int64=10)
    num = length(images)
    classes = []
    labels = []
    for k in 1:num
        for l in 1:class_size
            append!(classes, images[k,:,:])
            append!(labels, k)
        end
    end
    return shuffle_data(classes::AbstractArray, labels::AbstractArray)
end

function shuffle_data(classes::AbstractArray, labels::AbstractArray)
    dim = length(classes)
    p = shuffle(1:dim)
    classes = classes[p]
    labels = labels[p]
    return classes, labels
end



###### -> move to pluto
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
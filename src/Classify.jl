using NormalizedRadonCDT.radon_cdt
using MLDatasets, Random, Statistics
using Interpolations: LinearInterpolation as LinInter
using LIBSVM, LIBLINEAR

function prepare_data(data, num_angles, scale_radii, width)
    ref = ones(size(data[1,:,:]));
    data1 = rcdt(ref, data[1,:,:], num_angles, scale_radii, width)[1];
    data_transformed = zeros(size(data)[1], size(data1)[1], size(data1)[2]);
    data_transformed[1,:,:] = data1;
    for i in 2:size(data)[1]
        data_transformed[i,:,:] = rcdt(ref, data[i,:,:], num_angles, scale_radii, width)[1];
    end
    nsamples, nx, ny = size(data_transformed);
    data_transformed = reshape(data_transformed, (nsamples, nx * ny));

    return data_transformed
end;

function shuffle_data(data, labels, random_seed)
    rng = Random.default_rng(random_seed);
    ind = range(1, size(data)[1]);
    ind = shuffle(rng, ind);
    data = data[floor.(ind),:,:];
    labels = labels[floor.(ind)];
    
    return data, labels
end;

function split_data(data, labels, split)
    lab = unique(labels);
    train_ind = zeros(0);
    test_ind = zeros(0);
    for l in lab
        ind = findall(x->x==l, labels);
        ind_split = Int(round(size(ind)[1]*split));
        append!(train_ind, floor.(ind[1:ind_split]));
        append!(test_ind, floor.(ind[ind_split:end]));
    end
    sort!(train_ind);
    sort!(test_ind);
    data_train = data[Int.(train_ind),:,:];
    labels_train = labels[Int.(train_ind)];
    data_test = data[Int.(test_ind),:,:];
    labels_test = labels[Int.(test_ind)];

    return data_train, labels_train, data_test, labels_test
end;

function split_data_mnist(data, labels, split)
    lab = unique(labels);
    train_ind = zeros(0);
    test_ind = zeros(0);
    for l in lab
        ind = findall(x->x==l, labels);
        ind_split = Int(round(size(ind)[1]*split));
        append!(train_ind, floor.(ind[1:ind_split]));
        append!(test_ind, floor.(ind[ind_split:end]));
    end
    sort!(train_ind);
    sort!(test_ind);
    data_train = data[:,:,Int.(train_ind)];
    labels_train = labels[Int.(train_ind)];
    data_test = data[:,:,Int.(test_ind)];
    labels_test = labels[Int.(test_ind)];

    return data_train, labels_train, data_test, labels_test
end;

function classify_data_NRCDT(samp, image_size, size_data, random_seed, num_angles_rcdt, num_angles_rcdt_norm, scale_radii, noise, width)

    templates = load("temp.jld")["temp"]
    label = range(1, size(templates)[1])
    size_data_small = size_data

    #####
    #####
    # 
    # Generating the chosen tamps form "samp"-image tamps
    # 
    #
    #####
    #####

    temp = zeros(size(samp)[1], image_size, image_size)
    lab = zeros(size(samp)[1])
    k = 1
    for i in samp
        temp[k,:,:] = imresize(templates[i,:,:], image_size, image_size)    # imresize(convert(Array{Float64}, templates[i]), (image_size, image_size))
        lab[k] = label[i]
        k = k+1
    end
    
    temp_ext = zeros(size(temp)[1],2*image_size,2*image_size)

    for i in 1:size(samp)[1]
        image = zeros(2*image_size, 2*image_size)
        a = Int(round((2*image_size - size(temp[i,:,:])[1])/2))
        b = Int(round((2*image_size - size(temp[i,:,:])[2])/2))
        image[a:size(temp[i,:,:])[1]+a-1, b:size(temp[i,:,:])[2]+b-1] = temp[i,:,:]
        temp_ext[i,:,:] = image
    end
    temp = temp_ext

    
    # Plot of temp_ext
    plt1 = plot(layout=(1,size(samp)[1]), plot_title="choice of the synthetic template")
    # plt1 = []
    for i in 1:size(samp)[1]
        plot!(plt1, Gray.(temp[i,:,:]), subplot=i, xaxis=false, yaxis=false, grid=false); # plot each set in a different subplot
        # push!(plt1, heatmap(temp[i,:,:], aspect_ratio=:equal, axis=([], false), cbar=false))
    end
    display(plt1)

    savefig(plt1, "plt1.pdf")

    
    #####
    #####
    #
    # Generating the dataset from the chosen tamps
    #
    #
    #####
    #####

    if noise == 0
        parameters = [(0.75,1.25),(-1.0,1.0),(-5,5),(-15,15),(-15,15)] #noise: (4,20,2,5), (10,40,2,5)
    elseif noise == 0.5
        parameters = [(0.75,1.0),(0.,0.),(0.,0.),(-10,10),(-10,10)] #noise: (4,20,2,5), (10,40,2,5)
    elseif noise == 1
        parameters = [(0.75,1.0),(-0.5,0.5),(-5,5),(-10,10),(-10,10),(4,20,2,5)] #noise: (4,20,2,5), (10,40,2,5)
    elseif noise == 1.5
        parameters = [(0.75,1.0),(-0.5,0.5),(0.,0.),(-10,10),(-10,10)] #noise: (4,20,2,5), (10,40,2,5)
    end


    dataset, labels =  gen_dataset(temp, lab, 2*image_size, size_data_small, parameters, random_seed)


    size_data = size(dataset)[1]
    println("Size of data:", size_data)
    sel = rand(1:size_data, 9)

    # Plot random choice of dataimages
    plt2 = plot(layout=(3,3), plot_title="random choice of dataset")
    # plt2 = []
    for i in 1:9
        plot!(plt2, Gray.(dataset[sel[i],:,:]), subplot=i, xaxis=false, yaxis=false, grid=false); # plot each set in a different subplot
        # push!(plt2, heatmap(dataset[sel[i],:,:], aspect_ratio=:equal, axis=([], false), cbar=false))
    end
    display(plt2)
    plot(plt2)

    savefig(plt2, "plt2.pdf")

    #####
    #####
    #
    # RCDT --- Computation --- without normalization
    #
    #
    #####
    #####

    ref = ones(size(dataset[1,:,:])[1], size(dataset[1,:,:])[2])
    
    temp_rcdt1 = transpose(rcdt(ref, temp[1,:,:], num_angles_rcdt, scale_radii, width)[1])
    temp_rcdt = zeros(size(temp)[1], size(temp_rcdt1)[1], size(temp_rcdt1)[2])
    temp_rcdt[1,:,:] = temp_rcdt1
    for i in 2:size(temp)[1]
        temp_rcdt[i,:,:] = transpose(rcdt(ref, temp[i,:,:], num_angles_rcdt, scale_radii, width)[1])
    end
    #
    data_rcdt1 = transpose(rcdt(ref, dataset[1,:,:], num_angles_rcdt, scale_radii, width)[1])
    data_rcdt = zeros(size(dataset)[1], size(data_rcdt1)[1], size(data_rcdt1)[2])
    data_rcdt[1,:,:] = data_rcdt1
    for i in 2:size(dataset)[1]
        data_rcdt[i,:,:] = transpose(rcdt(ref, dataset[i,:,:], num_angles_rcdt, scale_radii, width)[1])
    end

    temp_rcdt_normalized = (temp_rcdt .- mean(temp_rcdt, dims=2))./sqrt.(var(temp_rcdt, dims=2))
    temp_rcdt_normalized = dropdims(maximum(temp_rcdt_normalized, dims=3), dims=3)
    data_rcdt_normalized = (data_rcdt .- mean(data_rcdt, dims=2))./sqrt.(var(data_rcdt, dims=2))
    data_rcdt_normalized = dropdims(maximum(data_rcdt_normalized, dims=3), dims=3)

    
    # Plot of results
    dd = size(data_rcdt)[2]
    plt3 = plot(layout=(size(samp)[1]+1, 2), plot_title="transform of unnormalized and normalized")
    for i in 1:size(samp)[1]
        for j in 1:size(dataset)[1]
            if labels[j] == lab[i]
                # plot each set in a different subplot
                plot!(plt3, data_rcdt[j,:,1], subplot=(i-1)*2+1, legend=false, yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
                plot!(plt3, data_rcdt_normalized[j,:], subplot=(i-1)*2+2, legend=false, yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));

                plot!(plt3, data_rcdt[j,:,1], subplot=size(samp)[1]*2+1, legend=false, yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
                plot!(plt3, data_rcdt_normalized[j,:], subplot=size(samp)[1]*2+2, legend=false, yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
            end
        end
    end
    for i in 1:size(samp)[1]
        plot!(plt3, temp_rcdt[i,:,1], subplot=(i-1)*2+1, legend=false, color=:black, yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
        plot!(plt3, temp_rcdt_normalized[i,:], subplot=(i-1)*2+2, legend=false, color=:black, yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));

        plot!(plt3, temp_rcdt[i,:,1], subplot=size(samp)[1]*2+1, legend=false, color=:black, yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
        plot!(plt3, temp_rcdt_normalized[i,:], subplot=size(samp)[1]*2+2, legend=false, color=:black, yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
    end
    display(plt3);

    savefig(plt3, "plt3.pdf")
    
    #####
    #####
    #
    # Generating training data
    #
    #
    #####
    #####


    split = Int(size_data_small/10)

    #####
    #####
    #
    # Euclidean --- Computation
    #
    #
    #####
    #####

    acc_euclid = zeros(10)
    for i in 1:10
        split_range = Array((i-1)*split+1:i*split)
        for k in 2:size(samp)[1]
            append!(split_range,  Array((k-1)*size_data_small+(i-1)*split+1:(k-1)*size_data_small+i*split))
        end
        train_data = dataset[split_range,:,:]
        train_labels = labels[split_range]

        test_range = Array(1:size_data)
        test_range = filter(e->!(e in split_range),test_range)

        test_data = dataset[test_range,:,:]
        test_labels = labels[test_range]

        train_data_reshaped = reshape(train_data, (size(train_data)[1], size(train_data)[2] * size(train_data)[3]));

        clf_euclid_model = linear_train(train_labels, transpose(train_data_reshaped), solver_type=LIBLINEAR.L2R_L2LOSS_SVC) #, verbose=false);

        test_data_reshaped = reshape(test_data, (size(test_data)[1], size(test_data)[2] * size(test_data)[3]));

        pred_euclid = linear_predict(clf_euclid_model, transpose(test_data_reshaped))[1];

        acc_euclid[i] = mean(test_labels .== pred_euclid)
    end     


    println("=============================================================================")
    println("Acc. of Euclidean : \t", mean(acc_euclid), "+/-", std(acc_euclid))
    println("-----------------------------------------------------------------------------")


    #####
    #####
    #
    # RCDT --- Computation --- without normalization
    #
    #
    #####
    #####

    #==
    acc_rcdt = zeros(0)
    acc_rcdt_std = zeros(0)
    # ik = 1
    angle_range = (num_angles_rcdt_norm == 1) ? [0.0] : LinRange(0.0, π, num_angles_rcdt_norm)
    for l in angle_range
        acc_rcdt_k = zeros(10)
        for i in 1:10
            split_range = Array((i-1)*split+1:i*split)
            for k in 2:size(samp)[1]
                append!(split_range,  Array((k-1)*size_data_small+(i-1)*split+1:(k-1)*size_data_small+i*split))
            end
            train_data = dataset[split_range,:,:]
            train_labels = labels[split_range]
    
            test_range = Array(1:size_data)
            test_range = filter(e->!(e in split_range),test_range)
    
            test_data = dataset[test_range,:,:]
            test_labels = labels[test_range]

            train_data_transformed = prepare_data(train_data, l, scale_radii, width)

            clf_rcdt_model = linear_train(train_labels, transpose(train_data_transformed),  solver_type=LIBLINEAR.L2R_L2LOSS_SVC) # C = 1e10, verbose=false);

            test_data_transformed = prepare_data(test_data, l, scale_radii, width)

            pred_rcdt = linear_predict(clf_rcdt_model, transpose(test_data_transformed))[1];
            
            acc_rcdt_k[i] = mean(test_labels .== pred_rcdt)
        end 

        append!(acc_rcdt, mean(acc_rcdt_k))
        append!(acc_rcdt_std, std(acc_rcdt_k))
        # println("Acc. of RCDT with \t", i, "\t inst.(s) : \t", acc_rcdt[ik])
        # ik = ik+1
        # println("-----------------------------------------------------------------------------")
    end
    
    println("Acc. of RCDT with \t", π/num_angles_rcdt_norm*argmax(acc_rcdt), "\t : \t", mean(acc_rcdt), "+/-", acc_rcdt_std[argmax(acc_rcdt)])
    println("-----------------------------------------------------------------------------")
    ==#

    acc_rcdt_k = zeros(10)
    for i in 1:10
        split_range = Array((i-1)*split+1:i*split)
        for k in 2:size(samp)[1]
            append!(split_range,  Array((k-1)*size_data_small+(i-1)*split+1:(k-1)*size_data_small+i*split))
        end
        train_data = dataset[split_range,:,:]
        train_labels = labels[split_range]

        test_range = Array(1:size_data)
        test_range = filter(e->!(e in split_range),test_range)

        test_data = dataset[test_range,:,:]
        test_labels = labels[test_range]

        train_data_transformed = prepare_data(train_data, num_angles_rcdt_norm, scale_radii, width)

        clf_rcdt_model = linear_train(train_labels, transpose(train_data_transformed),  solver_type=LIBLINEAR.L2R_L2LOSS_SVC) # C = 1e10, verbose=false);

        test_data_transformed = prepare_data(test_data, num_angles_rcdt_norm, scale_radii, width)

        pred_rcdt = linear_predict(clf_rcdt_model, transpose(test_data_transformed))[1];
        
        acc_rcdt_k[i] = mean(test_labels .== pred_rcdt)
    end 
    
    println("Acc. of RCDT : \t", mean(acc_rcdt_k), "+/-", std(acc_rcdt_k))
    println("-----------------------------------------------------------------------------")


    #####
    #####
    #
    # NRCDT --- Computation --- with normalization
    #
    #
    #####
    #####

    acc_rcdt_normalized_max = zeros(10)
    acc_rcdt_normalized_mean = zeros(10)
    for l in 1:10
        split_range = Array((l-1)*split+1:l*split)
        for k in 2:size(samp)[1]
            append!(split_range,  Array((k-1)*size_data_small+(l-1)*split+1:(k-1)*size_data_small+l*split))
        end
        train_data = dataset[split_range,:,:]
        train_labels = labels[split_range]

        test_range = Array(1:size_data)
        test_range = filter(e->!(e in split_range),test_range)

        test_data = dataset[test_range,:,:]
        test_labels = labels[test_range]

        ref = ones(size(train_data)[2], size(train_data)[3])
        
        train_rcdt1 = transpose(rcdt(ref, train_data[1,:,:], num_angles_rcdt_norm, scale_radii, width)[1])
        train_rcdt = zeros(size(train_data)[1], size(train_rcdt1)[1], size(train_rcdt1)[2])
        train_rcdt[1,:,:] = train_rcdt1
        for i in 2:size(train_data)[1]
            train_rcdt[i,:,:] = transpose(rcdt(ref, train_data[i,:,:], num_angles_rcdt_norm, scale_radii, width)[1])
        end
        train_rcdt_normalized = (train_rcdt .- mean(train_rcdt, dims=2))./sqrt.(var(train_rcdt, dims=2))
        train_rcdt_normalized_max = dropdims(maximum(train_rcdt_normalized, dims=3), dims=3)
        train_rcdt_normalized_mean = dropdims(mean(train_rcdt_normalized, dims=3), dims=3)

        clf_rcdt_normalized_model_max = linear_train(train_labels, transpose(train_rcdt_normalized_max),  solver_type=LIBLINEAR.L2R_L2LOSS_SVC) # C = 1e10, verbose=false);
        clf_rcdt_normalized_model_mean = linear_train(train_labels, transpose(train_rcdt_normalized_mean),  solver_type=LIBLINEAR.L2R_L2LOSS_SVC) #C = 1e10, verbose=false);


        test_rcdt1 = transpose(rcdt(ref, test_data[1,:,:], num_angles_rcdt_norm, scale_radii, width)[1])
        test_rcdt = zeros(size(test_data)[1], size(test_rcdt1)[1], size(test_rcdt1)[2])
        test_rcdt[1,:,:] = test_rcdt1
        for i in 2:size(test_data)[1]
            test_rcdt[i,:,:] = transpose(rcdt(ref, test_data[i,:,:], num_angles_rcdt_norm, scale_radii, width)[1])
        end
        test_rcdt_normalized = (test_rcdt .- mean(test_rcdt, dims=2))./sqrt.(var(test_rcdt, dims=2))
        test_rcdt_normalized_max = dropdims(maximum(test_rcdt_normalized, dims=3), dims=3)
        test_rcdt_normalized_mean = dropdims(mean(test_rcdt_normalized, dims=3), dims=3)


        pred_rcdt_normalized_max = linear_predict(clf_rcdt_normalized_model_max, transpose(test_rcdt_normalized_max))[1];
        pred_rcdt_normalized_mean = linear_predict(clf_rcdt_normalized_model_mean, transpose(test_rcdt_normalized_mean))[1];
        
        acc_rcdt_normalized_max[l] = mean(test_labels .== pred_rcdt_normalized_max)
        acc_rcdt_normalized_mean[l] = mean(test_labels .== pred_rcdt_normalized_mean)
    end

    println("Acc. of max-NRCDT : \t", mean(acc_rcdt_normalized_max), "+/-", std(acc_rcdt_normalized_max))
    println("-----------------------------------------------------------------------------")
    println("Acc. of mean-NRCDT : \t", mean(acc_rcdt_normalized_mean), "+/-", std(acc_rcdt_normalized_mean))
    println("-----------------------------------------------------------------------------")

    
end

function nearest_data_NRCDT(samp, image_size, size_data, random_seed, num_angles_rcdt_norm, scale_radii, noise, width)

    templates = load("temp.jld")["temp"]
    label = range(1, size(templates)[1])
    size_data_small = size_data

    #####
    #####
    # 
    # Generating the chosen tamps form "samp"-image tamps
    # 
    #
    #####
    #####

    temp = zeros(size(samp)[1], image_size, image_size)
    lab = zeros(size(samp)[1])
    k = 1
    for i in samp
        temp[k,:,:] = imresize(templates[i,:,:], image_size, image_size)    # imresize(convert(Array{Float64}, templates[i]), (image_size, image_size))
        lab[k] = label[i]
        k = k+1
    end
    
    temp_ext = zeros(size(temp)[1],2*image_size,2*image_size)

    for i in 1:size(samp)[1]
        image = zeros(2*image_size, 2*image_size)
        a = Int(round((2*image_size - size(temp[i,:,:])[1])/2))
        b = Int(round((2*image_size - size(temp[i,:,:])[2])/2))
        image[a:size(temp[i,:,:])[1]+a-1, b:size(temp[i,:,:])[2]+b-1] = temp[i,:,:]
        temp_ext[i,:,:] = image
    end
    temp = temp_ext

    
    # Plot of temp_ext
    plt1 = plot(layout=(1,size(samp)[1]), plot_title="choice of the synthetic template")
    # plt1 = []
    for i in 1:size(samp)[1]
        plot!(plt1, Gray.(temp[i,:,:]), subplot=i, xaxis=false, yaxis=false, grid=false); # plot each set in a different subplot
        # push!(plt1, heatmap(temp[i,:,:], aspect_ratio=:equal, axis=([], false), cbar=false))
    end
    display(plt1)
    savefig(plt1, "nearest_temp.pdf")

    
    #####
    #####
    #
    # Generating the dataset from the chosen tamps
    #
    #
    #####
    #####

    if noise == 0
        parameters = [(0.75,1.0),(-0.5,0.5),(-5,5),(-5,5),(-5,5)] #noise: (4,20,2,5), (10,40,2,5)
    elseif noise == 0.5
        parameters = [(0.75,1.0),(0.,0.),(0.,0.),(-5,5),(-5,5)] #noise: (4,20,2,5), (10,40,2,5)
    elseif noise == 1
        parameters = [(0.75,1.0),(-0.5,0.5),(-5,5),(-5,5),(-5,5),(4,20,2,5)] #noise: (4,20,2,5), (10,40,2,5)
    elseif noise == 1.5
        parameters = [(0.75,1.0),(-0.5,0.5),(0.,0.),(-5,5),(-5,5)] #noise: (4,20,2,5), (10,40,2,5)
    end


    dataset, labels =  gen_dataset(temp, lab, 2*image_size, size_data_small, parameters, random_seed)


    size_data = size(dataset)[1]
    println("Size of data: \t ", size_data)
    sel = rand(1:size_data, 9)

    # Plot random choice of dataimages
    plt2 = plot(layout=(3,3), plot_title="random choice of dataset")
    # plt2 = []
    for i in 1:9
        plot!(plt2, Gray.(dataset[sel[i],:,:]), subplot=i, xaxis=false, yaxis=false, grid=false); # plot each set in a different subplot
        # push!(plt2, heatmap(dataset[sel[i],:,:], aspect_ratio=:equal, axis=([], false), cbar=false))
    end
    display(plt2)
    plot(plt2)

    savefig(plt2, "nearest_data.pdf")

    plt3 = plot(layout=(4,5), plot_title="random choice of dataset")
    # plt2 = []
    for i in 1:20
        plot!(plt3, Gray.(dataset[i,:,:]), subplot=i, xaxis=false, yaxis=false, grid=false); # plot each set in a different subplot
        # push!(plt2, heatmap(dataset[sel[i],:,:], aspect_ratio=:equal, axis=([], false), cbar=false))
    end
    display(plt3)
    plot(plt3)

    savefig(plt3, "nearest_data_all.pdf")
    
    #####
    #####
    #
    # Generating training data
    #
    #
    #####
    #####   

    println("=============================================================================")

    #####
    #####
    #
    # NRCDT --- Computation --- with normalization
    #
    #
    #####
    #####
    
    train_data = temp
    train_labels = lab

    test_data = dataset
    test_labels = labels

    ref = ones(size(train_data)[2], size(train_data)[3])
    
    train_rcdt1 = transpose(rcdt(ref, train_data[1,:,:], num_angles_rcdt_norm, scale_radii, width)[1])
    train_rcdt = zeros(size(train_data)[1], size(train_rcdt1)[1], size(train_rcdt1)[2])
    train_rcdt[1,:,:] = train_rcdt1
    for i in 2:size(train_data)[1]
        train_rcdt[i,:,:] = transpose(rcdt(ref, train_data[i,:,:], num_angles_rcdt_norm, scale_radii, width)[1])
    end
    train_rcdt_normalized = (train_rcdt .- mean(train_rcdt, dims=2))./sqrt.(var(train_rcdt, dims=2))
    train_rcdt_normalized_max = dropdims(maximum(train_rcdt_normalized, dims=3), dims=3)
    train_rcdt_normalized_mean = dropdims(mean(train_rcdt_normalized, dims=3), dims=3)

    test_rcdt1 = transpose(rcdt(ref, test_data[1,:,:], num_angles_rcdt_norm, scale_radii, width)[1])
    test_rcdt = zeros(size(test_data)[1], size(test_rcdt1)[1], size(test_rcdt1)[2])
    test_rcdt[1,:,:] = test_rcdt1
    for i in 2:size(test_data)[1]
        test_rcdt[i,:,:] = transpose(rcdt(ref, test_data[i,:,:], num_angles_rcdt_norm, scale_radii, width)[1])
    end
    test_rcdt_normalized = (test_rcdt .- mean(test_rcdt, dims=2))./sqrt.(var(test_rcdt, dims=2))
    test_rcdt_normalized_max = dropdims(maximum(test_rcdt_normalized, dims=3), dims=3)
    test_rcdt_normalized_mean = dropdims(mean(test_rcdt_normalized, dims=3), dims=3)

    dd = size(train_rcdt_normalized_max)[2]
    plt3 = plot(layout=(size(samp)[1]+1, 2), plot_title="transform of max. NR-CDT", size = (1000,1400))
    for i in 1:size(samp)[1]
        for j in 1:size(test_data)[1]
            if test_labels[j] == train_labels[i]
                # plot each set in a different subplot
                plot!(plt3, test_rcdt_normalized_max[j,:], subplot=(i-1)*2+1, label=false, linecolor = RGBA(1-i/size(samp)[1]*0.85, 0, i/size(samp)[1]*0.85, 0.5), yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
                plot!(plt3, test_rcdt_normalized_mean[j,:], subplot=(i-1)*2+2, label=false, linecolor = RGBA(1-i/size(samp)[1]*0.85, 0, i/size(samp)[1]*0.85, 0.5), yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));

                plot!(plt3, test_rcdt_normalized_max[j,:], subplot=size(samp)[1]*2+1, label=false, linecolor = RGBA(1-i/size(samp)[1]*0.85, 0, i/size(samp)[1]*0.85, 0.5), yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
                plot!(plt3, test_rcdt_normalized_mean[j,:], subplot=size(samp)[1]*2+2, label=false, linecolor = RGBA(1-i/size(samp)[1]*0.85, 0, i/size(samp)[1]*0.85, 0.5), yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
            end
        end
    end
    for i in 1:size(samp)[1]
        plot!(plt3, train_rcdt_normalized_max[i,:], subplot=(i-1)*2+1, label=false, color=:black, yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
        plot!(plt3, train_rcdt_normalized_mean[i,:], subplot=(i-1)*2+2, label=false, color=:black, yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));

        plot!(plt3, train_rcdt_normalized_max[i,:], subplot=size(samp)[1]*2+1, fontfamily="Computer Modern", label=["class $i" "i"], linewidth=2 , linecolor = RGBA(1-i/size(samp)[1]*0.85, 0, i/size(samp)[1]*0.85, 1), yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
        plot!(plt3, train_rcdt_normalized_mean[i,:], subplot=size(samp)[1]*2+2, fontfamily="Computer Modern", label=["class $i" "i"], linewidth=2 , linecolor = RGBA(1-i/size(samp)[1]*0.85, 0, i/size(samp)[1]*0.85, 1), yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
    end
    display(plt3);
    savefig(plt3, "nearest_m_rc-det.pdf")


    pred_rcdt_normalized_max = zeros(size(train_data)[1],size(test_data)[1])
    pred_rcdt_normalized_max_2 = zeros(size(train_data)[1],size(test_data)[1])
    pred_rcdt_normalized_mean = zeros(size(train_data)[1],size(test_data)[1])

    for k in 1:size(test_data)[1]
        for kk in 1:size(train_data)[1]
            pred_rcdt_normalized_max[kk,k] = maximum(abs.(train_rcdt_normalized_max[kk,:] .- test_rcdt_normalized_max[k,:]))
            pred_rcdt_normalized_max_2[kk,k] = sqrt(sum((train_rcdt_normalized_max[kk,:] .- test_rcdt_normalized_max[k,:]).*(train_rcdt_normalized_max[kk,:] .- test_rcdt_normalized_max[k,:])))
            pred_rcdt_normalized_mean[kk,k] = sqrt(sum((train_rcdt_normalized_mean[kk,:] .- test_rcdt_normalized_mean[k,:]).*(train_rcdt_normalized_mean[kk,:] .- test_rcdt_normalized_mean[k,:])))
        end
    end
    
    pred_label_rcdt_normalized_max = argmin(pred_rcdt_normalized_max, dims=1)
    pred_label_rcdt_normalized_max_2 = argmin(pred_rcdt_normalized_max_2, dims=1)
    pred_label_rcdt_normalized_mean = argmin(pred_rcdt_normalized_mean, dims=1)

    label_rcdt_normalized_max = zeros(size(test_data)[1])
    label_rcdt_normalized_max_2 = zeros(size(test_data)[1])
    label_rcdt_normalized_mean = zeros(size(test_data)[1])

    for k in 1:size(test_data)[1]
        label_rcdt_normalized_max[k] = train_labels[pred_label_rcdt_normalized_max[k][1]]
        label_rcdt_normalized_max_2[k] = train_labels[pred_label_rcdt_normalized_max_2[k][1]]
        label_rcdt_normalized_mean[k] = train_labels[pred_label_rcdt_normalized_mean[k][1]]
    end 

    println("Acc. of max-NRCDT (||.||_inf) : \t", mean(test_labels .== label_rcdt_normalized_max))
    println("Acc. of max-NRCDT (||.||_2) : \t", mean(test_labels .== label_rcdt_normalized_max_2))
    println("-----------------------------------------------------------------------------")
    println("Acc. of mean-NRCDT : \t", mean(test_labels .== label_rcdt_normalized_mean))
    println("-----------------------------------------------------------------------------")

end

function nearest_cross_data_NRCDT(samp, image_size, size_data, random_seed, num_angles_rcdt_norm, scale_radii, noise, width)

    templates = load("temp.jld")["temp"]
    label = range(1, size(templates)[1])
    size_data_small = size_data

    #####
    #####
    # 
    # Generating the chosen tamps form "samp"-image tamps
    # 
    #
    #####
    #####

    temp = zeros(size(samp)[1], image_size, image_size)
    lab = zeros(size(samp)[1])
    k = 1
    for i in samp
        temp[k,:,:] = imresize(templates[i,:,:], image_size, image_size)    # imresize(convert(Array{Float64}, templates[i]), (image_size, image_size))
        lab[k] = label[i]
        k = k+1
    end
    
    temp_ext = zeros(size(temp)[1],2*image_size,2*image_size)

    for i in 1:size(samp)[1]
        image = zeros(2*image_size, 2*image_size)
        a = Int(round((2*image_size - size(temp[i,:,:])[1])/2))
        b = Int(round((2*image_size - size(temp[i,:,:])[2])/2))
        image[a:size(temp[i,:,:])[1]+a-1, b:size(temp[i,:,:])[2]+b-1] = temp[i,:,:]
        temp_ext[i,:,:] = image
    end
    temp = temp_ext

    
    # Plot of temp_ext
    plt1 = plot(layout=(1,size(samp)[1]), plot_title="choice of the synthetic template")
    # plt1 = []
    for i in 1:size(samp)[1]
        plot!(plt1, Gray.(temp[i,:,:]), subplot=i, xaxis=false, yaxis=false, grid=false); # plot each set in a different subplot
        # push!(plt1, heatmap(temp[i,:,:], aspect_ratio=:equal, axis=([], false), cbar=false))
    end
    display(plt1)

    savefig(plt1, "plt1.pdf")

    
    #####
    #####
    #
    # Generating the dataset from the chosen tamps
    #
    #
    #####
    #####

    if noise == 0
        parameters = [(0.75,1.0),(-0.5,0.5),(-5,5),(-5,5),(-5,5)] #noise: (4,20,2,5), (10,40,2,5)
    elseif noise == 0.5
        parameters = [(0.75,1.0),(0.,0.),(0.,0.),(-5,5),(-5,5)] #noise: (4,20,2,5), (10,40,2,5)
    elseif noise == 1
        parameters = [(0.75,1.0),(-0.5,0.5),(-5,5),(-5,5),(-5,5),(4,20,2,5)] #noise: (4,20,2,5), (10,40,2,5)
    elseif noise == 1.5
        parameters = [(0.75,1.0),(-0.5,0.5),(0.,0.),(-5,5),(-5,5)] #noise: (4,20,2,5), (10,40,2,5)
    end


    dataset, labels =  gen_dataset(temp, lab, 2*image_size, size_data_small, parameters, random_seed)


    size_data = size(dataset)[1]
    println("Size of data: \t ", size_data)
    sel = rand(1:size_data, 9)

    # Plot random choice of dataimages
    plt2 = plot(layout=(3,3), plot_title="random choice of dataset")
    # plt2 = []
    for i in 1:9
        plot!(plt2, Gray.(dataset[sel[i],:,:]), subplot=i, xaxis=false, yaxis=false, grid=false); # plot each set in a different subplot
        # push!(plt2, heatmap(dataset[sel[i],:,:], aspect_ratio=:equal, axis=([], false), cbar=false))
    end
    display(plt2)
    plot(plt2)

    savefig(plt2, "plt2.pdf")
    
    #####
    #####
    #
    # Generating training data
    #
    #
    #####
    #####   

    println("=============================================================================")

    #####
    #####
    #
    # NRCDT --- Computation --- with normalization
    #
    #
    #####
    #####

    acc_rcdt_normalized_max = zeros(size_data_small)
    acc_rcdt_normalized_mean = zeros(size_data_small)
    
    for l in 1:size_data_small
        split_range = Array([l])
        for k in 2:size(samp)[1]
            append!(split_range,  Array([l+(k-1)*size_data_small]))
        end
        train_data = dataset[split_range,:,:]
        train_labels = labels[split_range]

        test_range = Array(1:size_data)
        test_range = filter(e->!(e in split_range),test_range)

        test_data = dataset[test_range,:,:]
        test_labels = labels[test_range]

        ref = ones(size(train_data)[2], size(train_data)[3])
        
        train_rcdt1 = transpose(rcdt(ref, train_data[1,:,:], num_angles_rcdt_norm, scale_radii, width)[1])
        train_rcdt = zeros(size(train_data)[1], size(train_rcdt1)[1], size(train_rcdt1)[2])
        train_rcdt[1,:,:] = train_rcdt1
        for i in 2:size(train_data)[1]
            train_rcdt[i,:,:] = transpose(rcdt(ref, train_data[i,:,:], num_angles_rcdt_norm, scale_radii, width)[1])
        end
        train_rcdt_normalized = (train_rcdt .- mean(train_rcdt, dims=2))./sqrt.(var(train_rcdt, dims=2))
        train_rcdt_normalized_max = dropdims(maximum(train_rcdt_normalized, dims=3), dims=3)
        train_rcdt_normalized_mean = dropdims(mean(train_rcdt_normalized, dims=3), dims=3)

        # clf_rcdt_normalized_model_max = linear_train(train_labels, transpose(train_rcdt_normalized_max),  solver_type=LIBLINEAR.L2R_L2LOSS_SVC) # C = 1e10, verbose=false);
        # clf_rcdt_normalized_model_mean = linear_train(train_labels, transpose(train_rcdt_normalized_mean),  solver_type=LIBLINEAR.L2R_L2LOSS_SVC) #C = 1e10, verbose=false);


        test_rcdt1 = transpose(rcdt(ref, test_data[1,:,:], num_angles_rcdt_norm, scale_radii, width)[1])
        test_rcdt = zeros(size(test_data)[1], size(test_rcdt1)[1], size(test_rcdt1)[2])
        test_rcdt[1,:,:] = test_rcdt1
        for i in 2:size(test_data)[1]
            test_rcdt[i,:,:] = transpose(rcdt(ref, test_data[i,:,:], num_angles_rcdt_norm, scale_radii, width)[1])
        end
        test_rcdt_normalized = (test_rcdt .- mean(test_rcdt, dims=2))./sqrt.(var(test_rcdt, dims=2))
        test_rcdt_normalized_max = dropdims(maximum(test_rcdt_normalized, dims=3), dims=3)
        test_rcdt_normalized_mean = dropdims(mean(test_rcdt_normalized, dims=3), dims=3)


        # pred_rcdt_normalized_max = linear_predict(clf_rcdt_normalized_model_max, transpose(test_rcdt_normalized_max))[1];
        # pred_rcdt_normalized_mean = linear_predict(clf_rcdt_normalized_model_mean, transpose(test_rcdt_normalized_mean))[1];

        pred_rcdt_normalized_max = zeros(size(train_data)[1],size(test_data)[1])
        pred_rcdt_normalized_mean = zeros(size(train_data)[1],size(test_data)[1])

        for k in 1:size(test_data)[1]
            for kk in 1:size(train_data)[1]
                pred_rcdt_normalized_max[kk,k] = maximum(abs.(train_rcdt_normalized_max[kk,:] .- test_rcdt_normalized_max[k,:]))
                pred_rcdt_normalized_mean[kk,k] = sqrt(sum((train_rcdt_normalized_mean[kk,:] .- test_rcdt_normalized_mean[k,:]).*(train_rcdt_normalized_mean[kk,:] .- test_rcdt_normalized_mean[k,:])))
            end
        end
        
        pred_label_rcdt_normalized_max = argmin(pred_rcdt_normalized_max, dims=1)
        pred_label_rcdt_normalized_mean = argmin(pred_rcdt_normalized_mean, dims=1)

        label_rcdt_normalized_max = zeros(size(test_data)[1])
        label_rcdt_normalized_mean = zeros(size(test_data)[1])

        for k in 1:size(test_data)[1]
            label_rcdt_normalized_max[k] = train_labels[pred_label_rcdt_normalized_max[k][1]]
            label_rcdt_normalized_mean[k] = train_labels[pred_label_rcdt_normalized_mean[k][1]]
        end 

        acc_rcdt_normalized_max[l] = mean(test_labels .== label_rcdt_normalized_max)
        acc_rcdt_normalized_mean[l] = mean(test_labels .== label_rcdt_normalized_mean)
    end

    println("Acc. of max-NRCDT : \t", mean(acc_rcdt_normalized_max), "+/-", std(acc_rcdt_normalized_max))
    println("-----------------------------------------------------------------------------")
    println("Acc. of mean-NRCDT : \t", mean(acc_rcdt_normalized_mean), "+/-", std(acc_rcdt_normalized_mean))
    println("-----------------------------------------------------------------------------")

end


function samp_mnist(samp, size_data)
    trainset = MNIST(:train)
    target = trainset.targets;
    pos = findall(x->x==samp[1], target)
    pos = pos[1:size_data]
    for k in 2:size(samp)[1]
        pos1 = findall(x->x==samp[k], target)
        append!(pos, pos1[1:size_data])
    end
    # shuffle!(pos);
    data_mnist = trainset[pos].features
    label_mnist = trainset[pos].targets
    return data_mnist, label_mnist
end


function classify_mnist_NRCDT(samp, size_data, random_seed, num_angles_rcdt, num_angles_rcdt_norm, scale_radii, noise, width)

    datas, labels = NormalizedRadonCDT.samp_mnist(samp, size_data);
    size_data_small = size_data

    dataset = permutedims(datas, (3, 2, 1))

    size_data = size(labels)[1]

    # size_data = min(size_data, size(label)[1])
    # dataset = datass[1:size_data,:,:]
    # labels = label[1:size_data]

    println("Choise of labels: \t " , unique(labels))
    println("Size of data: \t " , size(labels)[1])

    image_size = size(dataset)[2]

    datas_ext = zeros(size_data,2*image_size,2*image_size)

    for i in 1:size_data
        image = zeros(2*image_size, 2*image_size)
        a = Int(round((2*image_size - size(dataset[i,:,:])[1])/2))
        b = Int(round((2*image_size - size(dataset[i,:,:])[2])/2))
        image[a:size(dataset[i,:,:])[1]+a-1, b:size(dataset[i,:,:])[2]+b-1] = dataset[i,:,:]
        datas_ext[i,:,:] = image
    end
    dataset = datas_ext


    sel = rand(1:size_data, 9)
    # Plot random choice of dataimages
    plt1 = plot(layout=(3,3), plot_title="random choice of dataset")
    # plt2 = []
    for i in 1:9
        plot!(plt1, Gray.(dataset[sel[i],:,:]), subplot=i, xaxis=false, yaxis=false, grid=false); # plot each set in a different subplot
        # push!(plt2, heatmap(dataset[sel[i],:,:], aspect_ratio=:equal, axis=([], false), cbar=false))
    end
    display(plt1)
    plot(plt1)

    savefig(plt1, "plt_mnist_rand_data.pdf")

    #####
    #####
    #
    # noisy - MNIST data
    #
    #
    #####
    #####

    if noise == 0
        parameters = [(0.75,1.25),(-1.0,1.0),(-5,5),(-15,15),(-15,15)] #noise: (4,20,2,5), (10,40,2,5)
    elseif noise == 0.5
        parameters = [(0.75,1.0),(0.,0.),(0.,0.),(-5,5),(-5,5)] #noise: (4,20,2,5), (10,40,2,5)
    elseif noise == 1
        parameters = [(0.75,1.0),(-0.5,0.5),(-5,5),(-5,5),(-5,5),(4,20,2,5)] #noise: (4,20,2,5), (10,40,2,5)
    elseif noise == 1.5
        parameters = [(0.75,1.0),(-0.5,0.5),(0.,0.),(-5,5),(-5,5)] #noise: (4,20,2,5), (10,40,2,5)
    end

    datas_noise = zeros(size_data,2*image_size,2*image_size)

    for i in 1:size_data
        datas_noise[i,:,:] = gen_dataset_mnist(dataset[i,:,:], 2*image_size, parameters, mod(i,10))
    end
    dataset = datas_noise

    
    for i in 1:size_data
        # dataset[i,:,:] = dataset[i,:,:]/sum(dataset[i,:,:])
        dataset[i,:,:] = dataset[i,:,:]/maximum(dataset[i,:,:])
    end

    sel = rand(1:size_data, 9)
    # Plot random choice of dataimages
    plt11 = plot(layout=(3,3), plot_title="random choice of noisy dataset")
    # plt2 = []
    for i in 1:9
        plot!(plt11, Gray.(dataset[sel[i],:,:]), subplot=i, xaxis=false, yaxis=false, grid=false); # plot each set in a different subplot
        # push!(plt2, heatmap(dataset[sel[i],:,:], aspect_ratio=:equal, axis=([], false), cbar=false))
    end
    display(plt11)
    plot(plt11)

    savefig(plt11, "plt_mnist_rand_data_noise.pdf")

    #####
    #####
    #
    # RCDT --- Computation --- without normalization
    #
    #
    #####
    #####

    ref = ones(size(dataset[1,:,:])[1], size(dataset[1,:,:])[2])
    
    data_rcdt1 = transpose(rcdt(ref, dataset[1,:,:], num_angles_rcdt, scale_radii, width)[1])
    data_rcdt = zeros(size(dataset)[1], size(data_rcdt1)[1], size(data_rcdt1)[2])
    data_rcdt[1,:,:] = data_rcdt1
    for i in 2:size(dataset)[1]
        data_rcdt[i,:,:] = transpose(rcdt(ref, dataset[i,:,:], num_angles_rcdt, scale_radii, width)[1])
    end

    data_rcdt_normalized = (data_rcdt .- mean(data_rcdt, dims=2))./sqrt.(var(data_rcdt, dims=2))
    data_rcdt_normalized = dropdims(maximum(data_rcdt_normalized, dims=3), dims=3)

    
    # Plot of results
    plt2 = plot(layout=(size(samp)[1], 2), plot_title="transform of unnormalized and normalized")
    for i in 1:size(samp)[1]
        for j in 1:size(dataset)[1]
            if labels[j] == samp[i]
                # plot each set in a different subplot
                plot!(plt2, data_rcdt[j,:,1], subplot=(i-1)*2+1, legend=false);
                plot!(plt2, data_rcdt_normalized[j,:], subplot=(i-1)*2+2, legend=false);
            end
        end
    end
    display(plt2);
    savefig(plt2, "plt_mnist_rcdt.pdf")
    
    #####
    #####
    #
    # Generating training data
    #
    #
    #####
    #####

    split = Int(size_data_small/10)

    #####
    #####
    #
    # Euclidean --- Computation
    #
    #
    #####
    #####

    acc_euclid = zeros(10)
    for i in 1:10
        split_range = Array((i-1)*split+1:i*split)
        for k in 2:size(samp)[1]
            append!(split_range,  Array((k-1)*size_data_small+(i-1)*split+1:(k-1)*size_data_small+i*split))
        end
        train_data = dataset[split_range,:,:]
        train_labels = labels[split_range]

        test_range = Array(1:size_data)
        test_range = filter(e->!(e in split_range),test_range)

        test_data = dataset[test_range,:,:]
        test_labels = labels[test_range]

        train_data_reshaped = reshape(train_data, (size(train_data)[1], size(train_data)[2] * size(train_data)[3]));

        clf_euclid_model = linear_train(train_labels, transpose(train_data_reshaped), solver_type=LIBLINEAR.L2R_L2LOSS_SVC) #, verbose=false);

        test_data_reshaped = reshape(test_data, (size(test_data)[1], size(test_data)[2] * size(test_data)[3]));

        pred_euclid = linear_predict(clf_euclid_model, transpose(test_data_reshaped))[1];

        acc_euclid[i] = mean(test_labels .== pred_euclid)
    end     


    println("=============================================================================")
    println("Acc. of Euclidean : \t", mean(acc_euclid), "+/-", std(acc_euclid))
    println("-----------------------------------------------------------------------------")


    #####
    #####
    #
    # RCDT --- Computation --- without normalization
    #
    #
    #####
    #####

    acc_rcdt_k = zeros(10)
    for i in 1:10
        split_range = Array((i-1)*split+1:i*split)
        for k in 2:size(samp)[1]
            append!(split_range,  Array((k-1)*size_data_small+(i-1)*split+1:(k-1)*size_data_small+i*split))
        end
        train_data = dataset[split_range,:,:]
        train_labels = labels[split_range]

        test_range = Array(1:size_data)
        test_range = filter(e->!(e in split_range),test_range)

        test_data = dataset[test_range,:,:]
        test_labels = labels[test_range]

        train_data_transformed = prepare_data(train_data, num_angles_rcdt_norm, scale_radii, width)

        clf_rcdt_model = linear_train(train_labels, transpose(train_data_transformed),  solver_type=LIBLINEAR.L2R_L2LOSS_SVC) # C = 1e10, verbose=false);

        test_data_transformed = prepare_data(test_data, num_angles_rcdt_norm, scale_radii, width)

        pred_rcdt = linear_predict(clf_rcdt_model, transpose(test_data_transformed))[1];
        
        acc_rcdt_k[i] = mean(test_labels .== pred_rcdt)
    end 
    
    println("Acc. of RCDT with : \t", maximum(acc_rcdt_k), "+/-", std(acc_rcdt_k))
    println("-----------------------------------------------------------------------------")

    #####
    #####
    #
    # NRCDT --- Computation --- with normalization
    #
    #
    #####
    #####

    acc_rcdt_normalized_max = zeros(10)
    acc_rcdt_normalized_mean = zeros(10)
    for l in 1:10
        split_range = Array((l-1)*split+1:l*split)
        for k in 2:size(samp)[1]
            append!(split_range,  Array((k-1)*size_data_small+(l-1)*split+1:(k-1)*size_data_small+l*split))
        end
        train_data = dataset[split_range,:,:]
        train_labels = labels[split_range]

        test_range = Array(1:size_data)
        test_range = filter(e->!(e in split_range),test_range)

        test_data = dataset[test_range,:,:]
        test_labels = labels[test_range]

        ref = ones(size(train_data)[2], size(train_data)[3])
        
        train_rcdt1 = transpose(rcdt(ref, train_data[1,:,:], num_angles_rcdt_norm, scale_radii, width)[1])
        train_rcdt = zeros(size(train_data)[1], size(train_rcdt1)[1], size(train_rcdt1)[2])
        train_rcdt[1,:,:] = train_rcdt1
        for i in 2:size(train_data)[1]
            train_rcdt[i,:,:] = transpose(rcdt(ref, train_data[i,:,:], num_angles_rcdt_norm, scale_radii, width)[1])
        end
        train_rcdt_normalized = (train_rcdt .- mean(train_rcdt, dims=2))./sqrt.(var(train_rcdt, dims=2))
        train_rcdt_normalized_max = dropdims(maximum(train_rcdt_normalized, dims=3), dims=3)
        train_rcdt_normalized_mean = dropdims(mean(train_rcdt_normalized, dims=3), dims=3)

        clf_rcdt_normalized_model_max = linear_train(train_labels, transpose(train_rcdt_normalized_max),  solver_type=LIBLINEAR.L2R_L2LOSS_SVC) # C = 1e10, verbose=false);
        clf_rcdt_normalized_model_mean = linear_train(train_labels, transpose(train_rcdt_normalized_mean),  solver_type=LIBLINEAR.L2R_L2LOSS_SVC) #C = 1e10, verbose=false);


        test_rcdt1 = transpose(rcdt(ref, test_data[1,:,:], num_angles_rcdt_norm, scale_radii, width)[1])
        test_rcdt = zeros(size(test_data)[1], size(test_rcdt1)[1], size(test_rcdt1)[2])
        test_rcdt[1,:,:] = test_rcdt1
        for i in 2:size(test_data)[1]
            test_rcdt[i,:,:] = transpose(rcdt(ref, test_data[i,:,:], num_angles_rcdt_norm, scale_radii, width)[1])
        end
        test_rcdt_normalized = (test_rcdt .- mean(test_rcdt, dims=2))./sqrt.(var(test_rcdt, dims=2))
        test_rcdt_normalized_max = dropdims(maximum(test_rcdt_normalized, dims=3), dims=3)
        test_rcdt_normalized_mean = dropdims(mean(test_rcdt_normalized, dims=3), dims=3)


        pred_rcdt_normalized_max = linear_predict(clf_rcdt_normalized_model_max, transpose(test_rcdt_normalized_max))[1];
        pred_rcdt_normalized_mean = linear_predict(clf_rcdt_normalized_model_mean, transpose(test_rcdt_normalized_mean))[1];
        
        acc_rcdt_normalized_max[l] = mean(test_labels .== pred_rcdt_normalized_max)
        acc_rcdt_normalized_mean[l] = mean(test_labels .== pred_rcdt_normalized_mean)
    end

    println("Acc. of max-NRCDT : \t", mean(acc_rcdt_normalized_max), "+/-", std(acc_rcdt_normalized_max))
    println("-----------------------------------------------------------------------------")
    println("Acc. of mean-NRCDT : \t", mean(acc_rcdt_normalized_mean), "+/-", std(acc_rcdt_normalized_mean))
    println("-----------------------------------------------------------------------------")
    
end

function nearest_cross_mnist_NRCDT(samp, size_data, random_seed, num_angles_rcdt_norm, scale_radii, noise, width)

    datas, labels = NormalizedRadonCDT.samp_mnist(samp, size_data);
    size_data_small = size_data

    dataset = permutedims(datas, (3, 2, 1))

    size_data = size(labels)[1]

    # size_data = min(size_data, size(label)[1])
    # dataset = datass[1:size_data,:,:]
    # labels = label[1:size_data]

    println("Choise of labels: \t " , unique(labels))
    println("Size of data: \t " , size(labels)[1])

    image_size = size(dataset)[2]

    datas_ext = zeros(size_data,2*image_size,2*image_size)

    for i in 1:size_data
        image = zeros(2*image_size, 2*image_size)
        a = Int(round((2*image_size - size(dataset[i,:,:])[1])/2))
        b = Int(round((2*image_size - size(dataset[i,:,:])[2])/2))
        image[a:size(dataset[i,:,:])[1]+a-1, b:size(dataset[i,:,:])[2]+b-1] = dataset[i,:,:]
        datas_ext[i,:,:] = image
    end
    dataset = datas_ext


    sel = rand(1:size_data, 9)
    # Plot random choice of dataimages
    plt1 = plot(layout=(3,3), plot_title="random choice of dataset")
    # plt2 = []
    for i in 1:9
        plot!(plt1, Gray.(dataset[sel[i],:,:]), subplot=i, xaxis=false, yaxis=false, grid=false); # plot each set in a different subplot
        # push!(plt2, heatmap(dataset[sel[i],:,:], aspect_ratio=:equal, axis=([], false), cbar=false))
    end
    display(plt1)
    plot(plt1)

    savefig(plt1, "plt_mnist_rand_data.pdf")

    #####
    #####
    #
    # noisy - MNIST data
    #
    #
    #####
    #####

    if noise == 0
        parameters = [(0.75,1.0),(-0.5,0.5),(-5,5),(-5,5),(-5,5)] #noise: (4,20,2,5), (10,40,2,5)
    elseif noise == 0.5
        parameters = [(0.75,1.0),(0.,0.),(0.,0.),(-5,5),(-5,5)] #noise: (4,20,2,5), (10,40,2,5)
    elseif noise == 1
        parameters = [(0.75,1.0),(-0.5,0.5),(-5,5),(-5,5),(-5,5),(4,20,2,5)] #noise: (4,20,2,5), (10,40,2,5)
    elseif noise == 1.5
        parameters = [(0.75,1.0),(-0.5,0.5),(0.,0.),(-5,5),(-5,5)] #noise: (4,20,2,5), (10,40,2,5)
    end

    datas_noise = zeros(size_data,2*image_size,2*image_size)

    for i in 1:size_data
        datas_noise[i,:,:] = gen_dataset_mnist(dataset[i,:,:], 2*image_size, parameters, mod(i,10))
    end
    dataset = datas_noise


    for i in 1:size_data
        # dataset[i,:,:] = dataset[i,:,:]/sum(dataset[i,:,:])
        dataset[i,:,:] = dataset[i,:,:]/maximum(dataset[i,:,:])
    end

    sel = rand(1:size_data, 9)
    # Plot random choice of dataimages
    plt11 = plot(layout=(3,3), plot_title="random choice of noisy dataset")
    # plt2 = []
    for i in 1:9
        plot!(plt11, Gray.(dataset[sel[i],:,:]), subplot=i, xaxis=false, yaxis=false, grid=false); # plot each set in a different subplot
        # push!(plt2, heatmap(dataset[sel[i],:,:], aspect_ratio=:equal, axis=([], false), cbar=false))
    end
    display(plt11)
    plot(plt11)

    savefig(plt11, "plt_mnist_rand_data_noise.pdf")
    
    #####
    #####
    #
    # Generating training data
    #
    #
    #####
    #####   

    println("=============================================================================")

    #####
    #####
    #
    # NRCDT --- Computation --- with normalization
    #
    #
    #####
    #####

    acc_rcdt_normalized_max = zeros(size_data_small)
    acc_rcdt_normalized_max_2 = zeros(size_data_small)
    acc_rcdt_normalized_mean = zeros(size_data_small)
    
    for l in 1:size_data_small
        split_range = Array([l])
        for k in 2:size(samp)[1]
            append!(split_range,  Array([l+(k-1)*size_data_small]))
        end
        train_data = dataset[split_range,:,:]
        train_labels = labels[split_range]

        test_range = Array(1:size_data)
        test_range = filter(e->!(e in split_range),test_range)

        test_data = dataset[test_range,:,:]
        test_labels = labels[test_range]

        ref = ones(size(train_data)[2], size(train_data)[3])
        
        train_rcdt1 = transpose(rcdt(ref, train_data[1,:,:], num_angles_rcdt_norm, scale_radii, width)[1])
        train_rcdt = zeros(size(train_data)[1], size(train_rcdt1)[1], size(train_rcdt1)[2])
        train_rcdt[1,:,:] = train_rcdt1
        for i in 2:size(train_data)[1]
            train_rcdt[i,:,:] = transpose(rcdt(ref, train_data[i,:,:], num_angles_rcdt_norm, scale_radii, width)[1])
        end
        train_rcdt_normalized = (train_rcdt .- mean(train_rcdt, dims=2))./sqrt.(var(train_rcdt, dims=2))
        train_rcdt_normalized_max = dropdims(maximum(train_rcdt_normalized, dims=3), dims=3)
        train_rcdt_normalized_mean = dropdims(mean(train_rcdt_normalized, dims=3), dims=3)

        test_rcdt1 = transpose(rcdt(ref, test_data[1,:,:], num_angles_rcdt_norm, scale_radii, width)[1])
        test_rcdt = zeros(size(test_data)[1], size(test_rcdt1)[1], size(test_rcdt1)[2])
        test_rcdt[1,:,:] = test_rcdt1
        for i in 2:size(test_data)[1]
            test_rcdt[i,:,:] = transpose(rcdt(ref, test_data[i,:,:], num_angles_rcdt_norm, scale_radii, width)[1])
        end
        test_rcdt_normalized = (test_rcdt .- mean(test_rcdt, dims=2))./sqrt.(var(test_rcdt, dims=2))
        test_rcdt_normalized_max = dropdims(maximum(test_rcdt_normalized, dims=3), dims=3)
        test_rcdt_normalized_mean = dropdims(mean(test_rcdt_normalized, dims=3), dims=3)


        dd = size(train_rcdt_normalized_max)[2]
        plt3 = plot(layout=(size(samp)[1]+1, 2), plot_title="transform of max. NR-CDT")
        for i in 1:size(samp)[1]
            for j in 1:size(test_data)[1]
                if test_labels[j] == train_labels[i]
                    # plot each set in a different subplot
                    plot!(plt3, test_rcdt_normalized_max[j,:], subplot=(i-1)*2+1, legend=false, linecolor = RGBA(1-i/size(samp)[1]*0.5, i/size(samp)[1]*0.5, 1, 0.5), yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
                    plot!(plt3, test_rcdt_normalized_mean[j,:], subplot=(i-1)*2+2, legend=false, linecolor = RGBA(1-i/size(samp)[1]*0.5, i/size(samp)[1]*0.5, 1, 0.5), yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));

                    plot!(plt3, test_rcdt_normalized_max[j,:], subplot=size(samp)[1]*2+1, legend=false, linecolor = RGBA(1-i/size(samp)[1]*0.5, i/size(samp)[1]*0.5, 1, 0.5), yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
                    plot!(plt3, test_rcdt_normalized_mean[j,:], subplot=size(samp)[1]*2+2, legend=false, linecolor = RGBA(1-i/size(samp)[1]*0.5, i/size(samp)[1]*0.5, 1, 0.5), yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
                end
            end
        end
        for i in 1:size(samp)[1]
            plot!(plt3, train_rcdt_normalized_max[i,:], subplot=(i-1)*2+1, legend=false, color=:black, yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
            plot!(plt3, train_rcdt_normalized_mean[i,:], subplot=(i-1)*2+2, legend=false, color=:black, yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));

            plot!(plt3, train_rcdt_normalized_max[i,:], subplot=size(samp)[1]*2+1, legend=false, color=:black, yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
            plot!(plt3, train_rcdt_normalized_mean[i,:], subplot=size(samp)[1]*2+2, legend=false, color=:black, yticks = false, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
        end
        display(plt3);
        savefig(plt3, "nearest_mnist_m_rc-det.pdf")

        pred_rcdt_normalized_max = zeros(size(train_data)[1],size(test_data)[1])
        pred_rcdt_normalized_max_2 = zeros(size(train_data)[1],size(test_data)[1])
        pred_rcdt_normalized_mean = zeros(size(train_data)[1],size(test_data)[1])

        for k in 1:size(test_data)[1]
            for kk in 1:size(train_data)[1]
                pred_rcdt_normalized_max[kk,k] = maximum(abs.(train_rcdt_normalized_max[kk,:] .- test_rcdt_normalized_max[k,:]))
                pred_rcdt_normalized_max_2[kk,k] = sqrt(sum((train_rcdt_normalized_max[kk,:] .- test_rcdt_normalized_max[k,:]).*(train_rcdt_normalized_max[kk,:] .- test_rcdt_normalized_max[k,:])))
                pred_rcdt_normalized_mean[kk,k] = sqrt(sum((train_rcdt_normalized_mean[kk,:] .- test_rcdt_normalized_mean[k,:]).*(train_rcdt_normalized_mean[kk,:] .- test_rcdt_normalized_mean[k,:])))
            end
        end
        
        pred_label_rcdt_normalized_max = argmin(pred_rcdt_normalized_max, dims=1)
        pred_label_rcdt_normalized_max_2 = argmin(pred_rcdt_normalized_max_2, dims=1)
        pred_label_rcdt_normalized_mean = argmin(pred_rcdt_normalized_mean, dims=1)

        label_rcdt_normalized_max = zeros(size(test_data)[1])
        label_rcdt_normalized_max_2 = zeros(size(test_data)[1])
        label_rcdt_normalized_mean = zeros(size(test_data)[1])

        for k in 1:size(test_data)[1]
            label_rcdt_normalized_max[k] = train_labels[pred_label_rcdt_normalized_max[k][1]]
            label_rcdt_normalized_max_2[k] = train_labels[pred_label_rcdt_normalized_max_2[k][1]]
            label_rcdt_normalized_mean[k] = train_labels[pred_label_rcdt_normalized_mean[k][1]]
        end 

        acc_rcdt_normalized_max[l] = mean(test_labels .== label_rcdt_normalized_max)
        acc_rcdt_normalized_max_2[l] = mean(test_labels .== label_rcdt_normalized_max_2)
        acc_rcdt_normalized_mean[l] = mean(test_labels .== label_rcdt_normalized_mean)
    end

    println("Acc. of max-NRCDT (||.||_inf): \t", mean(acc_rcdt_normalized_max), "+/-", std(acc_rcdt_normalized_max))
    println("Acc. of max-NRCDT (||.||_2): \t", mean(acc_rcdt_normalized_max_2), "+/-", std(acc_rcdt_normalized_max_2))
    println("-----------------------------------------------------------------------------")
    println("Acc. of mean-NRCDT : \t", mean(acc_rcdt_normalized_mean), "+/-", std(acc_rcdt_normalized_mean))
    println("-----------------------------------------------------------------------------")

end
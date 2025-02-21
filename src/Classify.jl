using Plots
# using MLDatasets, Random, Statistics
# using Interpolations: LinearInterpolation as LinInter
using LIBSVM, LIBLINEAR

function mNRCDT_quantiles(temp_q::AbstractArray, temp_lab::AbstractArray, data_q::AbstractArray, data_lab::AbstractArray)
    # plot of all of the projections (Theorem 1)
    dd = length(data_q[1])
    plt = plot(plot_title="Quantiles of mNR-CDT", size = (400,400))
    for i in 1:length(temp_lab), j in 1:length(data_lab)
        if data_lab[j] == temp_lab[i]
            plot!(plt, data_q[j], label=false, linecolor = RGBA(1-i/length(temp_lab)*0.85, 0, i/length(temp_lab)*0.85, 0.5), yticks=true, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
        end
    end
    for i in 1:length(temp_lab)
        plot!(plt, temp_q[i], fontfamily="Computer Modern", label=["class $i" "i"], linewidth=2 , linecolor = RGBA(1-i/length(temp_lab)*0.85, 0, i/length(temp_lab)*0.85, 1), yticks=true, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));    
    end
    savefig(plt, "nearest_m_rc-det.pdf")
    return plt
end

function mNRCDT_nearest_neighbour(temp_q::AbstractArray, temp_lab::AbstractArray, data_q::AbstractArray, data_lab::AbstractArray; ret::Int64=0)
    pred_rcdt_max_inf_normalized = zeros(length(temp_q),length(data_q))
    pred_rcdt_max_2_normalized = zeros(length(temp_q),length(data_q))

    for k in 1:length(data_q), kk in 1:length(temp_q)
        pred_rcdt_max_inf_normalized[kk,k] = maximum(abs.(temp_q[kk] .- data_q[k]))
        pred_rcdt_max_2_normalized[kk,k] = sqrt(sum((temp_q[kk] .- data_q[k]).*(temp_q[kk] .- data_q[k])))
    end
    
    pred_label_rcdt_max_inf_normalized = argmin(pred_rcdt_max_inf_normalized, dims=1)
    pred_label_rcdt_max_2_normalized = argmin(pred_rcdt_max_2_normalized, dims=1)

    label_rcdt_max_inf_normalized = zeros(length(data_q))
    label_rcdt_max_2_normalized = zeros(length(data_q))

    for k in 1:length(data_q)
        label_rcdt_max_inf_normalized[k] = temp_lab[pred_label_rcdt_max_inf_normalized[k][1]]
        label_rcdt_max_2_normalized[k] = temp_lab[pred_label_rcdt_max_2_normalized[k][1]]
    end 

    acc_rcdt_max_inf_normalized = mean(data_lab .== label_rcdt_max_inf_normalized)
    acc_rcdt_max_2_normalized = mean(data_lab .== label_rcdt_max_2_normalized)
    if ret==1
        @info "Acc. of max-NRCDT (||.||_inf): \t $(acc_rcdt_max_inf_normalized)"
        @info "Acc. of max-NRCDT (||.||_2): \t $(acc_rcdt_max_2_normalized)"
    end
    return acc_rcdt_max_inf_normalized, acc_rcdt_max_2_normalized
end

function mnist_mNRCDT_nearest_cross_neighbour(data_q::AbstractArray, data_lab::AbstractArray)

    acc_rcdt_max_inf_normalized = zeros(10)
    acc_rcdt_max_2_normalized = zeros(10)
    size_data = length(data_lab)
    samp = div(size_data,10)
    
    for l in 1:10
        split_range = Array([l])
        for k in 2:samp
            append!(split_range,  Array([l+(k-1)*10]))
        end
        train_data_q = data_q[split_range]
        train_labels = data_lab[split_range]

        test_range = Array(1:size_data)
        test_range = filter(e->!(e in split_range),test_range)

        test_data_q = data_q[test_range]
        test_labels = data_lab[test_range]

        acc_rcdt_max_inf_normalized[l], acc_rcdt_max_2_normalized[l] = mNRCDT_nearest_neighbour(train_data_q, train_labels, test_data_q, test_labels)
    end
    @info "Acc. of max-NRCDT (||.||_inf): \t $(mean(acc_rcdt_max_inf_normalized)) +/- $(std(acc_rcdt_max_inf_normalized))"
    @info "Acc. of max-NRCDT (||.||_2): \t $(mean(acc_rcdt_max_2_normalized)) +/- $(std(acc_rcdt_max_2_normalized))"
end

function classify_flatten_svm(data::AbstractArray, data_lab::AbstractArray)
    acc = zeros(10)
    size_data = length(data_lab)
    samp = div(size_data,10)
    data = Array{Float64}.(data)
    for i in 1:10
        split_range = Array([i])
        for k in 2:samp
            append!(split_range,  Array([i+(k-1)*10]))
        end
        train_data = data[split_range]
        train_labels = data_lab[split_range]

        test_range = Array(1:size_data)
        test_range = filter(e->!(e in split_range),test_range)

        test_data = data[test_range]
        test_labels = data_lab[test_range]

        train_data_reshaped = reshape(collect(Iterators.flatten(train_data)), (length(train_data[1]), length(train_data)))
        clf_model = linear_train(train_labels, train_data_reshaped, solver_type=LIBLINEAR.L2R_L2LOSS_SVC)
        test_data_reshaped = reshape(collect(Iterators.flatten(test_data)), (length(test_data[1]), length(test_data)))
        pred = linear_predict(clf_model, test_data_reshaped)[1]

        acc[i] = mean(test_labels .== pred)
    end     
    @info "Acc. : \t $(mean(acc)) +/- $(std(acc))"
end



#=

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

#####
#####

function gen_rcdt_temp(ref, temp, num_angles, scale_radii, width)
    gen_rcdt_temp_ini = transpose(rcdt(ref, temp[1,:,:], num_angles, scale_radii, width)[1])
    gen_rcdt_temp = zeros(size(temp)[1], size(gen_rcdt_temp_ini)[1], size(gen_rcdt_temp_ini)[2])
    gen_rcdt_temp[1,:,:] = gen_rcdt_temp_ini
    for i in 2:size(temp)[1]
        gen_rcdt_temp[i,:,:] = transpose(rcdt(ref, temp[i,:,:], num_angles, scale_radii, width)[1])
    end
    return gen_rcdt_temp
end

function gen_temp_ext(samp, templates, label, image_size)
    temp = zeros(size(samp)[1], image_size, image_size)
    lab = zeros(size(samp)[1])
    k = 1
    for i in samp
        temp[k,:,:] = imresize(templates[i,:,:], image_size, image_size)    # imresize(convert(Array{Float64}, templates[i]), (image_size, image_size))
        lab[k] = label[i]
        k = k+1
    end
    
    temp_ext = gen_ext(temp, size(temp)[1], image_size)

    return temp_ext, lab
end

function gen_ext(dataset, size_data, image_size)
    datas_ext = zeros(size_data,2*image_size,2*image_size)

    for i in 1:size_data
        image = zeros(2*image_size, 2*image_size)
        a = Int(round((2*image_size - size(dataset[i,:,:])[1])/2))
        b = Int(round((2*image_size - size(dataset[i,:,:])[2])/2))
        image[a:size(dataset[i,:,:])[1]+a-1, b:size(dataset[i,:,:])[2]+b-1] = dataset[i,:,:]
        datas_ext[i,:,:] = image
    end
    return datas_ext
end

#####
#####


function classify_mnist_NRCDT(samp, size_data, random_seed, num_angles_rcdt, num_angles_rcdt_norm, scale_radii, noise, width)

    #####
    #####
    # 
    # Generating the entire MNIST data set
    #
    #####
    #####

    datas, labels = NormalizedRadonCDT.samp_mnist(samp, size_data);
    size_data_small = size_data

    dataset = permutedims(datas, (3, 2, 1))

    @info "Choise of labels: \t $(unique(labels))"
    @info "Size of data: \t $(size(labels)[1])"

    size_data = size(labels)[1]
    image_size = size(dataset)[2]
    dataset = gen_ext(dataset, size_data, image_size)

    # Plot random choice of dataimages
    sel = rand(1:size_data, 9)
    plt1 = plot(layout=(3,3), plot_title="random choice of dataset")
    for i in 1:9
        # plot each set in a different subplot
        plot!(plt1, Gray.(dataset[sel[i],:,:]), subplot=i, xaxis=false, yaxis=false, grid=false)
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
        parameters = [(0.75,1.0),(-0.5,0.5),(-5,5),(-5,5),(-5,5),(1,5,1,1)] #noise: (4,20,2,5), (10,40,2,5)
    elseif noise == 1.5
        parameters = [(0.75,1.0),(-0.5,0.5),(0.,0.),(-5,5),(-5,5)] #noise: (4,20,2,5), (10,40,2,5)
    elseif noise == 2
        parameters = [[10.0, 1.0], [3.0, 10, 5]] 
    elseif noise == 2.3
        parameters = [[10.0, 1.0], [0]] 
        noise = 2
    elseif noise == 2.6
        parameters = [[0], [3.0, 10, 5]] 
        noise = 2
    elseif noise == 3
        parameters_aff = [(0.75,1.0),(-0.5,0.5),(-5,5),(-5,5),(-5,5)]
        parameters_non_aff = [[10.0, 1.0], [2.0, 10, 10]] 
    elseif noise == 3.3
        parameters_aff = [(0.75,1.0),(-0.5,0.5),(-5,5),(-5,5),(-5,5)]
        parameters_non_aff = [[10.0, 1.0], [0]] 
        noise = 3
    elseif noise == 3.6
        parameters_aff = [(1.00,1.00),(-1.0,1.0),(-0.00,0.00),(-15,15),(-15,15)]
        parameters_non_aff = [[0], [2.0, 10, 10]] 
        noise = 3
    end


    datas_noise = zeros(size_data,2*image_size,2*image_size)
    if noise < 2
        for i in 1:size_data
            datas_noise[i,:,:] = gen_dataset_mnist(dataset[i,:,:], 2*image_size, parameters, [[0], [0]], mod(i,10))
        end
    elseif noise == 2
        for i in 1:size_data
            datas_noise[i,:,:] = temp_distortion(dataset[i,:,:], [parameters[1], parameters[2]])
        end
    elseif noise == 3
        for i in 1:size_data
            datas_noise[i,:,:] = gen_dataset_mnist(dataset[i,:,:], 2*image_size, parameters_aff, parameters_non_aff, mod(i,10))
            datas_noise[i,:,:] = temp_distortion(datas_noise[i,:,:], [parameters_non_aff[1], [0]])
        end
    end
    dataset = datas_noise

    for i in 1:size_data
        dataset[i,:,:] = min.(dataset[i,:,:],1)
        # dataset[i,:,:] = ifelse.(dataset[i,:,:] .< 0.25, 0, 1)
        # dataset[i,:,:] = dataset[i,:,:]/sum(dataset[i,:,:])
        dataset[i,:,:] = dataset[i,:,:]/maximum(dataset[i,:,:])
    end

    # Plot random choice of dataimages
    sel = rand(1:size_data, 9)
    plt11 = plot(layout=(3,3), plot_title="random choice of noisy dataset")
    for i in 1:9
        # plot each set in a different subplot
        plot!(plt11, Gray.(dataset[sel[i],:,:])/maximum(dataset[sel[i],:,:]), subplot=i, xaxis=false, yaxis=false, grid=false)
    end
    display(plt11)
    plot(plt11)

    savefig(plt11, "plt_mnist_rand_data_noise.pdf")

    #####
    #####
    #
    # RCDT --- Visualization --- with(/out) normalization
    #
    #
    #####
    #####

    ref = ones(size(dataset[1,:,:])[1], size(dataset[1,:,:])[2])
    
    data_rcdt = gen_rcdt_temp(ref, dataset, num_angles_rcdt, scale_radii, width)

    data_rcdt_normalized = mNR_CDT(data_rcdt)

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

    split = Int(size_data_small/10)             # -> an 10-fold-cross validation

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


    @info "============================================================================="
    @info "Acc. of Euclidean : \t $(mean(acc_euclid)) +/- $(std(acc_euclid))"
    @info "-----------------------------------------------------------------------------"


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
    
    @info "Acc. of RCDT with : \t $(maximum(acc_rcdt_k)) +/- $(std(acc_rcdt_k))"
    @info "-----------------------------------------------------------------------------"

    #####
    #####
    #
    # NRCDT --- Computation --- with normalization
    #
    #
    #####
    #####

    acc_rcdt_max_normalized = zeros(10)
    acc_rcdt_mean_normalized = zeros(10)
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

        train_rcdt = gen_rcdt_temp(ref, train_data, num_angles_rcdt_norm, scale_radii, width)

        train_rcdt_max_normalized = mNR_CDT(train_rcdt)
        train_rcdt_mean_normalized = aNR_CDT(train_rcdt)

        clf_rcdt_normalized_max_model = linear_train(train_labels, transpose(train_rcdt_max_normalized),  solver_type=LIBLINEAR.L2R_L2LOSS_SVC) # C = 1e10, verbose=false);
        clf_rcdt_normalized_mean_model = linear_train(train_labels, transpose(train_rcdt_mean_normalized),  solver_type=LIBLINEAR.L2R_L2LOSS_SVC) # C = 1e10, verbose=false);

        test_rcdt = gen_rcdt_temp(ref, test_data, num_angles_rcdt_norm, scale_radii, width)

        test_rcdt_max_normalized = mNR_CDT(test_rcdt)
        test_rcdt_mean_normalized = aNR_CDT(test_rcdt)

        pred_rcdt_max_normalized = linear_predict(clf_rcdt_normalized_max_model, transpose(test_rcdt_max_normalized))[1];
        pred_rcdt_mean_normalized = linear_predict(clf_rcdt_normalized_mean_model, transpose(test_rcdt_mean_normalized))[1];

        acc_rcdt_max_normalized[l] = mean(test_labels .== pred_rcdt_max_normalized)
        acc_rcdt_mean_normalized[l] = mean(test_labels .== pred_rcdt_mean_normalized)
    end

    @info "Acc. of max-NRCDT : \t $(mean(acc_rcdt_max_normalized)) +/- $(std(acc_rcdt_max_normalized))"
    @info "Acc. of mean-NRCDT : \t $(mean(acc_rcdt_mean_normalized)) +/- $(std(acc_rcdt_mean_normalized))"
    @info "-----------------------------------------------------------------------------"
end

=#
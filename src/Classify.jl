using NormalizedRadonCDT.radon_cdt
using Interpolations: LinearInterpolation as LinInter
using LIBSVM, LIBLINEAR

function prepare_data(data, num_angles, width)
    ref = ones(size(data[1,:,:]));
    data1 = rcdt(ref, data[1,:,:], num_angles, width)[1];
    data_transformed = zeros(size(data)[1], size(data1)[1], size(data1)[2]);
    data_transformed[1,:,:] = data1;
    for i in 2:size(data)[1]
        data_transformed[i,:,:] = rcdt(ref, data[i,:,:], num_angles, width)[1];
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

function classify_data_NRCDT(samp, image_size, data_size, random_seed, num_angles_rcdt, num_angles_rcdt_norm, num_rcdt, width)

    templates = load("temp.jld")["temp"]
    label = range(1, size(templates)[1])

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

    parameters = [(0.75,1.25),(-5.,5.),(-0.5,0.5),(-10,10),(-10,10),(4,20,2,5)] #noise: (4,20,2,5), (10,40,2,5)

    dataset, labels =  gen_dataset(temp, lab, 2*image_size, data_size, parameters, random_seed)

    dataset = round.(5*dataset)
    dataset[dataset .> 1] .= 1

    length = size(dataset)[1]
    sel = rand(1:length, 9)

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
    
    temp_rcdt1 = transpose(rcdt(ref, temp[1,:,:], num_angles_rcdt, width)[1])
    temp_rcdt = zeros(size(temp)[1], size(temp_rcdt1)[1], size(temp_rcdt1)[2])
    temp_rcdt[1,:,:] = temp_rcdt1
    for i in 2:size(temp)[1]
        temp_rcdt[i,:,:] = transpose(rcdt(ref, temp[i,:,:], num_angles_rcdt, width)[1])
    end
    #
    data_rcdt1 = transpose(rcdt(ref, dataset[1,:,:], num_angles_rcdt, width)[1])
    data_rcdt = zeros(size(dataset)[1], size(data_rcdt1)[1], size(data_rcdt1)[2])
    data_rcdt[1,:,:] = data_rcdt1
    for i in 2:size(dataset)[1]
        data_rcdt[i,:,:] = transpose(rcdt(ref, dataset[i,:,:], num_angles_rcdt, width)[1])
    end

    temp_rcdt_normalized = (temp_rcdt .- mean(temp_rcdt, dims=2))./sqrt.(var(temp_rcdt, dims=2))
    temp_rcdt_normalized = dropdims(maximum(temp_rcdt_normalized, dims=3), dims=3)
    data_rcdt_normalized = (data_rcdt .- mean(data_rcdt, dims=2))./sqrt.(var(data_rcdt, dims=2))
    data_rcdt_normalized = dropdims(maximum(data_rcdt_normalized, dims=3), dims=3)

    
    # Plot of results
    plt3 = plot(layout=(size(samp)[1], 2), plot_title="transform of unnormalized and normalized")
    for i in 1:size(samp)[1]
        for j in 1:size(dataset)[1]
            if labels[j] == lab[i]
                # plot each set in a different subplot
                plot!(plt3, data_rcdt[j,:,1], subplot=(i-1)*2+1, legend=false);
                plot!(plt3, data_rcdt_normalized[j,:], subplot=(i-1)*2+2, legend=false);
            end
        end
    end
    for i in 1:size(samp)[1]
        plot!(plt3, temp_rcdt[i,:,1], subplot=(i-1)*2+1, legend=false, color=:black);
        plot!(plt3, temp_rcdt_normalized[i,:], subplot=(i-1)*2+2, legend=false, color=:black);
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

    dataset, labels = shuffle_data(dataset, labels, random_seed)
    split = 5*1e-2      # .05
    train_data, train_labels, test_data, test_labels = split_data(dataset, labels, split)

    #####
    #####
    #
    # Euclidean --- Computation
    #
    #
    #####
    #####

    train_data_reshaped = reshape(train_data, (size(train_data)[1], size(train_data)[2] * size(train_data)[3]));

    clf_euclid_model = linear_train(train_labels, transpose(train_data_reshaped), verbose=false);

    test_data_reshaped = reshape(test_data, (size(test_data)[1], size(test_data)[2] * size(test_data)[3]));

    pred_euclid = linear_predict(clf_euclid_model, transpose(test_data_reshaped))[1];

    acc_euclid = mean(test_labels .== pred_euclid)     
    println("=============================================================================================")
    println("Accuracy of Euclidean : \t", acc_euclid)
    println("---------------------------------------------------------------------------------------------")


    #####
    #####
    #
    # RCDT --- Computation --- without normalization
    #
    #
    #####
    #####

    acc_rcdt = zeros(0)
    for i in 1:num_rcdt

        train_data_transformed = prepare_data(train_data, i, width)

        clf_rcdt_model = linear_train(train_labels, transpose(train_data_transformed),  C = 1e10, verbose=false);

        test_data_transformed = prepare_data(test_data, i, width)

        pred_rcdt = linear_predict(clf_rcdt_model, transpose(test_data_transformed))[1];
        append!(acc_rcdt, mean(test_labels .== pred_rcdt))

        println("Accuracy of RCDT with \t", i, "\t instance(s) : \t", acc_rcdt[i])
        println("---------------------------------------------------------------------------------------------")
    end

    #####
    #####
    #
    # NRCDT --- Computation --- with normalization
    #
    #
    #####
    #####

    ref = ones(size(train_data)[2], size(train_data)[3])
    
    train_rcdt1 = transpose(rcdt(ref, train_data[1,:,:], num_angles_rcdt_norm, width)[1])
    train_rcdt = zeros(size(train_data)[1], size(train_rcdt1)[1], size(train_rcdt1)[2])
    train_rcdt[1,:,:] = train_rcdt1
    for i in 2:size(train_data)[1]
        train_rcdt[i,:,:] = transpose(rcdt(ref, train_data[i,:,:], num_angles_rcdt_norm, width)[1])
    end
    train_rcdt_normalized = (train_rcdt .- mean(train_rcdt, dims=2))./sqrt.(var(train_rcdt, dims=2))
    train_rcdt_normalized_max = dropdims(maximum(train_rcdt_normalized, dims=3), dims=3)
    train_rcdt_normalized_mean = dropdims(mean(train_rcdt_normalized, dims=3), dims=3)

    clf_rcdt_normalized_model_max = linear_train(train_labels, transpose(train_rcdt_normalized_max),  C = 1e10, verbose=false);
    clf_rcdt_normalized_model_mean = linear_train(train_labels, transpose(train_rcdt_normalized_mean),  C = 1e10, verbose=false);


    test_rcdt1 = transpose(rcdt(ref, test_data[1,:,:], num_angles_rcdt_norm, width)[1])
    test_rcdt = zeros(size(test_data)[1], size(test_rcdt1)[1], size(test_rcdt1)[2])
    test_rcdt[1,:,:] = test_rcdt1
    for i in 2:size(test_data)[1]
        test_rcdt[i,:,:] = transpose(rcdt(ref, test_data[i,:,:], num_angles_rcdt_norm, width)[1])
    end
    test_rcdt_normalized = (test_rcdt .- mean(test_rcdt, dims=2))./sqrt.(var(test_rcdt, dims=2))
    test_rcdt_normalized_max = dropdims(maximum(test_rcdt_normalized, dims=3), dims=3)
    test_rcdt_normalized_mean = dropdims(mean(test_rcdt_normalized, dims=3), dims=3)


    pred_rcdt_normalized_max = linear_predict(clf_rcdt_normalized_model_max, transpose(test_rcdt_normalized_max))[1];
    pred_rcdt_normalized_mean = linear_predict(clf_rcdt_normalized_model_mean, transpose(test_rcdt_normalized_mean))[1];
    
    acc_rcdt_normalized_max = mean(test_labels .== pred_rcdt_normalized_max)
    acc_rcdt_normalized_mean = mean(test_labels .== pred_rcdt_normalized_mean)

    println("Accuracy of max-NRCDT : \t", acc_rcdt_normalized_max, "\t", "Accuracy of mean-NRCDT : \t", acc_rcdt_normalized_mean)
    println("---------------------------------------------------------------------------------------------")
    
end
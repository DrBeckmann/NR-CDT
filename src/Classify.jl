module Classify 

using Plots
using LIBLINEAR
using Statistics
using StatsBase
using Random

export plot_quantiles
export accuracy_k_nearest_neighbour, accuracy_k_nearest_cross_neighbour, accuracy_cross_svm, accuracy_k_nearest_part_neighbour, accuracy_part_svm, plot_quantiles_single

function plot_quantiles(temp_q::AbstractArray, temp_lab::AbstractArray, data_q::AbstractArray, data_lab::AbstractArray)
    dd = length(data_q[1])
    plt = plot(size = (500,350))
    for i in 1:length(temp_lab), j in 1:length(data_lab)
        if data_lab[j] == temp_lab[i]
            plot!(plt, data_q[j], label=false, linecolor = RGBA((abs(0.5-(i-1)/(length(temp_lab)-1)))*(1-(i-1)/(length(temp_lab)-1))*0.99, ((i-1)/(length(temp_lab)-1))*(1-(i-1)/(length(temp_lab)-1))*0.99, (abs(0.5-(i-1)/(length(temp_lab)-1)))*((i-1)/(length(temp_lab)-1))*0.99, 0.5), yticks=true, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
        end
    end
    for i in 1:length(temp_lab)
        k = temp_lab[i]
        plot!(plt, temp_q[i], fontfamily="Computer Modern", label=["class $k" "k"], linewidth=2 , linecolor = RGBA((abs(0.5-(i-1)/(length(temp_lab)-1)))*(1-(i-1)/(length(temp_lab)-1))*0.99, ((i-1)/(length(temp_lab)-1))*(1-(i-1)/(length(temp_lab)-1))*0.85, (abs(0.5-(i-1)/(length(temp_lab)-1)))*((i-1)/(length(temp_lab)-1))*0.99, 1), yticks=true, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));    
    end
    return plt
end

function plot_quantiles_single(temp_q::AbstractArray, data_q::AbstractArray, i::Int64)
    dd = length(data_q[1])
    plt = plot(size = (500,350))
    for j in 1:length(data_q)
        plot!(plt, data_q[j], label=false, linecolor = RGBA((abs(0.5-(i-1)/i))*(1-(i-1)/i)*0.99, ((i-1)/i)*(1-(i-1)/i)*0.99, (abs(0.5-(i-1)/i))*((i-1)/i)*0.99, 0.5), yticks=true, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
    end
    plot!(plt, temp_q[1], fontfamily="Computer Modern", legend=false, linewidth=2 , linecolor = RGBA((abs(0.5-(i-1)/i))*(1-(i-1)/i)*0.99, ((i-1)/i)*(1-(i-1)/i)*0.85, (abs(0.5-(i-1)/i))*((i-1)/i)*0.99, 1), yticks=true, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));    
    return plt
end

function argmink(a::AbstractArray, k::Int64)
    b = partialsortperm(a, length(a)-k+1:length(a), rev=true)
    # return collect(zip(b, a[b]))
    return collect(b[end:-1:1])
end

function argmaxk(a::AbstractArray, k::Int64)
    b = partialsortperm(a, 1:k, rev=true)
    # return collect(zip(b, a[b]))
    return collect(b)
end

function slice_matrix(A::AbstractMatrix)
    return [c[:] for c in eachcol(A)]
end

function accuracy_k_nearest_neighbour(
    temp_q::AbstractArray, 
    temp_lab::AbstractArray, 
    data_q::AbstractArray, 
    data_lab::AbstractArray,
    norm::String;
    K::Int64=1, 
    ret::Int64=0)

    temp_q = collect.(Iterators.flatten.(collect.(temp_q)))
    data_q = collect.(Iterators.flatten.(collect.(data_q)))

    pred_rcdt = zeros(length(temp_q),length(data_q))
    for k in 1:length(data_q), kk in 1:length(temp_q)
        if norm=="inf"
            pred_rcdt[kk,k] = maximum(abs.(temp_q[kk] .- data_q[k]))
        end
        if norm=="euclidean"
            pred_rcdt[kk,k] = sqrt(sum((temp_q[kk] .- data_q[k]).*(temp_q[kk] .- data_q[k])))
        end
    end
    # pred_label_rcdt = argmin(pred_rcdt, dims=1)
    K_pred_index_rcdt = argmink.(slice_matrix(pred_rcdt), K)
    K_pred_label_rcdt = zeros(length(K_pred_index_rcdt),K)
    for l in 1:K, ll in 1:length(K_pred_index_rcdt)
        K_pred_label_rcdt[ll,l] = temp_lab[K_pred_index_rcdt[ll][l]]
    end
    dict_K_pred_label_rcdt = countmap.(slice_matrix(transpose(Int64.(K_pred_label_rcdt))))
    pred_label_rcdt = findmax.(dict_K_pred_label_rcdt) 
    label_rcdt = zeros(length(pred_label_rcdt))
    for k in 1:length(data_q)
        label_rcdt[k] = pred_label_rcdt[k][2]
    end 
    acc_rcdt = mean(data_lab .== label_rcdt)
    conf_matrix = confusion_matrix(data_lab, label_rcdt) 
    if ret==1
        @info "Acc. using $(norm)-norm : \t $(acc_rcdt)"
    end
    return acc_rcdt, conf_matrix
end

function accuracy_k_nearest_cross_neighbour(data_q::AbstractArray, data_lab::AbstractArray, norm::String; K::Int64=1)

    # variabel cross <-- TO DO

    acc_rcdt = zeros(10)
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

        acc_rcdt[l], _ = accuracy_k_nearest_neighbour(train_data_q, train_labels, test_data_q, test_labels, norm, K=K)
    end
    @info "Acc. using $(norm)-norm : \t $(mean(acc_rcdt)) +/- $(std(acc_rcdt))"
end

function accuracy_k_nearest_part_neighbour(
    part::Int64, 
    samp::Int64, 
    class_size::Int64, 
    class_number::Int64, 
    data_q::AbstractArray, 
    data_lab::AbstractArray, 
    norm::String; 
    K::Int64=1,
    ret::Int64=0)

    # variabel cross <-- TO DO

    if samp > class_size
        return "Error : unexpected devision of classes."
    end
    if ret==1
        conf_matrix = zeros(length(unique(data_lab)), length(unique(data_lab)))
    end

    acc_rcdt = zeros(part)
    size_data = length(data_lab)
    
    for i in 1:part
        split_range = []
        for l in 1:class_number
            choice = collect(1+(l-1)*class_size:l*class_size)
            shuffle!(choice)
            append!(split_range, choice[1:samp])
        end
        train_data_q = data_q[split_range]
        train_labels = data_lab[split_range]

        test_range = Array(1:size_data)
        test_range = filter(e->!(e in split_range),test_range)

        test_data_q = data_q[test_range]
        test_labels = data_lab[test_range]

        acc_rcdt[i], conf_matrix_part = accuracy_k_nearest_neighbour(train_data_q, train_labels, test_data_q, test_labels, norm, K=K)
        # acc_rcdt[i] = accuracy_k_nearest_neighbour(train_data_q, train_labels, test_data_q, test_labels, norm, K=K)
        if ret==1
            conf_matrix += conf_matrix_part 
        end
    end
    @info "Acc. using $(norm)-norm : \t $(mean(acc_rcdt)) +/- $(std(acc_rcdt))"
    if ret==1
        return conf_matrix
    end
end

function accuracy_part_svm(
        part::Int64, 
        samp::Int64, 
        class_size::Int64, 
        class_number::Int64, 
        data_q::AbstractArray, 
        data_lab::AbstractArray;
        ret::Int64=0)

    acc_rcdt = zeros(part)
    size_data = length(data_lab)
    data_q = Array{Float64}.(data_q)
    if ret==1
        conf_matrix = zeros(length(unique(data_lab)), length(unique(data_lab)))
    end
    for i in 1:part
        split_range = []
        for l in 1:class_number
            choice = collect(1+(l-1)*class_size:l*class_size)
            shuffle!(choice)
            append!(split_range, choice[1:samp])
        end
        train_data_q = data_q[split_range]
        train_labels = data_lab[split_range]
        
        test_range = Array(1:size_data)
        test_range = filter(e->!(e in split_range),test_range)

        test_data_q = data_q[test_range]
        test_labels = data_lab[test_range]

        train_data_q_reshaped = reshape(collect(Iterators.flatten(train_data_q)), (length(train_data_q[1]), length(train_data_q)))
        clf_model = linear_train(train_labels, train_data_q_reshaped, solver_type=LIBLINEAR.L2R_L1LOSS_SVC_DUAL)
        test_data_q_reshaped = reshape(collect(Iterators.flatten(test_data_q)), (length(test_data_q[1]), length(test_data_q)))
        pred_labels = linear_predict(clf_model, test_data_q_reshaped)[1]

        acc_rcdt[i] = mean(test_labels .== pred_labels)
        if ret==1
            conf_matrix += confusion_matrix(test_labels, pred_labels) 
        end
    end     
    @info "Acc. : \t $(mean(acc_rcdt)) +/- $(std(acc_rcdt))"
    if ret==1
        return conf_matrix
    end
end

function accuracy_cross_svm(
        data_q::AbstractArray, 
        data_lab::AbstractArray)
    acc = zeros(10)
    size_data = length(data_lab)
    samp = div(size_data,10)
    data_q = Array{Float64}.(data_q)
    for i in 1:10
        split_range = Array([i])
        for k in 2:samp
            append!(split_range,  Array([i+(k-1)*10]))
        end
        train_data_q = data_q[split_range]
        train_labels = data_lab[split_range]

        test_range = Array(1:size_data)
        test_range = filter(e->!(e in split_range),test_range)

        test_data_q = data_q[test_range]
        test_labels = data_lab[test_range]

        train_data_q_reshaped = reshape(collect(Iterators.flatten(train_data_q)), (length(train_data_q[1]), length(train_data_q)))
        clf_model = linear_train(train_labels, train_data_q_reshaped, solver_type=LIBLINEAR.L2R_L1LOSS_SVC_DUAL)
        test_data_q_reshaped = reshape(collect(Iterators.flatten(test_data_q)), (length(test_data_q[1]), length(test_data_q)))
        pred_labels = linear_predict(clf_model, test_data_q_reshaped)[1]

        acc[i] = mean(test_labels .== pred_labels)
    end     
    @info "Acc. : \t $(mean(acc)) +/- $(std(acc))"
end

function confusion_matrix(true_labels::AbstractArray, predicted_labels::AbstractArray)
    # Get the sorted unique labels from both true and predicted labels
    labels = sort(unique(vcat(true_labels, predicted_labels)))
    n_labels = length(labels)
    
    # Create a mapping from label to index
    label_to_index = Dict(label => idx for (idx, label) in enumerate(labels))
    
    # Initialize the confusion matrix as an integer matrix
    matrix = zeros(Int, n_labels, n_labels)
    
    # Fill in the confusion matrix
    for (t, p) in zip(true_labels, predicted_labels)
        i = label_to_index[t]
        j = label_to_index[p]
        matrix[i, j] += 1
    end
    
    return matrix
end

end
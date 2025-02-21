using Plots
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

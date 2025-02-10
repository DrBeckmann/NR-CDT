### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 03f9300e-692e-11ef-145c-85ecce1e4c7f
begin
	import Pkg
	Pkg.activate("..")
	using Revise
	using NormalizedRadonCDT.TestImages
	using NormalizedRadonCDT
end

# ╔═╡ 75fa041d-f4a0-4cb0-8af7-b09ff2249874
using MLDatasets, Plots, Random, Statistics;

# ╔═╡ 2c240951-7a75-4633-9414-bb4d102b5b86
using Metalhead, Flux

# ╔═╡ 0eb4cfdc-c875-4d0f-b46e-531e72fc06c3
trainset = MNIST(:train)

# ╔═╡ 907c64eb-30d8-4cbe-af47-3d4b6d72e3ca
Null = trainset[2].features;

# ╔═╡ 997d8a2c-b11b-4d9e-9812-927a75454479
heatmap(Null)

# ╔═╡ 161664b7-c90c-4ff3-b7d4-1f5ccd536459
heatmap(NormalizedRadonCDT.RadonTransform.radon(Float64.(Null), 40, 1060, 0.0))

# ╔═╡ a71828ea-c868-4a39-abf9-4f0a63b10844
number_mnist_1 = [1,7]

# ╔═╡ 4cb99a9d-f66c-4265-becb-c59aec9daf9f
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 1000, 42, 16, 16, 2, 0, 0)

# ╔═╡ 900fcbc7-f732-468f-af2f-cd5a4ab7cfc7
datas, labels = NormalizedRadonCDT.samp_mnist(number_mnist_1, 10);

# ╔═╡ 72027553-ee48-4a9e-a0b4-98af32129694
dataset = permutedims(datas, (3, 2, 1));

# ╔═╡ 6e351556-87a0-4682-86a9-3d5d78b8d31d
datas

# ╔═╡ 7a34e5f1-f944-45d7-bba1-82d36db7cf8f
split_range = [1,11];

# ╔═╡ 5762b667-24fa-428f-9298-9518e5dd7db4
test_range = Array(1:20);

# ╔═╡ 2a1f846a-c41c-4841-973d-7639c402abf2
filter!(e->!(e in split_range),test_range);

# ╔═╡ c3a37d79-3c66-482c-b565-97faaa5fced2
train_data = dataset[split_range,:,:];

# ╔═╡ 6cbb7a31-3b84-4bf0-9d78-b30a273b5fad
train_labels = labels[split_range];

# ╔═╡ 825e2f46-5408-492b-b1ac-f56769749ef5
test_data = dataset[test_range,:,:];

# ╔═╡ 931bcf4f-e6c8-4880-8162-4a8a4ed61f47
test_labels = labels[test_range];

# ╔═╡ 713c71db-c245-4c5c-a764-164afd2880dd
function preprocess(dataset, label)
    x = dataset
	y = label
	size = size(x[1,1,:])[1]
	print(size)

    # Add singleton color-channel dimension to features for Conv-layers
    x = reshape(x, 28, 28, 1, :)

    # One-hot encode targets
    y = Flux.onehotbatch(y, 0:9)

    return x, y
end;

# ╔═╡ 9df3b6b8-6aae-4f55-ab04-6a6a2fcace57
x_train, y_train = preprocess(train_data, train_labels)

# ╔═╡ 41f3223d-7f37-4591-a122-a49484f4cb92
train_data

# ╔═╡ 533a3c88-706d-466a-920d-fab53301a060
x_test, y_test = preprocess(test_data, test_labels)

# ╔═╡ a4e3b6a0-4e3c-4418-a929-5f30143e32a1
train_loader = Flux.DataLoader((x_train, y_train), batchsize=2, shuffle=true)

# ╔═╡ efb5ec73-5c3c-4bdc-9c3e-f0628b63496f
x_train

# ╔═╡ b877d1a0-0ef9-4e4f-8786-a4b3a0562688
resnet = ResNet(18);

# ╔═╡ 05665910-30be-404f-904a-453ce6765264
model = Chain(
    Conv((5, 5), 1 => 6, relu),  # 1 input color channel
    MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(256, 120, relu),
    Dense(120, 84, relu),
    Dense(84, 10),  # 10 output classes
)

# ╔═╡ 345018ab-6100-487c-a02a-d97c3e920b0e
loss_fn(ŷ, y) = Flux.logitcrossentropy(ŷ, y)

# ╔═╡ d5ca35d0-b015-4893-980c-758cb96d9859
optim = Flux.setup(Adam(3.0f-4), model)

# ╔═╡ f6020202-6323-4c34-aa0f-e8b5761f4f28
losses = Float32[]

# ╔═╡ a3d536fa-36c3-4a11-87bf-7f99e45ef0b1
function accuracy(model1, x_test1, y_test1)
    # Use onecold to return class index
    ŷ = Flux.onecold(model1(x_test1))
    y = Flux.onecold(y_test1)

    return mean(ŷ .== y)
end

# ╔═╡ 956211f2-3d0b-44c0-835d-bbe09be950b4
for epoch in 1:10
	for (x, y) in train_loader
		# Compute loss and gradients of model w.r.t. its parameters
		loss, grads = Flux.withgradient(m -> loss_fn(m(x), y), model)
		Flux.update!(optim, model, grads[1])
		push!(losses, loss)
		acc = accuracy(model, x_test, y_test) * 100
		@info "Epoch $epoch :\t loss = $(loss), acc = $(acc)%"
	end
end

# ╔═╡ faee3d81-b595-442d-b1b3-9b7b50d3b0e5
plot(losses; xlabel="Step", ylabel="Loss", yaxis=:log) # runs after training

# ╔═╡ 1c6e0bbd-3d85-42cd-a07c-2f9b13f58564
train_loader

# ╔═╡ 4c277e68-2ff5-4490-b27e-1d78d392ee51
enumerate(train_loader)

# ╔═╡ Cell order:
# ╠═03f9300e-692e-11ef-145c-85ecce1e4c7f
# ╠═75fa041d-f4a0-4cb0-8af7-b09ff2249874
# ╠═0eb4cfdc-c875-4d0f-b46e-531e72fc06c3
# ╠═907c64eb-30d8-4cbe-af47-3d4b6d72e3ca
# ╠═997d8a2c-b11b-4d9e-9812-927a75454479
# ╠═161664b7-c90c-4ff3-b7d4-1f5ccd536459
# ╠═a71828ea-c868-4a39-abf9-4f0a63b10844
# ╠═4cb99a9d-f66c-4265-becb-c59aec9daf9f
# ╠═2c240951-7a75-4633-9414-bb4d102b5b86
# ╠═900fcbc7-f732-468f-af2f-cd5a4ab7cfc7
# ╠═72027553-ee48-4a9e-a0b4-98af32129694
# ╠═6e351556-87a0-4682-86a9-3d5d78b8d31d
# ╠═7a34e5f1-f944-45d7-bba1-82d36db7cf8f
# ╠═5762b667-24fa-428f-9298-9518e5dd7db4
# ╠═2a1f846a-c41c-4841-973d-7639c402abf2
# ╠═c3a37d79-3c66-482c-b565-97faaa5fced2
# ╠═6cbb7a31-3b84-4bf0-9d78-b30a273b5fad
# ╠═825e2f46-5408-492b-b1ac-f56769749ef5
# ╠═931bcf4f-e6c8-4880-8162-4a8a4ed61f47
# ╠═713c71db-c245-4c5c-a764-164afd2880dd
# ╠═9df3b6b8-6aae-4f55-ab04-6a6a2fcace57
# ╠═41f3223d-7f37-4591-a122-a49484f4cb92
# ╠═533a3c88-706d-466a-920d-fab53301a060
# ╠═a4e3b6a0-4e3c-4418-a929-5f30143e32a1
# ╠═efb5ec73-5c3c-4bdc-9c3e-f0628b63496f
# ╠═b877d1a0-0ef9-4e4f-8786-a4b3a0562688
# ╠═05665910-30be-404f-904a-453ce6765264
# ╠═345018ab-6100-487c-a02a-d97c3e920b0e
# ╠═d5ca35d0-b015-4893-980c-758cb96d9859
# ╠═f6020202-6323-4c34-aa0f-e8b5761f4f28
# ╠═a3d536fa-36c3-4a11-87bf-7f99e45ef0b1
# ╠═956211f2-3d0b-44c0-835d-bbe09be950b4
# ╠═faee3d81-b595-442d-b1b3-9b7b50d3b0e5
# ╠═1c6e0bbd-3d85-42cd-a07c-2f9b13f58564
# ╠═4c277e68-2ff5-4490-b27e-1d78d392ee51

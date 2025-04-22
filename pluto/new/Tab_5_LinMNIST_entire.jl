### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 8cbe0300-edff-11ef-2fad-d3b8cca171a9
begin
	import Pkg
	Pkg.activate("../..")
	using Revise
	using NormalizedRadonCDT
	using NormalizedRadonCDT.TestImages
	using NormalizedRadonCDT.DataTransformations
	using NormalizedRadonCDT.Classify
	using Images
	using Plots
	using MLDatasets
	using JLD2
	using Random
	Random.seed!(42)
end

# ╔═╡ 81783bfb-d7a2-4c18-a4f8-b634f3bbc59b
trainset = MNIST(:train)

# ╔═╡ 98131234-5ab0-4954-bd68-0646241ed22a
typeof(trainset)

# ╔═╡ 81170d86-6140-41ce-a1e4-24e70c0530ff
MLClass, MLLabel = generate_ml_classes(trainset, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0], 500, 0.666);

# ╔═╡ bf8448db-7cb1-42ba-9f1e-03b775b31cb8
MLClass

# ╔═╡ 773832af-9099-4dcf-bd1b-c82baaa83424
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.75, 1.0), 
	scale_y = (0.75, 1.0),
	rotate=(-180.0, 180.0), 
	#shear_x=(-5.0, 5.0),
	#shear_y=(-5.0, 5.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))

# ╔═╡ fb3629dc-1860-4a96-a75e-2b4402f847fe
Random.seed!(42); TMLClass = A.(MLClass)

# ╔═╡ 8fb1f5c3-386e-4117-9b87-dedb75c1ae1d
R = RadonTransform(300,128,0.0)

# ╔═╡ bbbcd04c-8b4f-4c44-958d-9e4089ada051
RCDT = RadonCDT(64, R)

# ╔═╡ 81fd55d8-24df-4047-b235-20468b2c111c
# ╠═╡ disabled = true
#=╠═╡
NRCDT = NormRadonCDT(RCDT)
  ╠═╡ =#

# ╔═╡ 81e32395-78d9-4a5f-b6f0-ba2d6f01c8ee
# ╠═╡ disabled = true
#=╠═╡
mNRCDT = MaxNormRadonCDT(RCDT); mqClass = mNRCDT.(TMLClass)
  ╠═╡ =#

# ╔═╡ 13849a92-c3d3-42e9-a959-ca456452aaeb
# ╠═╡ disabled = true
#=╠═╡
aNRCDT = MeanNormRadonCDT(RCDT); aqClass = aNRCDT.(TMLClass)
  ╠═╡ =#

# ╔═╡ 908e9571-1cea-4333-b087-76f08de3f485
rcdt = RCDT.(TMLClass)

# ╔═╡ 0084b432-9cf1-4f6e-a8dc-8127ec832643
# ╠═╡ disabled = true
#=╠═╡
rqMLClass = filter_angles.(rcdt, 128, 128)
  ╠═╡ =#

# ╔═╡ c678f383-6fc4-4264-a045-b85decb41528
mqMLClass = max_normalization.(rcdt)

# ╔═╡ 4bd83071-ed45-4eb1-b6a8-667c44c27a7e
aqMLClass = mean_normalization.(rcdt)

# ╔═╡ 68dc3cd6-7b79-44fa-9116-6470b7b0b485
# ╠═╡ disabled = true
#=╠═╡
for angle in [1,2,3,4,5,6,128,256]
	R = RadonTransform(256,angle,0.0);
	RCDT = RadonCDT(256, R);
	NRCDT = NormRadonCDT(RCDT);
	mNRCDT = MaxNormRadonCDT(RCDT);
	aNRCDT = MeanNormRadonCDT(RCDT);
	mqTemp = mNRCDT.(TMLClass);
	aqTemp = aNRCDT.(TMLClass);
	@info "number of equispaced angles:" angle
	accuracy_k_nearest_part_neighbour(20, 10, 100, 3, mqTemp, MLLabel, "inf", K=1);
	accuracy_k_nearest_part_neighbour(20, 10, 100, 3, mqTemp, MLLabel, "euclidean", K=1);
	accuracy_k_nearest_part_neighbour(20, 10, 100, 3, aqTemp, MLLabel, "inf", K=1);
	accuracy_k_nearest_part_neighbour(20, 10, 100, 3, aqTemp, MLLabel, "euclidean", K=1);
end
  ╠═╡ =#

# ╔═╡ edae2657-66a9-4976-abe5-d8576dd6eab6
for prop in [11,25,50]
	for KK in [1,5,11]
		@info "split" prop, "k-NN" KK
		Random.seed!(42); CC = accuracy_k_nearest_part_neighbour(20, prop, 500, 10, rcdt, MLLabel, "inf", K=KK, ret=1);
		jldsave("conf_LinMNIST_$(KK)NN_$(prop)_RCDT_inf.jld2"; CC)
		Random.seed!(42); CC = accuracy_k_nearest_part_neighbour(20, prop, 500, 10, rcdt, MLLabel, "euclidean", K=KK, ret=1);
		jldsave("conf_LinMNIST_$(KK)NN_$(prop)_RCDT_eucl.jld2"; CC)
		Random.seed!(42); CC = accuracy_k_nearest_part_neighbour(20, prop, 500, 10, mqMLClass, MLLabel, "inf", K=KK, ret=1);
		jldsave("conf_LinMNIST_$(KK)NN_$(prop)_maxNRCDT_inf.jld2"; CC)
		Random.seed!(42); CC = accuracy_k_nearest_part_neighbour(20, prop, 500, 10, mqMLClass, MLLabel, "euclidean", K=KK, ret=1);
		jldsave("conf_LinMNIST_$(KK)NN_$(prop)_maxNRCDT_eucl.jld2"; CC)
		Random.seed!(42); CC = accuracy_k_nearest_part_neighbour(20, prop, 500, 10, aqMLClass, MLLabel, "inf", K=KK, ret=1);
		jldsave("conf_LinMNIST_$(KK)NN_$(prop)_meanNRCDT_inf.jld2"; CC)
		Random.seed!(42); CC = accuracy_k_nearest_part_neighbour(20, prop, 500, 10, aqMLClass, MLLabel, "euclidean", K=KK, ret=1);
		jldsave("conf_LinMNIST_$(KK)NN_$(prop)_meanNRCDT_eucl.jld2"; CC)
	end
end

# ╔═╡ ecb0c307-2f8d-4fb7-8688-d9f72cc9e992
for prop in [11,25,50]
	for KK in [1,5,11]
		@info "split" prop, "k-NN" KK
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 500, 10, Array{Float64}.(TMLClass), MLLabel, "inf", K=KK);
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 500, 10, Array{Float64}.(TMLClass), MLLabel, "euclidean", K=KK);
	end
end

# ╔═╡ 88baa5c9-3de6-47a9-941d-57c8d6b2ef3d
# ╠═╡ disabled = true
#=╠═╡
Random.seed!(42); CC = accuracy_k_nearest_part_neighbour(20, 50, 500, 10, aqMLClass, MLLabel, "euclidean", K=11, ret=1);
  ╠═╡ =#

# ╔═╡ 6bf26e05-1d50-4516-9432-deaa0a00e4bf
# ╠═╡ disabled = true
#=╠═╡
hh = heatmap(CC/20, fontfamily="Computer Modern", size=(550,500), xticks=(1:10, 0:9), yticks=(1:10, 0:9))
  ╠═╡ =#

# ╔═╡ 62095eb0-aa7d-4c55-b8a6-34e7d344461a
# ╠═╡ disabled = true
#=╠═╡
savefig(hh, "linmnist_11NN_50_mean_eucl.pdf")
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╠═81783bfb-d7a2-4c18-a4f8-b634f3bbc59b
# ╠═98131234-5ab0-4954-bd68-0646241ed22a
# ╠═81170d86-6140-41ce-a1e4-24e70c0530ff
# ╠═bf8448db-7cb1-42ba-9f1e-03b775b31cb8
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═fb3629dc-1860-4a96-a75e-2b4402f847fe
# ╠═8fb1f5c3-386e-4117-9b87-dedb75c1ae1d
# ╠═bbbcd04c-8b4f-4c44-958d-9e4089ada051
# ╠═81fd55d8-24df-4047-b235-20468b2c111c
# ╠═81e32395-78d9-4a5f-b6f0-ba2d6f01c8ee
# ╠═13849a92-c3d3-42e9-a959-ca456452aaeb
# ╠═908e9571-1cea-4333-b087-76f08de3f485
# ╠═0084b432-9cf1-4f6e-a8dc-8127ec832643
# ╠═c678f383-6fc4-4264-a045-b85decb41528
# ╠═4bd83071-ed45-4eb1-b6a8-667c44c27a7e
# ╠═68dc3cd6-7b79-44fa-9116-6470b7b0b485
# ╠═edae2657-66a9-4976-abe5-d8576dd6eab6
# ╠═ecb0c307-2f8d-4fb7-8688-d9f72cc9e992
# ╠═88baa5c9-3de6-47a9-941d-57c8d6b2ef3d
# ╠═6bf26e05-1d50-4516-9432-deaa0a00e4bf
# ╠═62095eb0-aa7d-4c55-b8a6-34e7d344461a

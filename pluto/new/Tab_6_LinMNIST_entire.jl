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

# ╔═╡ a8d8409d-72cb-4e42-832f-0cc142aa021f
md"""
# XXXX 2025 -- Table 6
This pluto notebook reproduces the numerical experiment
for Table 6 from

- Matthias Beckmann, Robert Beinert, Jonas Bresch, 
  'Normalized Radon Cummulative Distribution Transforms for Invariance and Robustness in Optimal Transport Based Image Classification',
  XXXX 2025.
"""

# ╔═╡ 0380ed85-ee40-47f8-9a12-689a0ad857f2
md"""
## Dataset
Load MNIST 
using `MLDatasets`.
Further information about MNIST can be found in

- L. Deng, 
  '[The MNIST database of handwritten digit images 
  for machine learning research]
  (https://doi.org/10.1109/MSP.2012.2211477)',
  *IEEE Signal Processing Magazine* **29**(6),
  141--142 (2012). 

!!! warning "Download MNIST"
	In order to load MNIST
	using `MLDatasets`,
	MNIST has to be downloaded first!
	For this,
	one can activate the enivironment of the project
	and load the dataset one time explicitly.
	Starting a Julia REPL 
	in the main directory of the project,
	this can be done by
	```
	import Pkg
	Pkg.activate()
	using MLDatasets
	MNIST(:train)
	```
	Julia then asks to download the corresponding dataset.
"""

# ╔═╡ 81783bfb-d7a2-4c18-a4f8-b634f3bbc59b
trainset = MNIST(:train)

# ╔═╡ 98131234-5ab0-4954-bd68-0646241ed22a
md"""
Generate LinMNIST
by selecting a subset of samples
and applying random affine transformations
using the submodule `DataTransformations`.
Further information about LinMNIST can be found in

- M. Beckmann, N. Heilenkötter,
  '[Equivariant neural networks 
  for indirect measurements]
  (https://doi.org/10.1137/23M1582862)',
  *SIAM Journal on Mathematics of Data Science* **6**(3),
  579--601 (2024).
"""

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

# ╔═╡ eff42224-7bdb-4d52-be3c-0fe19911cf2f
md"
# Setting the RCDT 
with 300 radii, 128 Radon angles, and 64 interpolation points.
The max- and mean-normalized RCDT are applied on the generated dataset.
"

# ╔═╡ 8fb1f5c3-386e-4117-9b87-dedb75c1ae1d
R = RadonTransform(300,128,0.0)

# ╔═╡ bbbcd04c-8b4f-4c44-958d-9e4089ada051
RCDT = RadonCDT(64, R)

# ╔═╡ 81fd55d8-24df-4047-b235-20468b2c111c
#=╠═╡
NRCDT = NormRadonCDT(RCDT)
  ╠═╡ =#

# ╔═╡ a4860242-45bb-4f48-816c-3ed1bc82a9c6
rcdt = RCDT.(TMLClass);

# ╔═╡ 81e32395-78d9-4a5f-b6f0-ba2d6f01c8ee
#=╠═╡
mqClass = max_normalization.(rcdt)
  ╠═╡ =#

# ╔═╡ 13849a92-c3d3-42e9-a959-ca456452aaeb
#=╠═╡
aqClass = mean_normalization.(rcdt)
  ╠═╡ =#

# ╔═╡ c0499ac9-97b0-4ec9-bc0c-a6031b7fa223
md"""
## Nearest Neighbour Classification -- Table 2
Use the nearest neighbour classification
with respect to 11, 25, and 50 randomly chosen trainings samples 
from the transformed dataset
per class
to classify the generated dataset.
The max- and mean-normalized RCDT is applied.
Each experiment is repeated twenty times.
"""

# ╔═╡ 68dc3cd6-7b79-44fa-9116-6470b7b0b485
# ╠═╡ disabled = true
#=╠═╡
for angle in [1,2,3,4,5,6,128,256]
	R = RadonTransform(300,angle,0.0);
	RCDT = RadonCDT(64, R);
	NRCDT = NormRadonCDT(RCDT);
	#mNRCDT = MaxNormRadonCDT(RCDT);
	#aNRCDT = MeanNormRadonCDT(RCDT);
	rcdt = RCDT.(TMLClass)
	mqTemp = max_normalization.(TMLClass);
	aqTemp = mean_normalization.(TMLClass);
	@info "number of equispaced angles:" angle
	accuracy_k_nearest_part_neighbour(20, 10, 100, 3, mqTemp, MLLabel, "inf", K=1);
	accuracy_k_nearest_part_neighbour(20, 10, 100, 3, mqTemp, MLLabel, "euclidean", K=1);
	accuracy_k_nearest_part_neighbour(20, 10, 100, 3, aqTemp, MLLabel, "inf", K=1);
	accuracy_k_nearest_part_neighbour(20, 10, 100, 3, aqTemp, MLLabel, "euclidean", K=1);
end
  ╠═╡ =#

# ╔═╡ 570bf90e-0258-470c-90ef-a0ae53144a5f
md"
- using the RCDT, max- and mean-normalized RCDT embedding.
"

# ╔═╡ edae2657-66a9-4976-abe5-d8576dd6eab6
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ 5a7de93e-774d-4bcc-a488-dde5012e1a3f
md"
- using the Euclidean embedding.
"

# ╔═╡ ecb0c307-2f8d-4fb7-8688-d9f72cc9e992
#=╠═╡
for prop in [11,25,50]
	for KK in [1,5,11]
		@info "split" prop, "k-NN" KK
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 500, 10, Array{Float64}.(TMLClass), MLLabel, "inf", K=KK);
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 500, 10, Array{Float64}.(TMLClass), MLLabel, "euclidean", K=KK);
	end
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─a8d8409d-72cb-4e42-832f-0cc142aa021f
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╠═0380ed85-ee40-47f8-9a12-689a0ad857f2
# ╠═81783bfb-d7a2-4c18-a4f8-b634f3bbc59b
# ╟─98131234-5ab0-4954-bd68-0646241ed22a
# ╠═81170d86-6140-41ce-a1e4-24e70c0530ff
# ╠═bf8448db-7cb1-42ba-9f1e-03b775b31cb8
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═fb3629dc-1860-4a96-a75e-2b4402f847fe
# ╟─eff42224-7bdb-4d52-be3c-0fe19911cf2f
# ╠═8fb1f5c3-386e-4117-9b87-dedb75c1ae1d
# ╠═bbbcd04c-8b4f-4c44-958d-9e4089ada051
# ╠═81fd55d8-24df-4047-b235-20468b2c111c
# ╠═a4860242-45bb-4f48-816c-3ed1bc82a9c6
# ╠═81e32395-78d9-4a5f-b6f0-ba2d6f01c8ee
# ╠═13849a92-c3d3-42e9-a959-ca456452aaeb
# ╟─c0499ac9-97b0-4ec9-bc0c-a6031b7fa223
# ╠═68dc3cd6-7b79-44fa-9116-6470b7b0b485
# ╟─570bf90e-0258-470c-90ef-a0ae53144a5f
# ╠═edae2657-66a9-4976-abe5-d8576dd6eab6
# ╟─5a7de93e-774d-4bcc-a488-dde5012e1a3f
# ╠═ecb0c307-2f8d-4fb7-8688-d9f72cc9e992

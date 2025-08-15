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
	using Random
	Random.seed!(42)
end

# ╔═╡ da709924-bbbc-4233-80f8-03b4c2a22376
md"""
# SSVM 2025 -- Table 2 (right)
This pluto notebook reproduces the numerical experiment
for Table 2 (right) from

- Matthias Beckmann, Robert Beinert, Jonas Bresch, 
  'Max-Normalized Radon Cumulative Distribution
  Transform for Limited Data Classification',
  SSVM 2025.
"""

# ╔═╡ 92f547a1-1f91-4e31-a746-bb81969a728f
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

# ╔═╡ 9876c55a-c5eb-4902-bf59-92f242a393a6
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
MLClass, MLLabel = generate_ml_classes(trainset, [1, 7], 500);

# ╔═╡ bf8448db-7cb1-42ba-9f1e-03b775b31cb8
MLClass

# ╔═╡ 773832af-9099-4dcf-bd1b-c82baaa83424
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.75, 1.25), 
	scale_y = (0.75, 1.25),
	rotate=(-45.0, 45.0), 
	shear_x=(-10.0, 10.0),
	shear_y=(-10.0, 10.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))

# ╔═╡ fb3629dc-1860-4a96-a75e-2b4402f847fe
TMLClass = A.(MLClass)

# ╔═╡ 4a677cc5-ccdb-4d5e-9ba6-fdd925184dc9
# ╠═╡ disabled = true
#=╠═╡
R = RadonTransform(851,128,0.0)
  ╠═╡ =#

# ╔═╡ 1374b3cf-0e14-41e9-a2be-901aae16111b
# ╠═╡ disabled = true
#=╠═╡
RCDT = RadonCDT(64, R)
  ╠═╡ =#

# ╔═╡ b12b8ef1-0678-459f-9479-c21969989d42
# ╠═╡ disabled = true
#=╠═╡
qClass = RCDT.(TMLClass)
  ╠═╡ =#

# ╔═╡ 070cc31b-07c9-4120-86b9-24638c3c139a
# ╠═╡ disabled = true
#=╠═╡
mqClass = max_normalization.(qClass)
  ╠═╡ =#

# ╔═╡ 616228c8-1bd3-48a0-8328-7f6ca513f504
# ╠═╡ disabled = true
#=╠═╡
miqClass = maxmin_normalization.(qClass)
  ╠═╡ =#

# ╔═╡ dd0a47b4-9c14-4a5d-9899-8c8001a59b2a
# ╠═╡ disabled = true
#=╠═╡
tvqClass = tv_normalization.(qClass)
  ╠═╡ =#

# ╔═╡ 8b0121f4-9506-4af8-89ee-2fd2932c24a2
md"""
## Nearest Neighbour Classification -- Table 2
Use the nearest neighbour classification
with respect to five randomly chosen representatives
per class
to classify the generated dataset.
The max-normalized RCDT is applied
with different numbers of used angles.
Each experiment is repeated ten times.
"""

# ╔═╡ 348b704c-c1b5-456b-ad6b-d19a5057e84b
# ╠═╡ disabled = true
#=╠═╡
for angle in [2,4,8,16,32,64,128,256]
	R = RadonTransform(851,angle,0.0)
	RCDT = RadonCDT(64, R)
	qClass = RCDT.(TMLClass)
	mqClass = max_normalization.(qClass)
	miqClass = maxmin_normalization.(qClass)
	tvqClass = tv_normalization.(qClass)
	@info "number of equispaced angles:" angle
	accuracy_nearest_cross_neighbour(mqClass, MLLabel, "inf")
	accuracy_nearest_cross_neighbour(mqClass, MLLabel, "euclidean")
	accuracy_nearest_cross_neighbour(miqClass, MLLabel, "inf")
	accuracy_nearest_cross_neighbour(miqClass, MLLabel, "euclidean")
	accuracy_nearest_cross_neighbour(tvqClass, MLLabel, "inf")
	accuracy_nearest_cross_neighbour(tvqClass, MLLabel, "euclidean")
end
  ╠═╡ =#

# ╔═╡ 5f3194f7-b75c-4562-b475-2e724fccf2b4
for prop in [11,25,50]
	for angle in [2,4,8,16]
		R = RadonTransform(301,angle,0.0)
		RCDT = RadonCDT(64, R)
		qClass = RCDT.(TMLClass)
		mqClass = max_normalization.(qClass)
		miqClass = maxmin_normalization.(qClass)
		tvqClass = tv_normalization.(qClass)
		maqClass = maxabs_normalization.(qClass)
		iaqClass = minabs_normalization.(qClass)
		miaqClass = maxminabs_normalization.(qClass)
		@info "number of equispaced angles and split:" angle, prop
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, mqClass, MLLabel)
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, miqClass, MLLabel)
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, maqClass, MLLabel)
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, iaqClass, MLLabel)
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, miaqClass, MLLabel)
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, tvqClass, MLLabel)
	end
end

# ╔═╡ b09878e1-5ca1-43a4-8a82-358b5075dac6
for angle in [32,64,128,256]
	R = RadonTransform(301,angle,0.0)
	RCDT = RadonCDT(64, R)
	qClass = RCDT.(TMLClass)
	mqClass = max_normalization.(qClass)
	miqClass = maxmin_normalization.(qClass)
	tvqClass = tv_normalization.(qClass)
	maqClass = maxabs_normalization.(qClass)
	iaqClass = minabs_normalization.(qClass)
	miaqClass = maxminabs_normalization.(qClass)
	for prop in [11,25,50]
		@info "number of equispaced angles and split:" angle, prop
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, mqClass, MLLabel)
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, miqClass, MLLabel)
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, maqClass, MLLabel)
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, iaqClass, MLLabel)
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, miaqClass, MLLabel)
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, tvqClass, MLLabel)
	end
end

# ╔═╡ f9e1d846-611f-4c3c-9073-11777d660093
for prop in [11,25,50]
	for angle in [2,4,8,16,32,64,128,256]
		R = RadonTransform(301,angle,0.0)
		RCDT = RadonCDT(64, R)
		qClass = RCDT.(TMLClass)
		@info "number of equispaced angles and split:" angle, prop
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, qClass, MLLabel)
	end
	Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, Array{Float64}.(TMLClass), MLLabel)
end

# ╔═╡ Cell order:
# ╟─da709924-bbbc-4233-80f8-03b4c2a22376
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╟─92f547a1-1f91-4e31-a746-bb81969a728f
# ╠═81783bfb-d7a2-4c18-a4f8-b634f3bbc59b
# ╟─9876c55a-c5eb-4902-bf59-92f242a393a6
# ╠═81170d86-6140-41ce-a1e4-24e70c0530ff
# ╠═bf8448db-7cb1-42ba-9f1e-03b775b31cb8
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═fb3629dc-1860-4a96-a75e-2b4402f847fe
# ╠═4a677cc5-ccdb-4d5e-9ba6-fdd925184dc9
# ╠═1374b3cf-0e14-41e9-a2be-901aae16111b
# ╠═b12b8ef1-0678-459f-9479-c21969989d42
# ╠═070cc31b-07c9-4120-86b9-24638c3c139a
# ╠═616228c8-1bd3-48a0-8328-7f6ca513f504
# ╠═dd0a47b4-9c14-4a5d-9899-8c8001a59b2a
# ╟─8b0121f4-9506-4af8-89ee-2fd2932c24a2
# ╠═348b704c-c1b5-456b-ad6b-d19a5057e84b
# ╠═5f3194f7-b75c-4562-b475-2e724fccf2b4
# ╠═b09878e1-5ca1-43a4-8a82-358b5075dac6
# ╠═f9e1d846-611f-4c3c-9073-11777d660093

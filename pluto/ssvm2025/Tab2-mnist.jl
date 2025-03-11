### A Pluto.jl notebook ###
# v0.20.4

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
	Pkg.activate
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
MLClass, MLLabel = generate_ml_classes(trainset, [1, 5, 7], 10);

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
for angle in [2,4,8,16,32,64,128]
	R = RadonTransform(floor(Int,sqrt(2)*256),angle,0.0)
	RCDT = RadonCDT(floor(Int,sqrt(2)*256), R)
	mNRCDT = MaxNormRadonCDT(RCDT)
	qClass = mNRCDT.(TMLClass)
	@info "number of equispaced angles:" angle
	accuracy_nearest_cross_neighbour(qClass, MLLabel, "inf")
	accuracy_nearest_cross_neighbour(qClass, MLLabel, "euclidean")
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
# ╟─8b0121f4-9506-4af8-89ee-2fd2932c24a2
# ╠═348b704c-c1b5-456b-ad6b-d19a5057e84b

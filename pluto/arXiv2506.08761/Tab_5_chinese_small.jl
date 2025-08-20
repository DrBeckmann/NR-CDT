### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 19587268-0828-11f0-01fa-e979f61f03a3
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
	using JLD2
	using Random
	Random.seed!(42)
end

# ╔═╡ 24bfee91-efc3-437d-ae96-ed3448b6fab6
md"""
# XXXX 2025 -- Table 5 (upper left part)
This pluto notebook reproduces the numerical experiment
for Table 5 (upper left part) from

- Matthias Beckmann, Robert Beinert, Jonas Bresch, 
  'Normalized Radon Cummulative Distribution Transforms for Invariance and Robustness in Optimal Transport Based Image Classification',
  XXXX 2025.
"""

# ╔═╡ 69f91584-2745-43b2-8283-9221ba2b3ec4
md"""
## Dataset
Load a selection of the Chinese hand-written character dataset `CASIA Chinese Handwriting Databases`.
For Further information we refer to '[Handwritten chinese character hanzi datasets] (https://nlpr.ia.ac.cn/databases/handwriting/Home.html)'. 

- C.-L. Liu, F. Yin, D.-H. Wang, Q.-F. Wang, CASIA online and offline Chinese handwriting databases, *Proc. 11th International Conference on Document Analysis and Recognition (ICDAR)*, Beijing, China, 2011, pp.37-41.
"""

# ╔═╡ 431b7d62-cc7a-4d2e-a21a-4c0ec5b806e6
begin
	dict = JLD2.load("../../data/CASIA_selection/chinese_character.jld2");
	chinese_character = dict["chinese_character"];
	for k in 1:length(chinese_character)
		chinese_character[k] = 1 .- chinese_character[k]
	end
	ext_chinese_character = extend_image.(chinese_character, 128)
end

# ╔═╡ 46e9a3b9-48dd-457a-890f-74ac08364b3a
md"""
## Templates
Generate the 100 templates
using the submodule `TestImages`.
"""

# ╔═╡ 28cbf21b-37a0-4307-bad7-7ff6d1efd511
Class, Labels = generate_academic_classes(ext_chinese_character[1:100], 1:100, class_size=50);

# ╔═╡ fc1f2b19-fe55-4421-a904-398a2448597b
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.5, 1.0), 
	scale_y = (0.5, 1.0),
	rotate=(-180.0, 180.0), 
	shear_x=(-45.0, 45.0),
	shear_y=(-45.0, 45.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))

# ╔═╡ 7e5f32b7-2f27-4877-a6fe-c6b41750aa1b
Random.seed!(42); TClass = A.(Class)

# ╔═╡ 1986d601-d6f6-44ff-bb14-48ed05b5f88e
md"""
## Nearest Neighbour Classification -- Table 5
Use the nearest neighbour classification
with respect to the chosen templates
to classify the generated dataset.
The max- and mean-normalized RCDT is applied
with different numbers of used angles.
"""

# ╔═╡ 058e9e41-37bc-4ef0-b8c8-8b5727dff8d8
accuracy_k_nearest_neighbour(Array{Float64}.(ext_chinese_character), unique(Labels), Array{Float64}.(TClass), Labels, "inf", ret=1)

# ╔═╡ 57b41939-a469-4511-8bef-def1e6f74ce5
accuracy_k_nearest_neighbour(Array{Float64}.(ext_chinese_character), unique(Labels), Array{Float64}.(TClass), Labels, "euclidean", ret=1)

# ╔═╡ cede8b89-ffad-4b36-a234-d563c7a9acca
for angle in [2,4,8,16,32,64,128]
	R = RadonTransform(256,angle,0.0)
	RCDT = RadonCDT(256, R)
	mNRCDT = MaxNormRadonCDT(RCDT)
	aNRCDT = MeanNormRadonCDT(RCDT)
	qClass = RCDT.(TClass)
	qTemp = RCDT.(ext_chinese_character)
	#mqClass = mNRCDT.(TClass)
	mqClass = max_normalization.(qClass)
	#mqTemp = mNRCDT.(ext_chinese_character)
	mqTemp = max_normalization.(qTemp)
	#aqClass = aNRCDT.(TClass)
	aqClass = mean_normalization.(qClass)
	#aqTemp = aNRCDT.(ext_chinese_character)
	aqTemp = mean_normalization.(qTemp)
	@info "number of equispaced angles:" angle
	accuracy_k_nearest_neighbour(qTemp, unique(Labels), qClass, Labels, "inf", ret=1)
	accuracy_k_nearest_neighbour(qTemp, unique(Labels), qClass, Labels, "euclidean", ret=1)
	accuracy_k_nearest_neighbour(mqTemp, unique(Labels), mqClass, Labels, "inf", ret=1)
	accuracy_k_nearest_neighbour(mqTemp, unique(Labels), mqClass, Labels, "euclidean", ret=1)
	accuracy_k_nearest_neighbour(aqTemp, unique(Labels), aqClass, Labels, "inf", ret=1)
	accuracy_k_nearest_neighbour(aqTemp, unique(Labels), aqClass, Labels, "euclidean", ret=1)
end

# ╔═╡ Cell order:
# ╟─24bfee91-efc3-437d-ae96-ed3448b6fab6
# ╠═19587268-0828-11f0-01fa-e979f61f03a3
# ╟─69f91584-2745-43b2-8283-9221ba2b3ec4
# ╠═431b7d62-cc7a-4d2e-a21a-4c0ec5b806e6
# ╟─46e9a3b9-48dd-457a-890f-74ac08364b3a
# ╠═28cbf21b-37a0-4307-bad7-7ff6d1efd511
# ╠═fc1f2b19-fe55-4421-a904-398a2448597b
# ╠═7e5f32b7-2f27-4877-a6fe-c6b41750aa1b
# ╟─1986d601-d6f6-44ff-bb14-48ed05b5f88e
# ╠═058e9e41-37bc-4ef0-b8c8-8b5727dff8d8
# ╠═57b41939-a469-4511-8bef-def1e6f74ce5
# ╠═cede8b89-ffad-4b36-a234-d563c7a9acca

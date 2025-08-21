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

# ╔═╡ a8ce5c20-23c2-43a0-875a-e93bd61b8e23
md"""
# arXiv:2506.08761 -- Table 5 (lower left block)
This Pluto notebook reproduces the numerical experiment
for Table 5 (lower left block) from

- Matthias Beckmann, Robert Beinert, Jonas Bresch, 
  'Normalized Radon Cummulative Distribution Transforms for Invariance and Robustness in Optimal Transport Based Image Classification',
  arXiv:2506.08761, 2025.
"""

# ╔═╡ 6b753037-4286-407d-823b-7ff08f7aa505
md"""
## Dataset
Load a selection of the Chinese hand-written character dataset `CASIA Chinese Handwriting Databases`.
For Further information we refer to '[Handwritten chinese character hanzi datasets] (https://nlpr.ia.ac.cn/databases/handwriting/Home.html)'. 

- C.-L. Liu, F. Yin, D.-H. Wang, Q.-F. Wang, CASIA online and offline Chinese handwriting databases, *Proc. 11th International Conference on Document Analysis and Recognition (ICDAR)*, Beijing, China, 2011, pp.37-41.
"""

# ╔═╡ bdb19381-b1c7-4496-911c-5d6de31f63d9
begin
	dict = JLD2.load("../../data/CASIA_selection/chinese_character.jld2");
	chinese_character = dict["chinese_character"];
	for k in 1:length(chinese_character)
		chinese_character[k] = 1 .- chinese_character[k]
	end
	ext_chinese_character = extend_image.(chinese_character, 128)
end

# ╔═╡ 0d5b04f4-d7b2-4133-a3cf-0a8e88fcc955
md"""
## Templates
Generate the 1000 templates
using the submodule `TestImages`.
"""

# ╔═╡ 28cbf21b-37a0-4307-bad7-7ff6d1efd511
Class, Labels = generate_academic_classes(ext_chinese_character[1:1000],1:1000, class_size=50);

# ╔═╡ fc1f2b19-fe55-4421-a904-398a2448597b
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.5, 1.0), 
	scale_y = (0.5, 1.0),
	rotate=(-180.0, 180.0), 
	shear_x=(-25.0, 25.0),
	shear_y=(-25.0, 25.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))

# ╔═╡ 7e5f32b7-2f27-4877-a6fe-c6b41750aa1b
Random.seed!(42); TClass = A.(Class);

# ╔═╡ ed15d805-9d84-4619-8909-304a83842b86
md"""
## Nearest Neighbour Classification -- Table 5
Use the nearest neighbour classification
with respect to the chosen templates
to classify the generated dataset.
The max- and mean-normalized R-CDT is applied
with different numbers of angles.
"""

# ╔═╡ 2fab6713-3367-4696-b326-76d0f81b7429
accuracy_k_nearest_neighbour(Array{Float64}.(ext_chinese_character[1:1000]), unique(Labels), Array{Float64}.(TClass), Labels, "euclidean", ret=1)

# ╔═╡ 1edcfb3c-14fd-4c66-bccd-2e54acd0c544
for angle in [2,4,8,16,32,64,128]
	R = RadonTransform(850,angle,0.0)
	RCDT = RadonCDT(64, R)
	mNRCDT = MaxNormRadonCDT(RCDT)
	aNRCDT = MeanNormRadonCDT(RCDT)
	qClass = RCDT.(TClass)
	qTemp = RCDT.(ext_chinese_character[1:1000])
	mqClass = max_normalization.(qClass)
	mqTemp = max_normalization.(qTemp)
	aqClass = mean_normalization.(qClass)
	aqTemp = mean_normalization.(qTemp)
	@info "number of equispaced angles:" angle
	accuracy_k_nearest_neighbour(qTemp, unique(Labels), qClass, Labels, "euclidean", ret=1)
	accuracy_k_nearest_neighbour(mqTemp, unique(Labels), mqClass, Labels, "euclidean", ret=1)
	accuracy_k_nearest_neighbour(aqTemp, unique(Labels), aqClass, Labels, "euclidean", ret=1)
end

# ╔═╡ Cell order:
# ╟─a8ce5c20-23c2-43a0-875a-e93bd61b8e23
# ╠═19587268-0828-11f0-01fa-e979f61f03a3
# ╟─6b753037-4286-407d-823b-7ff08f7aa505
# ╠═bdb19381-b1c7-4496-911c-5d6de31f63d9
# ╟─0d5b04f4-d7b2-4133-a3cf-0a8e88fcc955
# ╠═28cbf21b-37a0-4307-bad7-7ff6d1efd511
# ╠═fc1f2b19-fe55-4421-a904-398a2448597b
# ╠═7e5f32b7-2f27-4877-a6fe-c6b41750aa1b
# ╟─ed15d805-9d84-4619-8909-304a83842b86
# ╠═2fab6713-3367-4696-b326-76d0f81b7429
# ╠═1edcfb3c-14fd-4c66-bccd-2e54acd0c544

### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 19587268-0828-11f0-01fa-e979f61f03a3
begin
	import Pkg
	Pkg.activate("..")
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

# ╔═╡ 8de4c852-8bdf-4a34-83c3-c5ddd3826ed6
md"""
# arXiv:2506.08761 -- Table 5 (upper right block)
This Pluto notebook reproduces the numerical experiment
for Table 5 (upper right block) from

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

# ╔═╡ 0b7d3e09-9166-45c8-9497-40ed74c45995
begin
	dict = JLD2.load("../../data/CASIA_selection/chinese_character.jld2");
	chinese_character = dict["chinese_character"];
	for k in 1:length(chinese_character)
		chinese_character[k] = 1 .- chinese_character[k]
	end
	ext_chinese_character = extend_image.(chinese_character, 128)
end

# ╔═╡ e4ebd3f1-1e9f-4bd0-8beb-eb8e2ff1c2db
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
	shear_x=(-25.0, 25.0),
	shear_y=(-25.0, 25.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))

# ╔═╡ 7e5f32b7-2f27-4877-a6fe-c6b41750aa1b
Random.seed!(42); TClass = A.(Class);

# ╔═╡ 0720d726-4b3d-490d-bb0d-c3264fca1ff9
md"""
## Setting the R-CDT
with 850 radii, 128 Radon angles, and 64 interpolation points. 
The max- and mean-normalized R-CDT are applied
on the entire dataset.
"""

# ╔═╡ 17b1028a-f314-43b2-9965-5a023467c31e
R = RadonTransform(850,128,0.0)

# ╔═╡ 0bbf22d4-61c5-4915-ac02-4f64ad328732
RCDT = RadonCDT(64, R)

# ╔═╡ a87c3b6a-11df-42b8-8f97-157b5c5ab763
mNRCDT = MaxNormRadonCDT(RCDT)

# ╔═╡ 8a9bbc83-b783-4ead-844e-9f9e3632d90a
aNRCDT = MeanNormRadonCDT(RCDT)

# ╔═╡ b1e25447-dbec-4eeb-a554-0c6d08b71f04
 qClass = RCDT.(TClass)

# ╔═╡ 745baebd-b48a-4468-aea5-5e9d5806a25e
 mqClass = max_normalization.(qClass)		#mqClass = mNRCDT.(TClass)

# ╔═╡ c065302a-21d8-4924-9f6d-e118b141bb3c
 aqClass = mean_normalization.(qClass)	#aqClass = aNRCDT.(TClass)

# ╔═╡ 7b4016cb-ba6f-45c4-a647-3dbc718ace0a
md"""
## Nearest Neighbour Classification -- Table 5
Use the nearest neighbour classification
with respect to the 5 and 10 randomly chosen training samples
from the transformed dataset
to classify the generated dataset.
Each experiment is repeated twenty times.
"""

# ╔═╡ e552a3d5-ec73-432a-8f50-3f916da127d3
md"
- using the max- and mean-normalized R-CDT embedding.
"

# ╔═╡ f05629b6-34b6-46f4-8991-d64557322202
for prop in [5,10]
	for K in [1,3,5]
		@info "split:" prop, "K-NN:" K 
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 100, mqClass, Labels, "inf", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 100, mqClass, Labels, "euclidean", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 100, aqClass, Labels, "inf", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 100, aqClass, Labels, "euclidean", K=K, ret=0)
	end
end

# ╔═╡ b170d534-6719-4b91-a944-021228c11156
md"
- using the Euclidean and R-CDT embedding.
"

# ╔═╡ bcd3da7f-4c5c-41cc-b1dc-4577304aa184
for prop in [5,10]
	for K in [1,3,5]
		@info "split:" prop, "K-NN:" K 
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 100, Array{Float64}.(TClass), Labels, "inf", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 100, Array{Float64}.(TClass), Labels, "euclidean", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 100, qClass, Labels, "inf", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 100, qClass, Labels, "euclidean", K=K, ret=0)
	end
end

# ╔═╡ Cell order:
# ╟─8de4c852-8bdf-4a34-83c3-c5ddd3826ed6
# ╠═19587268-0828-11f0-01fa-e979f61f03a3
# ╟─6b753037-4286-407d-823b-7ff08f7aa505
# ╠═0b7d3e09-9166-45c8-9497-40ed74c45995
# ╟─e4ebd3f1-1e9f-4bd0-8beb-eb8e2ff1c2db
# ╠═28cbf21b-37a0-4307-bad7-7ff6d1efd511
# ╠═fc1f2b19-fe55-4421-a904-398a2448597b
# ╠═7e5f32b7-2f27-4877-a6fe-c6b41750aa1b
# ╟─0720d726-4b3d-490d-bb0d-c3264fca1ff9
# ╠═17b1028a-f314-43b2-9965-5a023467c31e
# ╠═0bbf22d4-61c5-4915-ac02-4f64ad328732
# ╠═a87c3b6a-11df-42b8-8f97-157b5c5ab763
# ╠═8a9bbc83-b783-4ead-844e-9f9e3632d90a
# ╠═b1e25447-dbec-4eeb-a554-0c6d08b71f04
# ╠═745baebd-b48a-4468-aea5-5e9d5806a25e
# ╠═c065302a-21d8-4924-9f6d-e118b141bb3c
# ╟─7b4016cb-ba6f-45c4-a647-3dbc718ace0a
# ╟─e552a3d5-ec73-432a-8f50-3f916da127d3
# ╠═f05629b6-34b6-46f4-8991-d64557322202
# ╟─b170d534-6719-4b91-a944-021228c11156
# ╠═bcd3da7f-4c5c-41cc-b1dc-4577304aa184

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

# ╔═╡ efcdfcff-cef6-48c9-992e-8a2eb9348424
md"""
# arXiv:2506.08761 -- Table 5 (lower right block)
This Pluto notebook reproduces the numerical experiment
for Table 5 (lower right block) from

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

# ╔═╡ c1c19c32-1219-427e-a156-a1b408c3ed9e
begin
	dict = JLD2.load("../../data/CASIA_selection/chinese_character.jld2");
	chinese_character = dict["chinese_character"];
	for k in 1:length(chinese_character)
		chinese_character[k] = 1 .- chinese_character[k]
	end
	ext_chinese_character = extend_image.(chinese_character, 128)
end

# ╔═╡ ff5ba6ba-3657-4642-8db5-c04f9a0454c8
md"""
## Templates
Generate the 1000 templates
using the submodule `TestImages`.
"""

# ╔═╡ 28cbf21b-37a0-4307-bad7-7ff6d1efd511
Class, Labels = generate_academic_classes(ext_chinese_character[1:1000], 1:1000, class_size=50);

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

# ╔═╡ 466edf1c-ef46-4776-840a-b493d8c687b1
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

# ╔═╡ 9ccddf1f-9a7b-4010-bbfa-905fe28320be
qClass = RCDT.(TClass)

# ╔═╡ 745baebd-b48a-4468-aea5-5e9d5806a25e
mqClass = max_normalization.(qClass)

# ╔═╡ c065302a-21d8-4924-9f6d-e118b141bb3c
aqClass = mean_normalization.(qClass)

# ╔═╡ 2554552f-086c-48a3-a968-b921ecbe3bb6
md"""
## Nearest Neighbour Classification -- Table 5
Use the nearest neighbour classification
with respect to the 5 and 10 randomly chosen training samples
from the transformed dataset
to classify the generated dataset.
Each experiment is repeated twenty times.
"""

# ╔═╡ 2155c12d-6cbc-47e5-a243-986e945a8085
md"
- using the max- and mean-normalized R-CDT embedding.
"

# ╔═╡ 0383b820-3203-4188-8bc6-946599a43c81
for prop in [5,10]
	for K in [1] #[1,3,5]
		@info "split:" prop, "K-NN:" K 
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 1000, mqClass, Labels, "euclidean", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 1000, aqClass, Labels, "euclidean", K=K, ret=0)
	end
end

# ╔═╡ eca6c85b-964d-4ff7-9fc2-e3192f19b440
md"
- using the Euclidean and RCDT embedding.
"

# ╔═╡ 4d819a08-a727-4b6c-b373-c6917d113fd7
for prop in [5,10]
	for K in [1] #[1,3,5]
		@info "split:" prop, "K-NN:" K 
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 1000, Array{Float64}.(TClass), Labels, "euclidean", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 1000, qClass, Labels, "euclidean", K=K, ret=0)
	end
end

# ╔═╡ Cell order:
# ╟─efcdfcff-cef6-48c9-992e-8a2eb9348424
# ╠═19587268-0828-11f0-01fa-e979f61f03a3
# ╟─6b753037-4286-407d-823b-7ff08f7aa505
# ╠═c1c19c32-1219-427e-a156-a1b408c3ed9e
# ╟─ff5ba6ba-3657-4642-8db5-c04f9a0454c8
# ╠═28cbf21b-37a0-4307-bad7-7ff6d1efd511
# ╠═fc1f2b19-fe55-4421-a904-398a2448597b
# ╠═7e5f32b7-2f27-4877-a6fe-c6b41750aa1b
# ╟─466edf1c-ef46-4776-840a-b493d8c687b1
# ╠═17b1028a-f314-43b2-9965-5a023467c31e
# ╠═0bbf22d4-61c5-4915-ac02-4f64ad328732
# ╠═a87c3b6a-11df-42b8-8f97-157b5c5ab763
# ╠═8a9bbc83-b783-4ead-844e-9f9e3632d90a
# ╠═9ccddf1f-9a7b-4010-bbfa-905fe28320be
# ╠═745baebd-b48a-4468-aea5-5e9d5806a25e
# ╠═c065302a-21d8-4924-9f6d-e118b141bb3c
# ╟─2554552f-086c-48a3-a968-b921ecbe3bb6
# ╟─2155c12d-6cbc-47e5-a243-986e945a8085
# ╠═0383b820-3203-4188-8bc6-946599a43c81
# ╟─eca6c85b-964d-4ff7-9fc2-e3192f19b440
# ╠═4d819a08-a727-4b6c-b373-c6917d113fd7

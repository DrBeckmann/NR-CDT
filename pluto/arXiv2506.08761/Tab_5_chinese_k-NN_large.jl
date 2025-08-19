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
	using Glob
	using Random
	Random.seed!(42)
end

# ╔═╡ efcdfcff-cef6-48c9-992e-8a2eb9348424
md"""
# XXXX 2025 -- Table 5 (lower right part)
This pluto notebook reproduces the numerical experiment
for Table 5 (lower right part) from

- Matthias Beckmann, Robert Beinert, Jonas Bresch, 
  'Normalized Radon Cummulative Distribution Transforms for Invariance and Robustness in Optimal Transport Based Image Classification',
  XXXX 2025.
"""

# ╔═╡ 6b753037-4286-407d-823b-7ff08f7aa505
md"""
## Dataset
Load the Chinese hand-written character dataset `handwritten-chinese-character-hanzi-datasets`.
Further information can be found in

- P. Bliem, '[Handwritten chinese character hanzi datasets] (https://www.kaggle.com/datasets/pascalbliem/handwritten-chinese-character-hanzi-datasets)', (2022), Accessed: March 26, 2025.  

!!! warning "Download the Chinese handwritten character dataset"
	In order to load the dataset
	using `kaggle`,
	it has to be loaded first!
	For this,
	one can activate the enivironment of the project
	and load the dataset one time explicitly.
	Starting a Julia REPL 
	in the main directory of the project,
	this can be done by
	```
	import kaggle
	# This will use your credentials from ~/.kaggle/kaggle.json
	from kaggle.api.kaggle_api_extended import KaggleApi
	api = KaggleApi()
	api.authenticate()
	api.dataset_download_files('pascalbliem/handwritten-chinese-character-hanzi-datasets', path='./data', unzip=True)
	```
	Julia then asks to download the corresponding dataset.
"""

# ╔═╡ 1911d0f8-219a-40ba-a2f6-ba353e14eef4
md"
- navigate the dataset to the currect path, i.e. the folder in which the notbook runs.
- the following steps collect from each of the first 1000 classes the first handwritten symbol 
"

# ╔═╡ 2450a95a-b230-4b73-b369-3aa36d3172b5
root_folder = "chinese_character";

# ╔═╡ 782377c4-21bf-4f9f-a8ba-4b3bf849c696
subfolders = filter(name -> isdir(joinpath(root_folder, name)), readdir(root_folder));

# ╔═╡ 639e3aed-dde6-491c-a4a3-8be172766755
chinese_character = [];

# ╔═╡ dbe82aad-857e-45db-b56f-35e0acffde0c
for sub in subfolders[1:1000]
    subpath = joinpath(root_folder, sub)
    # Get all .png files in the subfolder
     image_files = Glob.glob("*.png", subpath)
    
    if !isempty(image_files)
        # Load the first image from the subfolder
        first_image = image_files[1]
        # println("Loading first image from subfolder ", sub, ": ", first_image)
        img = load(first_image)
        # Display the image (or process as needed)
        # display(img)
		push!(chinese_character, 1 .- img)
    else
        println("No image found in subfolder ", sub)
    end
end

# ╔═╡ ff5ba6ba-3657-4642-8db5-c04f9a0454c8
md"""
## Templates
Generate the 1000 templates
using the submodule `TestImages`.
"""

# ╔═╡ 8f5dfb6e-9c0d-4187-9fd1-1c0acccc4ac1
ext_chinese_character = extend_image.(chinese_character, 128)

# ╔═╡ 28cbf21b-37a0-4307-bad7-7ff6d1efd511
Class, Labels = generate_academic_classes(ext_chinese_character, 1:1000, class_size=50);

# ╔═╡ fc1f2b19-fe55-4421-a904-398a2448597b
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.75, 1.25), 
	scale_y = (0.75, 1.25),
	rotate=(-45.0, 45.0), 
	shear_x=(-5.0, 5.0),
	shear_y=(-5.0, 5.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))

# ╔═╡ 7e5f32b7-2f27-4877-a6fe-c6b41750aa1b
Random.seed!(42); TClass = A.(Class);

# ╔═╡ 466edf1c-ef46-4776-840a-b493d8c687b1
md"""
## Setting the Radon CDT
with 850 radii, 128 Radon angles, and 64 interpolation points. 
The max- and mean-normalized RCDT are applied
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
mqClass = max_normalization.(qClass)		#mqClass = mNRCDT.(TClass)

# ╔═╡ c065302a-21d8-4924-9f6d-e118b141bb3c
aqClass = mean_normalization.(qClass)		#aqClass = aNRCDT.(TClass)

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
- using the max- and mean-normalized RCDT embedding.
"

# ╔═╡ 0383b820-3203-4188-8bc6-946599a43c81
for prop in [5,10]
	for K in [1,3,5]
		@info "split:" prop, "K-NN:" K 
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 1000, mqClass, Labels, "inf", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 1000, mqClass, Labels, "euclidean", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 1000, aqClass, Labels, "inf", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 1000, aqClass, Labels, "euclidean", K=K, ret=0)
	end
end

# ╔═╡ eca6c85b-964d-4ff7-9fc2-e3192f19b440
md"
- using the Euclidean and RCDT embedding.
"

# ╔═╡ 4d819a08-a727-4b6c-b373-c6917d113fd7
for prop in [5,10]
	for K in [1,3,5]
		@info "split:" prop, "K-NN:" K 
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 1000, Array{Float64}.(TClass), Labels, "inf", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 1000, Array{Float64}.(TClass), Labels, "euclidean", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 1000, qClass, Labels, "inf", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 1000, qClass, Labels, "euclidean", K=K, ret=0)
	end
end

# ╔═╡ Cell order:
# ╟─efcdfcff-cef6-48c9-992e-8a2eb9348424
# ╠═19587268-0828-11f0-01fa-e979f61f03a3
# ╟─6b753037-4286-407d-823b-7ff08f7aa505
# ╟─1911d0f8-219a-40ba-a2f6-ba353e14eef4
# ╠═2450a95a-b230-4b73-b369-3aa36d3172b5
# ╠═782377c4-21bf-4f9f-a8ba-4b3bf849c696
# ╠═639e3aed-dde6-491c-a4a3-8be172766755
# ╠═dbe82aad-857e-45db-b56f-35e0acffde0c
# ╟─ff5ba6ba-3657-4642-8db5-c04f9a0454c8
# ╠═8f5dfb6e-9c0d-4187-9fd1-1c0acccc4ac1
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

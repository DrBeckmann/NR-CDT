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

# ╔═╡ db221814-64e6-46c0-a288-b435876a4ed2
md"
- navigate the dataset to the currect path, i.e. the folder in which the notbook runs.
- the following steps collect from each of the first 100 classes the first handwritten symbol 
"

# ╔═╡ 2450a95a-b230-4b73-b369-3aa36d3172b5
root_folder = "chinese_character";

# ╔═╡ 782377c4-21bf-4f9f-a8ba-4b3bf849c696
subfolders = filter(name -> isdir(joinpath(root_folder, name)), readdir(root_folder));

# ╔═╡ 639e3aed-dde6-491c-a4a3-8be172766755
chinese_character = [];

# ╔═╡ dbe82aad-857e-45db-b56f-35e0acffde0c
for sub in subfolders[1:100]
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

# ╔═╡ e4ebd3f1-1e9f-4bd0-8beb-eb8e2ff1c2db
md"""
## Templates
Generate the 100 templates
using the submodule `TestImages`.
"""

# ╔═╡ 8f5dfb6e-9c0d-4187-9fd1-1c0acccc4ac1
ext_chinese_character = extend_image.(chinese_character, 128)

# ╔═╡ 28cbf21b-37a0-4307-bad7-7ff6d1efd511
Class, Labels = generate_academic_classes(ext_chinese_character, 1:100, class_size=50);

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
# ╟─db221814-64e6-46c0-a288-b435876a4ed2
# ╠═2450a95a-b230-4b73-b369-3aa36d3172b5
# ╠═782377c4-21bf-4f9f-a8ba-4b3bf849c696
# ╠═639e3aed-dde6-491c-a4a3-8be172766755
# ╠═dbe82aad-857e-45db-b56f-35e0acffde0c
# ╟─e4ebd3f1-1e9f-4bd0-8beb-eb8e2ff1c2db
# ╠═8f5dfb6e-9c0d-4187-9fd1-1c0acccc4ac1
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

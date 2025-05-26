### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 19587268-0828-11f0-01fa-e979f61f03a3
begin
	import Pkg
	Pkg.activate(".")
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

# ╔═╡ a8ce5c20-23c2-43a0-875a-e93bd61b8e23
md"""
# XXXX 2025 -- Table 5 (lower left part)
This pluto notebook reproduces the numerical experiment
for Table 5 (lower left part) from

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

# ╔═╡ a969acb4-7fc1-4977-85f5-e8218b356458
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

# ╔═╡ 0d5b04f4-d7b2-4133-a3cf-0a8e88fcc955
md"""
## Templates
Generate the 1000 templates
using the submodule `TestImages`.
"""

# ╔═╡ 8f5dfb6e-9c0d-4187-9fd1-1c0acccc4ac1
ext_chinese_character = extend_image.(chinese_character, 128)

# ╔═╡ 28cbf21b-37a0-4307-bad7-7ff6d1efd511
Class, Labels = generate_academic_classes(ext_chinese_character, class_size=50);

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

# ╔═╡ ed15d805-9d84-4619-8909-304a83842b86
md"""
## Nearest Neighbour Classification -- Table 5
Use the nearest neighbour classification
with respect to the chosen templates
to classify the generated dataset.
The max- and mean-normalized RCDT is applied
with different numbers of used angles.
"""

# ╔═╡ 0a193211-86e6-46cf-a265-e87ff8d00fc6
accuracy_k_nearest_neighbour(Array{Float64}.(ext_chinese_character), unique(Labels), Array{Float64}.(TClass), Labels, "inf", ret=1)

# ╔═╡ 2fab6713-3367-4696-b326-76d0f81b7429
accuracy_k_nearest_neighbour(Array{Float64}.(ext_chinese_character), unique(Labels), Array{Float64}.(TClass), Labels, "euclidean", ret=1)

# ╔═╡ 1edcfb3c-14fd-4c66-bccd-2e54acd0c544
for angle in [2,4,8,16,32,64,128]
	R = RadonTransform(128,angle,0.0)
	RCDT = RadonCDT(128, R)
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
# ╟─a8ce5c20-23c2-43a0-875a-e93bd61b8e23
# ╠═19587268-0828-11f0-01fa-e979f61f03a3
# ╟─6b753037-4286-407d-823b-7ff08f7aa505
# ╟─a969acb4-7fc1-4977-85f5-e8218b356458
# ╠═2450a95a-b230-4b73-b369-3aa36d3172b5
# ╠═782377c4-21bf-4f9f-a8ba-4b3bf849c696
# ╠═639e3aed-dde6-491c-a4a3-8be172766755
# ╠═dbe82aad-857e-45db-b56f-35e0acffde0c
# ╟─0d5b04f4-d7b2-4133-a3cf-0a8e88fcc955
# ╠═8f5dfb6e-9c0d-4187-9fd1-1c0acccc4ac1
# ╠═28cbf21b-37a0-4307-bad7-7ff6d1efd511
# ╠═fc1f2b19-fe55-4421-a904-398a2448597b
# ╠═7e5f32b7-2f27-4877-a6fe-c6b41750aa1b
# ╟─ed15d805-9d84-4619-8909-304a83842b86
# ╠═0a193211-86e6-46cf-a265-e87ff8d00fc6
# ╠═2fab6713-3367-4696-b326-76d0f81b7429
# ╠═1edcfb3c-14fd-4c66-bccd-2e54acd0c544

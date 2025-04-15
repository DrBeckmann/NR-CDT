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
	using Random
	Random.seed!(42)
end

# ╔═╡ 9d398d31-2bc5-449b-b750-6a88c1ce6767
using Glob

# ╔═╡ 6b753037-4286-407d-823b-7ff08f7aa505
"import kaggle

# This will use your credentials from ~/.kaggle/kaggle.json
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.dataset_download_files('pascalbliem/handwritten-chinese-character-hanzi-datasets', path='./data', unzip=True)"

# ╔═╡ 2450a95a-b230-4b73-b369-3aa36d3172b5
root_folder = "chinese_character"

# ╔═╡ 782377c4-21bf-4f9f-a8ba-4b3bf849c696
subfolders = filter(name -> isdir(joinpath(root_folder, name)), readdir(root_folder))

# ╔═╡ 639e3aed-dde6-491c-a4a3-8be172766755
chinese_character = []

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

# ╔═╡ e75d8f10-576f-4121-b9d7-87c1af825d41
chinese_character[100]

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
	mqClass = maxnorm.(qClass)
	#mqTemp = mNRCDT.(ext_chinese_character)
	mqTemp = maxnorm.(qTemp)
	#aqClass = aNRCDT.(TClass)
	aqClass = meannorm.(qClass)
	#aqTemp = aNRCDT.(ext_chinese_character)
	aqTemp = meannorm.(qTemp)
	@info "number of equispaced angles:" angle
	accuracy_k_nearest_neighbour(qTemp, unique(Labels), qClass, Labels, "inf", ret=1)
	accuracy_k_nearest_neighbour(qTemp, unique(Labels), qClass, Labels, "euclidean", ret=1)
	accuracy_k_nearest_neighbour(mqTemp, unique(Labels), mqClass, Labels, "inf", ret=1)
	accuracy_k_nearest_neighbour(mqTemp, unique(Labels), mqClass, Labels, "euclidean", ret=1)
	accuracy_k_nearest_neighbour(aqTemp, unique(Labels), aqClass, Labels, "inf", ret=1)
	accuracy_k_nearest_neighbour(aqTemp, unique(Labels), aqClass, Labels, "euclidean", ret=1)
end

# ╔═╡ Cell order:
# ╠═19587268-0828-11f0-01fa-e979f61f03a3
# ╠═6b753037-4286-407d-823b-7ff08f7aa505
# ╠═9d398d31-2bc5-449b-b750-6a88c1ce6767
# ╠═2450a95a-b230-4b73-b369-3aa36d3172b5
# ╠═782377c4-21bf-4f9f-a8ba-4b3bf849c696
# ╠═639e3aed-dde6-491c-a4a3-8be172766755
# ╠═dbe82aad-857e-45db-b56f-35e0acffde0c
# ╠═e75d8f10-576f-4121-b9d7-87c1af825d41
# ╠═8f5dfb6e-9c0d-4187-9fd1-1c0acccc4ac1
# ╠═28cbf21b-37a0-4307-bad7-7ff6d1efd511
# ╠═fc1f2b19-fe55-4421-a904-398a2448597b
# ╠═7e5f32b7-2f27-4877-a6fe-c6b41750aa1b
# ╠═0a193211-86e6-46cf-a265-e87ff8d00fc6
# ╠═2fab6713-3367-4696-b326-76d0f81b7429
# ╠═1edcfb3c-14fd-4c66-bccd-2e54acd0c544

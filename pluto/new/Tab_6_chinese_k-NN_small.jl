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

# ╔═╡ e75d8f10-576f-4121-b9d7-87c1af825d41
chinese_character[100]

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

# ╔═╡ 17b1028a-f314-43b2-9965-5a023467c31e
R = RadonTransform(256,128,0.0)

# ╔═╡ 0bbf22d4-61c5-4915-ac02-4f64ad328732
RCDT = RadonCDT(256, R)

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

# ╔═╡ f05629b6-34b6-46f4-8991-d64557322202
# ╠═╡ disabled = true
#=╠═╡
for prop in [5,10]
	for K in [1,3,5]
		@info "split:" prop, "K-NN:" K 
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 100, mqClass, Labels, "inf", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 100, mqClass, Labels, "euclidean", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 100, aqClass, Labels, "inf", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 100, aqClass, Labels, "euclidean", K=K, ret=0)
	end
end
  ╠═╡ =#

# ╔═╡ bcd3da7f-4c5c-41cc-b1dc-4577304aa184
# ╠═╡ disabled = true
#=╠═╡
for prop in [5,10]
	for K in [1,3,5]
		@info "split:" prop, "K-NN:" K 
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 100, Array{Float64}.(TClass), Labels, "inf", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 100, Array{Float64}.(TClass), Labels, "euclidean", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 100, qClass, Labels, "inf", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 100, qClass, Labels, "euclidean", K=K, ret=0)
	end
end
  ╠═╡ =#

# ╔═╡ 18bf4c54-bf10-42d8-a3d4-547b12fc16ba
# ╠═╡ disabled = true
#=╠═╡
Random.seed!(42); CC_31 = accuracy_k_nearest_part_neighbour(20, 5, 50, 100, mqClass, Labels, "euclidean", K=1, ret=1)
  ╠═╡ =#

# ╔═╡ 9235db9e-f230-454f-be5a-4090fd686ba1
# ╠═╡ disabled = true
#=╠═╡
pp = heatmap(exp.(-CC_31/20), fontfamily="Computer Modern", size=(550,500))
  ╠═╡ =#

# ╔═╡ 0d8f3d77-e013-48d4-84ce-9eb70003a818
# ╠═╡ disabled = true
#=╠═╡
savefig(pp, "maxNRCDT_conf_chinese_character_100_50_10_expm.pdf")
  ╠═╡ =#

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
# ╠═17b1028a-f314-43b2-9965-5a023467c31e
# ╠═0bbf22d4-61c5-4915-ac02-4f64ad328732
# ╠═a87c3b6a-11df-42b8-8f97-157b5c5ab763
# ╠═8a9bbc83-b783-4ead-844e-9f9e3632d90a
# ╠═b1e25447-dbec-4eeb-a554-0c6d08b71f04
# ╠═745baebd-b48a-4468-aea5-5e9d5806a25e
# ╠═c065302a-21d8-4924-9f6d-e118b141bb3c
# ╠═f05629b6-34b6-46f4-8991-d64557322202
# ╠═bcd3da7f-4c5c-41cc-b1dc-4577304aa184
# ╠═18bf4c54-bf10-42d8-a3d4-547b12fc16ba
# ╠═9235db9e-f230-454f-be5a-4090fd686ba1
# ╠═0d8f3d77-e013-48d4-84ce-9eb70003a818

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
chinese_character[101]

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

# ╔═╡ 17b1028a-f314-43b2-9965-5a023467c31e
R = RadonTransform(128,128,0.0)

# ╔═╡ 0bbf22d4-61c5-4915-ac02-4f64ad328732
RCDT = RadonCDT(128, R)

# ╔═╡ a87c3b6a-11df-42b8-8f97-157b5c5ab763
mNRCDT = MaxNormRadonCDT(RCDT)

# ╔═╡ 8a9bbc83-b783-4ead-844e-9f9e3632d90a
aNRCDT = MeanNormRadonCDT(RCDT)

# ╔═╡ 9ccddf1f-9a7b-4010-bbfa-905fe28320be
qClass = RCDT.(TClass)

# ╔═╡ 745baebd-b48a-4468-aea5-5e9d5806a25e
# ╠═╡ disabled = true
#=╠═╡
mqClass = max_normalization.(qClass)		#mqClass = mNRCDT.(TClass)
  ╠═╡ =#

# ╔═╡ c065302a-21d8-4924-9f6d-e118b141bb3c
# ╠═╡ disabled = true
#=╠═╡
aqClass = mean_normalization.(qClass)		#aqClass = aNRCDT.(TClass)
  ╠═╡ =#

# ╔═╡ 0383b820-3203-4188-8bc6-946599a43c81
# ╠═╡ disabled = true
#=╠═╡
for prop in [5,10]
	for K in [1,3,5]
		@info "split:" prop, "K-NN:" K 
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 1000, mqClass, Labels, "inf", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 1000, mqClass, Labels, "euclidean", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 1000, aqClass, Labels, "inf", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 1000, aqClass, Labels, "euclidean", K=K, ret=0)
	end
end
  ╠═╡ =#

# ╔═╡ 4d819a08-a727-4b6c-b373-c6917d113fd7
for prop in [5,10]
	for K in [1,3,5]
		@info "split:" prop, "K-NN:" K 
		#Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 1000, Array{Float64}.(TClass), Labels, "inf", K=K, ret=0)
		#Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 1000, Array{Float64}.(TClass), Labels, "euclidean", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 1000, qClass, Labels, "inf", K=K, ret=0)
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 50, 1000, qClass, Labels, "euclidean", K=K, ret=0)
	end
end

# ╔═╡ e1635663-8725-4907-95ff-cf4b25db49f1
# ╠═╡ disabled = true
#=╠═╡
Random.seed!(42); CC_11 = accuracy_k_nearest_part_neighbour(20, 5, 50, 1000, mqClass, Labels, "euclidean", K=1, ret=1)
  ╠═╡ =#

# ╔═╡ 20284ce5-34b6-4967-b3aa-f6947825cd1a
# ╠═╡ disabled = true
#=╠═╡
hn_11 = heatmap(CC_11/20, size=(500,450), fontfamily="Computer Modern")
  ╠═╡ =#

# ╔═╡ bce8db10-1336-4e8d-8387-b51330dfc6b0
# ╠═╡ disabled = true
#=╠═╡
hhp_11 = heatmap(log1p.(CC_11/20), size=(500,450), fontfamily="Computer Modern")
  ╠═╡ =#

# ╔═╡ 9ee58ff0-3ba5-4ff8-869c-3262db02493c
# ╠═╡ disabled = true
#=╠═╡
he_11 = heatmap(exp.(-CC_11), size=(500,450), fontfamily="Computer Modern")
  ╠═╡ =#

# ╔═╡ b05bef2e-6476-4c68-81b1-6c0b22d5f3d6
# ╠═╡ disabled = true
#=╠═╡
savefig(hn_11, "chinese_mnist_1000_1NN_max_eucl.pdf")
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
# ╠═9ccddf1f-9a7b-4010-bbfa-905fe28320be
# ╠═745baebd-b48a-4468-aea5-5e9d5806a25e
# ╠═c065302a-21d8-4924-9f6d-e118b141bb3c
# ╠═0383b820-3203-4188-8bc6-946599a43c81
# ╠═4d819a08-a727-4b6c-b373-c6917d113fd7
# ╠═e1635663-8725-4907-95ff-cf4b25db49f1
# ╠═20284ce5-34b6-4967-b3aa-f6947825cd1a
# ╠═bce8db10-1336-4e8d-8387-b51330dfc6b0
# ╠═9ee58ff0-3ba5-4ff8-869c-3262db02493c
# ╠═b05bef2e-6476-4c68-81b1-6c0b22d5f3d6

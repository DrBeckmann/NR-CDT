### A Pluto.jl notebook ###
# v0.20.5

using Markdown
using InteractiveUtils

# ╔═╡ 8cbe0300-edff-11ef-2fad-d3b8cca171a9
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
	using Random
	using JLD2
end

# ╔═╡ 5b58bf37-7019-4b8d-9dd4-36c04040a393
md"""
## Templates
Generate the three templates
using the submodule `TestImages`.
"""

# ╔═╡ c9a1f57f-1874-40e4-b47f-d66f7dd4a064
I₁ = render(OrbAndCross(Circle(),Star(1)), width=4);

# ╔═╡ 237a6d1c-6d30-40a0-8eb6-aa0ae913b6d2
J₁ = extend_image(I₁, (256, 256))

# ╔═╡ 8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
I₂ = render(OrbAndCross(Square(),Star(4)), width=4);

# ╔═╡ 29d43338-99a6-42ce-9cf6-eee91d3905b8
J₂ = extend_image(I₂, (256, 256))

# ╔═╡ c1959bcd-b5ce-40ae-9c41-142fca3027b6
# J = [J₁, J₂, J₃]; Label = [1, 2, 3];
J = [J₁, J₂]; Label = [1, 2];

# ╔═╡ 0a3ccf51-e303-4f8d-b627-0676df4f561d
md"""
## Dataset
Generate the dataset 
by duplicating the templates
and by applying random affine transformations
using the submodule `DataTransformations`.
"""

# ╔═╡ 14864b75-d2e6-476a-bf63-5ffffa95a61d
Class, Labels = generate_academic_classes(J, Label, class_size=10);

# ╔═╡ 773832af-9099-4dcf-bd1b-c82baaa83424
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.5, 1.25), 
	scale_y = (0.5, 1.25),
	rotate=(-180.0, 180.0), 
	shear_x=(-45.0, 45.0),
	shear_y=(-45.0, 45.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20)
	)

# ╔═╡ 580b2b44-6fb9-430c-ade9-32ab807fc59e
CC = zeros(2, 2, 8, 30, 30);

# ╔═╡ 0fd9510e-22e2-4892-91a9-179fec3d0ba9
radii= [16,20,24,28,32,40,48,57,64,80,96,112,128,149,170,213,256,322,388,450,512,640,768,896,1024,1280,1536,1792,2048]

# ╔═╡ 129ff404-469c-4a5d-9f30-169fe21ad714
inter = [1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,57,64,80,96,112,128,149,170,213,256]

# ╔═╡ a772fa92-bafd-4022-946d-fda53abb8ebf
for angle in [1,2,3,4,5,6,7,8] # 1,3,5,7
	for radii_i in 1:29
		for inter_i in 1:28
			R = RadonTransform(radii[radii_i],2^angle-1,0.0)
			RCDT = RadonCDT(inter[inter_i], R)
			mNRCDT = MaxNormRadonCDT(RCDT)
			aNRCDT = MeanNormRadonCDT(RCDT)
			TClass = A.(Class)
			try
				qTemp = RCDT.(J)
				qClass = RCDT.(TClass)
				mqTemp = max_normalization.(qTemp)
				mqClass = max_normalization.(qClass)
				aqTemp = mean_normalization.(qTemp)
				aqClass = mean_normalization.(qClass)
				@info "number of equispaced angles:" 2^angle-1 
				CC[1,1,angle,radii_i,inter_i], _ = accuracy_k_nearest_neighbour(mqTemp, Label, mqClass, Labels, "inf", ret=1)
				CC[1,2,angle,radii_i,inter_i], _ = accuracy_k_nearest_neighbour(mqTemp, Label, mqClass, Labels, "euclidean", ret=1)
				CC[2,1,angle,radii_i,inter_i], _ = accuracy_k_nearest_neighbour(aqTemp, Label, aqClass, Labels, "inf", ret=1)
				CC[2,2,angle,radii_i,inter_i], _ = accuracy_k_nearest_neighbour(aqTemp, Label, aqClass, Labels, "euclidean", ret=1)
			catch e
			end
			jldsave("acc_dis_2_NRCDTs.jld2"; CC)
		end
	end
end

# ╔═╡ 380facd0-73e4-40af-9be1-035481b37ea7
# ╠═╡ disabled = true
#=╠═╡
hh = heatmap(CC[2,2,8,:,:], size=(555,500), xticks = (1:8,[2,4,8,16,32,64,128,256]), yticks = (1:8,[16,32,64,128,256,512,1024,2048]), clim=(0,1), fontfamily="Computer Modern")
  ╠═╡ =#

# ╔═╡ 314dd40b-5fce-4454-a7ee-d18825221be1
# ╠═╡ disabled = true
#=╠═╡
savefig(hh, "meanNRCDT_eucl_256.pdf")
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╟─5b58bf37-7019-4b8d-9dd4-36c04040a393
# ╠═c9a1f57f-1874-40e4-b47f-d66f7dd4a064
# ╠═237a6d1c-6d30-40a0-8eb6-aa0ae913b6d2
# ╠═8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
# ╠═29d43338-99a6-42ce-9cf6-eee91d3905b8
# ╠═c1959bcd-b5ce-40ae-9c41-142fca3027b6
# ╟─0a3ccf51-e303-4f8d-b627-0676df4f561d
# ╠═14864b75-d2e6-476a-bf63-5ffffa95a61d
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═580b2b44-6fb9-430c-ade9-32ab807fc59e
# ╠═0fd9510e-22e2-4892-91a9-179fec3d0ba9
# ╠═129ff404-469c-4a5d-9f30-169fe21ad714
# ╠═a772fa92-bafd-4022-946d-fda53abb8ebf
# ╠═380facd0-73e4-40af-9be1-035481b37ea7
# ╠═314dd40b-5fce-4454-a7ee-d18825221be1

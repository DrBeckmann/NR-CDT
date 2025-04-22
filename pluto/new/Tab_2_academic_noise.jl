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

# ╔═╡ e6c9fb45-2ec7-4925-bc2c-efbed91caa46
I₃ = render(Shield(Triangle()), width=4);

# ╔═╡ 25220f99-8cbd-4387-b4fd-bb4a0e6fad96
J₃ = extend_image(I₃, (256, 256))

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
Class, Labels = generate_academic_classes(J, Label, class_size=50);

# ╔═╡ 773832af-9099-4dcf-bd1b-c82baaa83424
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.75, 1.0), 
	scale_y = (0.75, 1.0),
	rotate=(-180.0, 180.0), 
	#shear_x=(-5.0, 5.0),
	#shear_y=(-5.0, 5.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20)
	)

# ╔═╡ 5573ac62-3795-418c-857f-f66ef6211fa8
# ╠═╡ disabled = true
#=╠═╡
E = ElasticNoise(
    amplitude_x=(2.5, 7.5),
    amplitude_y=(2.5, 7.5),
    frequency_x=(0.5, 2.0),
    frequency_y=(0.5, 2.0),
	)
  ╠═╡ =#

# ╔═╡ a72700df-622e-4008-bdf5-97cdfb6f6951
# ╠═╡ disabled = true
#=╠═╡
S = SaltNoise(
    dots=(7, 10),
    width=(3/128, 3/128)
	)
  ╠═╡ =#

# ╔═╡ fb3629dc-1860-4a96-a75e-2b4402f847fe
# ╠═╡ disabled = true
#=╠═╡
TClass = S.(A.(E.(Class)))
  ╠═╡ =#

# ╔═╡ 16aaed33-371e-428f-8ba5-fd56d4ee1b76
md"""
## Max-Normalized RCDT -- Figure 4
Setup the max-normalized RCDT,
and apply it 
to the dataset and templates.
"""

# ╔═╡ b03bd8d5-eeaf-4713-85e3-c392bf59dd32
md"""
Plot the computed max-normalized RCDTs.
"""

# ╔═╡ 536d2775-bb0f-48c9-b51a-e67208803c75
md"""
## Nearest Neighbour Classification -- Table 2
Use the nearest neighbour classification
with respect to the chosen templates
to classify the generated dataset.
The max-normalized RCDT is applied
with different numbers of used angles.
"""

# ╔═╡ 580b2b44-6fb9-430c-ade9-32ab807fc59e
CC = zeros(2, 2, 3, 21, 21);

# ╔═╡ a772fa92-bafd-4022-946d-fda53abb8ebf
for angle in 1:3
	for num in 0:20
		for r in 1:21
			S = SaltNoise(
			    dots=(num, num),
			    width=(r/128/3, r/128/3)
				)
			TClass = S.(A.(Class))
			R = RadonTransform(850,4*2^(2*angle),0.0)
			RCDT = RadonCDT(64, R)
			qTemp = RCDT.(J)
			qClass = RCDT.(TClass)
			mqTemp = max_normalization.(qTemp)
			mqClass = max_normalization.(qClass)
			aqTemp = mean_normalization.(qTemp)
			aqClass = mean_normalization.(qClass)
			@info "number of equispaced angles:" 4*2^(2*angle)
			CC[1,1,angle,num+1,r], _ = accuracy_k_nearest_neighbour(mqTemp, Label, mqClass, Labels, "inf", ret=1)
			CC[1,2,angle,num+1,r], _ = accuracy_k_nearest_neighbour(mqTemp, Label, mqClass, Labels, "euclidean", ret=1)
			CC[2,1,angle,num+1,r], _ = accuracy_k_nearest_neighbour(aqTemp, Label, aqClass, Labels, "inf", ret=1)
			CC[2,2,angle,num+1,r], _ = accuracy_k_nearest_neighbour(aqTemp, Label, aqClass, Labels, "euclidean", ret=1)
			jldsave("acc_noise_NRCDTs.jld2"; CC)
		end
	end
end

# ╔═╡ 9b1580f7-2364-4c74-bb9b-1d36fe3c60d9
# ╠═╡ disabled = true
#=╠═╡
hh = heatmap(CC[2,2,3,:,:], size=(555,500), xticks = ([1,11,21],[1,11,21]), yticks = ([1,11,21],[0,10,20]), clim=(0,1), fontfamily="Computer Modern")
  ╠═╡ =#

# ╔═╡ 20796793-bd32-4bae-8706-1337025df15c
# ╠═╡ disabled = true
#=╠═╡
savefig(hh, "meanNRCDT_eucl_256_noise.pdf")
  ╠═╡ =#

# ╔═╡ 251b5bfd-55f7-4e9c-8a30-040e0baabdec
# ╠═╡ disabled = true
#=╠═╡
G = zeros(256,256); GG = zeros(21,256,256); sG = zeros(21,256); pG = zeros(21,256);
  ╠═╡ =#

# ╔═╡ d3732137-6696-4e9a-90ab-256c9a30d264
# ╠═╡ disabled = true
#=╠═╡
for r in 1:21
	S = SaltNoise(
		dots=(1, 1),
		width=(r/128/3, r/128/3),
		)
	GG[r,:,:] = S(G[:,:])
	sG[r,:] = maximum(GG[r,:,:],dims=1)
end
  ╠═╡ =#

# ╔═╡ 62c418f2-e3fe-472f-93ba-22c4732ca15b
# ╠═╡ disabled = true
#=╠═╡
using Plots.PlotMeasures
  ╠═╡ =#

# ╔═╡ f7a26bb8-a4a2-46d1-a2d7-d24d201a255a
# ╠═╡ disabled = true
#=╠═╡
for r in 1:21
	d = Int64(floor((sum(findall(!iszero,sG[r,:]))/length(findall(!iszero,sG[r,:])))))
	l = 11*r - d
	pG[r,:] = circshift(sG[r,:], l)
end
  ╠═╡ =#

# ╔═╡ 12552521-8e7d-4b6f-95ce-dcccb2c14fb8
# ╠═╡ disabled = true
#=╠═╡
pp = plot(maximum(pG, dims=1)[1,:], size=(1300,100), yticks=([0,0.5,1], [0,0.5,1]), xticks=(11*(1:21), 1:21), legend=false, fontfamily="Computer Modern", margin=10px)
  ╠═╡ =#

# ╔═╡ eb7f5db9-86f4-4fce-bc2a-f0046410462c
# ╠═╡ disabled = true
#=╠═╡
savefig(pp, "noise_dis.pdf")
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╟─5b58bf37-7019-4b8d-9dd4-36c04040a393
# ╠═c9a1f57f-1874-40e4-b47f-d66f7dd4a064
# ╠═237a6d1c-6d30-40a0-8eb6-aa0ae913b6d2
# ╠═8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
# ╠═29d43338-99a6-42ce-9cf6-eee91d3905b8
# ╠═e6c9fb45-2ec7-4925-bc2c-efbed91caa46
# ╠═25220f99-8cbd-4387-b4fd-bb4a0e6fad96
# ╠═c1959bcd-b5ce-40ae-9c41-142fca3027b6
# ╟─0a3ccf51-e303-4f8d-b627-0676df4f561d
# ╠═14864b75-d2e6-476a-bf63-5ffffa95a61d
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═5573ac62-3795-418c-857f-f66ef6211fa8
# ╠═a72700df-622e-4008-bdf5-97cdfb6f6951
# ╠═fb3629dc-1860-4a96-a75e-2b4402f847fe
# ╟─16aaed33-371e-428f-8ba5-fd56d4ee1b76
# ╟─b03bd8d5-eeaf-4713-85e3-c392bf59dd32
# ╟─536d2775-bb0f-48c9-b51a-e67208803c75
# ╠═580b2b44-6fb9-430c-ade9-32ab807fc59e
# ╠═a772fa92-bafd-4022-946d-fda53abb8ebf
# ╠═9b1580f7-2364-4c74-bb9b-1d36fe3c60d9
# ╠═20796793-bd32-4bae-8706-1337025df15c
# ╠═251b5bfd-55f7-4e9c-8a30-040e0baabdec
# ╠═d3732137-6696-4e9a-90ab-256c9a30d264
# ╠═62c418f2-e3fe-472f-93ba-22c4732ca15b
# ╠═f7a26bb8-a4a2-46d1-a2d7-d24d201a255a
# ╠═12552521-8e7d-4b6f-95ce-dcccb2c14fb8
# ╠═eb7f5db9-86f4-4fce-bc2a-f0046410462c

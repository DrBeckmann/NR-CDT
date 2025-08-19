### A Pluto.jl notebook ###
# v0.19.47

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

# ╔═╡ a827fd33-c5ec-4c5b-b209-1364f91fa1e6
md"""
# XXXX 2025 -- Figure 7
This pluto notebook reproduces the numerical experiment
for Figure 7 from

- Matthias Beckmann, Robert Beinert, Jonas Bresch, 
  'Normalized Radon Cummulative Distribution Transforms for Invariance and Robustness in Optimal Transport Based Image Classification',
  XXXX 2025.
"""

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
J = [J₁, J₂, J₃]; Label = [1, 2, 3];

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
	#scale_x = (0.75, 1.0), 
	#scale_y = (0.75, 1.0),
	rotate=(-180.0, 180.0), 
	#shear_x=(-5.0, 5.0),
	#shear_y=(-5.0, 5.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20)
	)

# ╔═╡ 536d2775-bb0f-48c9-b51a-e67208803c75
md"""
## Phase Transition -- Figure 7
Use the nearest neighbour classification
with respect to the chosen templates
to classify the generated dataset
with:

- noise stength (size of the salt noise) ranging from 1 to 21 according to Figure 5,
- noise component numbers (number of salt dots) ranging from 0 to 19.

The max- and mean-normalized RCDT is applied
with different numbers of used Radon angles, here (16,64,256).
"""

# ╔═╡ 580b2b44-6fb9-430c-ade9-32ab807fc59e
CC = zeros(2, 2, 3, 21, 21);

# ╔═╡ a772fa92-bafd-4022-946d-fda53abb8ebf
for angle in 1:3
	for num in 0:19
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
			mp = plot_quantiles(mqTemp, Label, mqClass, Labels);
			savefig(mp, "max_quantiles_salt_$(num)_$(r).pdf")
			CC[2,1,angle,num+1,r], _ = accuracy_k_nearest_neighbour(aqTemp, Label, aqClass, Labels, "inf", ret=1)
			CC[2,2,angle,num+1,r], _ = accuracy_k_nearest_neighbour(aqTemp, Label, aqClass, Labels, "euclidean", ret=1)
			ap = plot_quantiles(aqTemp, Label, aqClass, Labels);
			savefig(ap, "mean_quantiles_salt_$(num)_$(r).pdf")
			jldsave("acc_noise_2_NRCDTs.jld2"; CC)
		end
	end
end

# ╔═╡ ae1d6c6b-df40-4b27-aedd-2e0c7180e6f8
md"Visualizing the Phase transitions as provided in Figure 7."

# ╔═╡ 9b1580f7-2364-4c74-bb9b-1d36fe3c60d9
#=╠═╡
hh = heatmap(CC[2,2,3,:,:], size=(555,500), xticks = ([1,11,21],[1,11,21]), yticks = ([1,11,21],[0,10,20]), clim=(0,1), fontfamily="Computer Modern")
  ╠═╡ =#

# ╔═╡ 20796793-bd32-4bae-8706-1337025df15c
# ╠═╡ disabled = true
#=╠═╡
savefig(hh, "meanNRCDT_eucl_256_noise.pdf")
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─a827fd33-c5ec-4c5b-b209-1364f91fa1e6
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
# ╟─536d2775-bb0f-48c9-b51a-e67208803c75
# ╠═580b2b44-6fb9-430c-ade9-32ab807fc59e
# ╠═a772fa92-bafd-4022-946d-fda53abb8ebf
# ╟─ae1d6c6b-df40-4b27-aedd-2e0c7180e6f8
# ╠═9b1580f7-2364-4c74-bb9b-1d36fe3c60d9
# ╠═20796793-bd32-4bae-8706-1337025df15c

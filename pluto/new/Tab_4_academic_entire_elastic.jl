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
	using Colors, Plots, StatsBase, MultivariateStats
	using JLD2
	using Random
	Random.seed!(42)
end

# ╔═╡ 517d267d-5df3-49f7-9a55-3c16c56a91e0
md"""
# XXXX 2025 -- Table 4 (second rwo)
This pluto notebook reproduces the numerical experiment
for Table 4 (second row) from

- Matthias Beckmann, Robert Beinert, Jonas Bresch, 
  'Normalized Radon Cummulative Distribution Transforms for Invariance and Robustness in Optimal Transport Based Image Classification',
  XXXX 2025.
"""

# ╔═╡ 34899662-b911-435c-9bd6-126b5d1a01b3
md"""
## Templates
Generate the 12 templates
using the submodule `TestImages`.
"""

# ╔═╡ c9a1f57f-1874-40e4-b47f-d66f7dd4a064
I₁₁ = render(OrbAndCross(Circle(),Star(1)), width=4)

# ╔═╡ 79449727-86d4-45b7-b4c1-9ac2fcd88c52
J₁₁ = extend_image(I₁₁, (256, 256));

# ╔═╡ af494be1-3291-473a-8160-19de1869dd1d
I₁₂ = render(OrbAndCross(Circle(),Star(4)), width=4)

# ╔═╡ 5552472d-b396-4321-a02c-751952b18425
J₁₂ = extend_image(I₁₂, (256, 256));

# ╔═╡ 2aace893-44fe-4756-8ac1-5819ca509596
I₁₃ = render(OrbAndCross(Circle(),Star(8)), width=4)

# ╔═╡ 237a6d1c-6d30-40a0-8eb6-aa0ae913b6d2
J₁₃ = extend_image(I₁₃, (256, 256));

# ╔═╡ 8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
I₂₁ = render(OrbAndCross(Square(),Star(1)), width=4)

# ╔═╡ 29d43338-99a6-42ce-9cf6-eee91d3905b8
J₂₁ = extend_image(I₂₁, (256, 256));

# ╔═╡ 43d55941-e2b7-4228-95d4-0c43858d1089
I₂₂ = render(OrbAndCross(Square(),Star(4)), width=4)

# ╔═╡ f9d33d62-4b48-4878-b504-4d1d00f79c5a
J₂₂ = extend_image(I₂₂, (256, 256));

# ╔═╡ 59a3e2b7-4591-455f-885f-35a619329ce0
I₂₃ = render(OrbAndCross(Square(),Star(8)), width=4)

# ╔═╡ 767979b7-9e8b-4bfb-909a-5a50daef1c06
J₂₃ = extend_image(I₂₃, (256, 256));

# ╔═╡ 2519d179-3383-4bb3-bb19-97376cae9dbc
I₃₁ = render(OrbAndCross(Triangle(),Star(1)), width=4)

# ╔═╡ 83f5a7f3-94ea-42a2-b6f4-23ea07ae2357
J₃₁ = extend_image(I₃₁, (256, 256));

# ╔═╡ f03fd686-9bf6-44b2-839d-29f4c470a26d
I₃₂ = render(OrbAndCross(Triangle(),Star(4)), width=4)

# ╔═╡ 39e8dc3d-792d-425e-b4d1-04d617b2a338
J₃₂ = extend_image(I₃₂, (256, 256));

# ╔═╡ e6027f6b-1369-46cc-b9fb-399b7a6d0032
I₃₃ = render(OrbAndCross(Triangle(),Star(8)), width=4)

# ╔═╡ 4ad9a1c4-54e9-4e09-ac1e-76cc0d39686e
J₃₃ = extend_image(I₃₃, (256, 256));

# ╔═╡ 52c43eb3-c951-4124-9358-94073007df01
I₄₁ = render(Shield(Circle()), width=4)

# ╔═╡ 4305f1da-4d1f-4264-8c90-055a5127b917
J₄₁ = extend_image(I₄₁, (256, 256));

# ╔═╡ feae7daf-7267-4543-9707-286e52b15db7
I₄₂ = render(Shield(Square()), width=4)

# ╔═╡ f4dc4243-8223-48c1-bde5-4144072cc94e
J₄₂ = extend_image(I₄₂, (256, 256));

# ╔═╡ 875e9a13-7d49-4669-bdd6-f819f571f2d6
I₄₃ = render(Shield(Triangle()), width=4)

# ╔═╡ cad515d1-c4ed-47c2-90f9-b8b88ee30ded
J₄₃ = extend_image(I₄₃, (256, 256));

# ╔═╡ c1959bcd-b5ce-40ae-9c41-142fca3027b6
J = [J₁₁, J₁₂, J₁₃, J₂₁, J₂₂, J₂₃, J₃₁, J₃₂, J₃₃, J₄₁, J₄₂, J₄₃]; Label = collect(1:12);

# ╔═╡ 1d8e6bd2-4b26-4ea1-a983-8ae1ac2b8b7e
md"""
## Dataset
Generate the dataset 
by duplicating the templates
and by applying random affine transformations
using the submodule `DataTransformations`.
"""

# ╔═╡ 14864b75-d2e6-476a-bf63-5ffffa95a61d
Class, Labels = generate_academic_classes(J, Label, class_size=100);

# ╔═╡ 773832af-9099-4dcf-bd1b-c82baaa83424
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.5, 1.25), 
	scale_y = (0.5, 1.25),
	rotate=(-180.0, 180.0), 
	shear_x=(-45.0, 45.0),
	shear_y=(-45.0, 45.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))

# ╔═╡ c8585729-1dc6-437d-807f-f04896f067f1
E = DataTransformations.ElasticNoise(
	amplitude_x=(2.5, 7.5), 
	amplitude_y=(2.5, 7.5),
	frequency_x=(0.5, 2.0),
	frequency_y=(0.5, 2.0))

# ╔═╡ fb3629dc-1860-4a96-a75e-2b4402f847fe
Random.seed!(42); TClass = A.(E.(Class))

# ╔═╡ a19fe0fb-ba3f-4c81-8ba1-41123dfbbbde
md"""
## Nearest Neighbour Classification -- Table 4
Use the nearest neighbour classification
with respect to a 5 and 10 randomly chosen training samples 
from the generated dataset
to classify the generated dataset.
The max- and mean-normalized RCDT is applied
with different numbers of used angles.
Each experiment is repeated twenty times.
"""

# ╔═╡ 320c772a-ebc6-4248-a961-03b6246f429b
md"- NN using the RCDT, max- and mean-normalized RCDT embedding."

# ╔═╡ 493f7375-06e3-4f27-a2e6-8dfbd74e79be
for angle in [128] #[1,2,4,8,16,32,64,128]
	R = RadonTransform(850,angle,0.0);
	RCDT = RadonCDT(64, R);
	#NRCDT = NormRadonCDT(RCDT);
	#mNRCDT = MaxNormRadonCDT(RCDT);
	#aNRCDT = MeanNormRadonCDT(RCDT);
	qTemp = RCDT.(TClass);
	mqTemp = max_normalization.(qTemp);
	aqTemp = mean_normalization.(qTemp);
	@info "number of equispaced angles:" angle
	for prop in [5,10]
		for KK in [1,3,5]
			@info "split" prop, "k-NN" KK
			Random.seed!(42); CC = accuracy_k_nearest_part_neighbour(20, prop, 100, 12, qTemp, Labels, "inf", K=KK, ret=1);
			jldsave("conf_academic_elastic_$(KK)NN_$(prop)_RCDT_inf.jld2"; CC)
			Random.seed!(42); CC = accuracy_k_nearest_part_neighbour(20, prop, 100, 12, qTemp, Labels, "euclidean", K=KK, ret=1);
			jldsave("conf_academic_elastic_$(KK)NN_$(prop)_RCDT_eucl.jld2"; CC)
			Random.seed!(42); CC = accuracy_k_nearest_part_neighbour(20, prop, 100, 12, mqTemp, Labels, "inf", K=KK, ret=1);
			jldsave("conf_academic_elastic_$(KK)NN_$(prop)_maxNRCDT_inf.jld2"; CC)
			Random.seed!(42); CC = accuracy_k_nearest_part_neighbour(20, prop, 100, 12, mqTemp, Labels, "euclidean", K=KK, ret=1);
			jldsave("conf_academic_elastic_$(KK)NN_$(prop)_maxNRCDT_eucl.jld2"; CC)
			Random.seed!(42); CC = accuracy_k_nearest_part_neighbour(20, prop, 100, 12, aqTemp, Labels, "inf", K=KK, ret=1);
			jldsave("conf_academic_elastic_$(KK)NN_$(prop)_meanNRCDT_inf.jld2"; CC)
			Random.seed!(42); CC = accuracy_k_nearest_part_neighbour(20, prop, 100, 12, aqTemp, Labels, "euclidean", K=KK, ret=1);
			jldsave("conf_academic_elastic_$(KK)NN_$(prop)_meanNRCDT_eucl.jld2"; CC)
		end
	end
end

# ╔═╡ 930dfad0-65f9-4b77-b554-d5ebaf5fb30e
md"- NN using the Euclidean embedding."

# ╔═╡ 97a6f3d6-b473-42e7-925a-d70321d1b236
for prop in [5,10]
	for KK in [1,3,5]
		@info "split" prop, "k-NN" KK
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 100, 12, Array{Float64}.(TClass), Labels, "inf", K=KK);
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 100, 12, Array{Float64}.(TClass), Labels, "euclidean", K=KK);
	end
end

# ╔═╡ Cell order:
# ╟─517d267d-5df3-49f7-9a55-3c16c56a91e0
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╟─34899662-b911-435c-9bd6-126b5d1a01b3
# ╠═c9a1f57f-1874-40e4-b47f-d66f7dd4a064
# ╠═79449727-86d4-45b7-b4c1-9ac2fcd88c52
# ╠═af494be1-3291-473a-8160-19de1869dd1d
# ╠═5552472d-b396-4321-a02c-751952b18425
# ╠═2aace893-44fe-4756-8ac1-5819ca509596
# ╠═237a6d1c-6d30-40a0-8eb6-aa0ae913b6d2
# ╠═8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
# ╠═29d43338-99a6-42ce-9cf6-eee91d3905b8
# ╠═43d55941-e2b7-4228-95d4-0c43858d1089
# ╠═f9d33d62-4b48-4878-b504-4d1d00f79c5a
# ╠═59a3e2b7-4591-455f-885f-35a619329ce0
# ╠═767979b7-9e8b-4bfb-909a-5a50daef1c06
# ╠═2519d179-3383-4bb3-bb19-97376cae9dbc
# ╠═83f5a7f3-94ea-42a2-b6f4-23ea07ae2357
# ╠═f03fd686-9bf6-44b2-839d-29f4c470a26d
# ╠═39e8dc3d-792d-425e-b4d1-04d617b2a338
# ╠═e6027f6b-1369-46cc-b9fb-399b7a6d0032
# ╠═4ad9a1c4-54e9-4e09-ac1e-76cc0d39686e
# ╠═52c43eb3-c951-4124-9358-94073007df01
# ╠═4305f1da-4d1f-4264-8c90-055a5127b917
# ╠═feae7daf-7267-4543-9707-286e52b15db7
# ╠═f4dc4243-8223-48c1-bde5-4144072cc94e
# ╠═875e9a13-7d49-4669-bdd6-f819f571f2d6
# ╠═cad515d1-c4ed-47c2-90f9-b8b88ee30ded
# ╠═c1959bcd-b5ce-40ae-9c41-142fca3027b6
# ╟─1d8e6bd2-4b26-4ea1-a983-8ae1ac2b8b7e
# ╠═14864b75-d2e6-476a-bf63-5ffffa95a61d
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═c8585729-1dc6-437d-807f-f04896f067f1
# ╠═fb3629dc-1860-4a96-a75e-2b4402f847fe
# ╟─a19fe0fb-ba3f-4c81-8ba1-41123dfbbbde
# ╟─320c772a-ebc6-4248-a961-03b6246f429b
# ╠═493f7375-06e3-4f27-a2e6-8dfbd74e79be
# ╟─930dfad0-65f9-4b77-b554-d5ebaf5fb30e
# ╠═97a6f3d6-b473-42e7-925a-d70321d1b236

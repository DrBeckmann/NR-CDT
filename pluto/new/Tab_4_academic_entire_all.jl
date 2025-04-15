### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 8cbe0300-edff-11ef-2fad-d3b8cca171a9
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
	using Colors, Plots, StatsBase, MultivariateStats
	using Random
	Random.seed!(42)
end

# ╔═╡ c9a1f57f-1874-40e4-b47f-d66f7dd4a064
I₁₁ = render(OrbAndCross(Circle(),Star(1)))

# ╔═╡ 79449727-86d4-45b7-b4c1-9ac2fcd88c52
J₁₁ = extend_image(I₁₁, (256, 256));

# ╔═╡ af494be1-3291-473a-8160-19de1869dd1d
I₁₂ = render(OrbAndCross(Circle(),Star(4)))

# ╔═╡ 5552472d-b396-4321-a02c-751952b18425
J₁₂ = extend_image(I₁₂, (256, 256));

# ╔═╡ 2aace893-44fe-4756-8ac1-5819ca509596
I₁₃ = render(OrbAndCross(Circle(),Star(8)))

# ╔═╡ 237a6d1c-6d30-40a0-8eb6-aa0ae913b6d2
J₁₃ = extend_image(I₁₃, (256, 256));

# ╔═╡ 8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
I₂₁ = render(OrbAndCross(Square(),Star(1)))

# ╔═╡ 29d43338-99a6-42ce-9cf6-eee91d3905b8
J₂₁ = extend_image(I₂₁, (256, 256));

# ╔═╡ 43d55941-e2b7-4228-95d4-0c43858d1089
I₂₂ = render(OrbAndCross(Square(),Star(4)))

# ╔═╡ f9d33d62-4b48-4878-b504-4d1d00f79c5a
J₂₂ = extend_image(I₂₂, (256, 256));

# ╔═╡ 59a3e2b7-4591-455f-885f-35a619329ce0
I₂₃ = render(OrbAndCross(Square(),Star(8)))

# ╔═╡ 767979b7-9e8b-4bfb-909a-5a50daef1c06
J₂₃ = extend_image(I₂₃, (256, 256));

# ╔═╡ 2519d179-3383-4bb3-bb19-97376cae9dbc
I₃₁ = render(OrbAndCross(Triangle(),Star(1)))

# ╔═╡ 83f5a7f3-94ea-42a2-b6f4-23ea07ae2357
J₃₁ = extend_image(I₃₁, (256, 256));

# ╔═╡ f03fd686-9bf6-44b2-839d-29f4c470a26d
I₃₂ = render(OrbAndCross(Triangle(),Star(4)))

# ╔═╡ 39e8dc3d-792d-425e-b4d1-04d617b2a338
J₃₂ = extend_image(I₃₂, (256, 256));

# ╔═╡ e6027f6b-1369-46cc-b9fb-399b7a6d0032
I₃₃ = render(OrbAndCross(Triangle(),Star(8)))

# ╔═╡ 4ad9a1c4-54e9-4e09-ac1e-76cc0d39686e
J₃₃ = extend_image(I₃₃, (256, 256));

# ╔═╡ 52c43eb3-c951-4124-9358-94073007df01
I₄₁ = render(Shield(Circle()))

# ╔═╡ 4305f1da-4d1f-4264-8c90-055a5127b917
J₄₁ = extend_image(I₄₁, (256, 256));

# ╔═╡ feae7daf-7267-4543-9707-286e52b15db7
I₄₂ = render(Shield(Square()))

# ╔═╡ f4dc4243-8223-48c1-bde5-4144072cc94e
J₄₂ = extend_image(I₄₂, (256, 256));

# ╔═╡ 875e9a13-7d49-4669-bdd6-f819f571f2d6
I₄₃ = render(Shield(Triangle()))

# ╔═╡ cad515d1-c4ed-47c2-90f9-b8b88ee30ded
J₄₃ = extend_image(I₄₃, (256, 256));

# ╔═╡ a2ce201f-456c-449b-9d4a-34b02a7579c3
I₄ = render(OrbAndCross(Triangle(),Star(5)));

# ╔═╡ c1959bcd-b5ce-40ae-9c41-142fca3027b6
J = [J₁₁, J₁₂, J₁₃, J₂₁, J₂₂, J₂₃, J₃₁, J₃₂, J₃₃, J₄₁, J₄₂, J₄₃]; Label = collect(1:12);

# ╔═╡ 14864b75-d2e6-476a-bf63-5ffffa95a61d
Class, Labels = generate_academic_classes(J, Label, class_size=100);

# ╔═╡ 773832af-9099-4dcf-bd1b-c82baaa83424
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.75, 1.25), 
	scale_y = (0.75, 1.25),
	rotate=(-45.0, 45.0), 
	shear_x=(-5.0, 5.0),
	shear_y=(-5.0, 5.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))

# ╔═╡ 1f303cbf-8caf-4c85-8f2a-a1460a4c31c3
S = DataTransformations.SaltNoise((5,10), (3/128, 3/128))

# ╔═╡ c8585729-1dc6-437d-807f-f04896f067f1
E = DataTransformations.ElasticNoise(
	amplitude_x=(2.5, 7.5), 
	amplitude_y=(2.5, 7.5),
	frequency_x=(0.5, 2.0),
	frequency_y=(0.5, 2.0))

# ╔═╡ fb3629dc-1860-4a96-a75e-2b4402f847fe
# TClass = S.(A.(E.(Class)))
Random.seed!(42); TClass = S.(A.(E.(Class)))
# TClass = S.(B.(A.(Class)))

# ╔═╡ a8293654-d83e-4b37-8d3f-1070cb5c6571
for angle in [128] #[1,2,4,8,16,32,64,128]
	R = RadonTransform(256,angle,0.0);
	RCDT = RadonCDT(256, R);
	#NRCDT = NormRadonCDT(RCDT);
	#mNRCDT = MaxNormRadonCDT(RCDT);
	#aNRCDT = MeanNormRadonCDT(RCDT);
	qTemp = RCDT.(TClass);
	#mqTemp = mNRCDT.(TClass);
	mqTemp = max_normalization.(qTemp)
	#aqTemp = aNRCDT.(TClass);
	aqTemp = mean_normalization.(qTemp)
	@info "number of equispaced angles:" angle
	for prop in [5,10]
		for KK in [1,3,5]
			@info "split" prop, "k-NN" KK
			Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 100, 12, qTemp, Labels, "inf", K=KK);
			Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 100, 12, qTemp, Labels, "euclidean", K=KK);
			Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 100, 12, mqTemp, Labels, "inf", K=KK);
			Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 100, 12, mqTemp, Labels, "euclidean", K=KK);
			Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 100, 12, aqTemp, Labels, "inf", K=KK);
			Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 100, 12, aqTemp, Labels, "euclidean", K=KK);
		end
	end
end

# ╔═╡ 1470cd30-2f5a-4411-9e9c-7ec87a1365a0
for prop in [5,10]
	for KK in [1,3,5]
		@info "split" prop, "k-NN" KK
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 100, 12, Array{Float64}.(TClass), Labels, "inf", K=KK);
		Random.seed!(42); accuracy_k_nearest_part_neighbour(20, prop, 100, 12, Array{Float64}.(TClass), Labels, "euclidean", K=KK);
	end
end

# ╔═╡ 485c0121-b555-4b28-b8d7-d7e2525003a0
# ╠═╡ disabled = true
#=╠═╡
R = RadonTransform(256,128,0.0);
  ╠═╡ =#

# ╔═╡ aded54a8-3850-4981-9676-10040ff7a9b5
# ╠═╡ disabled = true
#=╠═╡
RCDT = RadonCDT(256, R);
  ╠═╡ =#

# ╔═╡ b4debe92-0af9-47d2-b242-a1e5a1b0eb19
# ╠═╡ disabled = true
#=╠═╡
mNRCDT = MeanNormRadonCDT(RCDT);
  ╠═╡ =#

# ╔═╡ d5e67310-6f8d-4f38-91c4-9e4aa003f57d
# ╠═╡ disabled = true
#=╠═╡
aNRCDT = MeanNormRadonCDT(RCDT);
  ╠═╡ =#

# ╔═╡ 253befee-8f8f-432a-b374-a50883296d04
# ╠═╡ disabled = true
#=╠═╡
aqClass = aNRCDT.(TClass);
  ╠═╡ =#

# ╔═╡ ee031d2b-7653-4259-8a16-d818c9be9e28
# ╠═╡ disabled = true
#=╠═╡
mqClass = mNRCDT.(TClass);
  ╠═╡ =#

# ╔═╡ 170c5ef2-b3b1-42ab-9311-451dc7713558
# ╠═╡ disabled = true
#=╠═╡
CC = accuracy_k_nearest_part_neighbour(20, 5, 100, 12, aqClass, Labels, "euclidean", K=1, ret=1);
  ╠═╡ =#

# ╔═╡ 3ce4e9e1-542a-4747-a5b7-3df22401f8d9
# ╠═╡ disabled = true
#=╠═╡
hh = heatmap(CC/20, fontfamily="Computer Modern", size=(550,500), xticks=(1:12, 1:12), yticks=(1:12, 1:12))
  ╠═╡ =#

# ╔═╡ 90841bf1-9ef2-4fb8-bfba-9c214467858a
# ╠═╡ disabled = true
#=╠═╡
hhp = heatmap(log1p.(CC/20), fontfamily="Computer Modern", size=(550,500), xticks=(1:12, 1:12), yticks=(1:12, 1:12))
  ╠═╡ =#

# ╔═╡ 5ddd00f3-ca97-4b74-af75-b8a16783c666
# ╠═╡ disabled = true
#=╠═╡
savefig(hh, "academic_entire_all_1NN_5_mean_eucl.pdf")
  ╠═╡ =#

# ╔═╡ 175d2884-a973-47e8-85be-17888a4eaa2c
# ╠═╡ disabled = true
#=╠═╡
savefig(hhp, "academic_entire_all_1NN_5_mean_eucl_1p.pdf")
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
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
# ╠═a2ce201f-456c-449b-9d4a-34b02a7579c3
# ╠═c1959bcd-b5ce-40ae-9c41-142fca3027b6
# ╠═14864b75-d2e6-476a-bf63-5ffffa95a61d
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═1f303cbf-8caf-4c85-8f2a-a1460a4c31c3
# ╠═c8585729-1dc6-437d-807f-f04896f067f1
# ╠═fb3629dc-1860-4a96-a75e-2b4402f847fe
# ╠═a8293654-d83e-4b37-8d3f-1070cb5c6571
# ╠═1470cd30-2f5a-4411-9e9c-7ec87a1365a0
# ╠═485c0121-b555-4b28-b8d7-d7e2525003a0
# ╠═aded54a8-3850-4981-9676-10040ff7a9b5
# ╠═b4debe92-0af9-47d2-b242-a1e5a1b0eb19
# ╠═d5e67310-6f8d-4f38-91c4-9e4aa003f57d
# ╠═253befee-8f8f-432a-b374-a50883296d04
# ╠═ee031d2b-7653-4259-8a16-d818c9be9e28
# ╠═170c5ef2-b3b1-42ab-9311-451dc7713558
# ╠═3ce4e9e1-542a-4747-a5b7-3df22401f8d9
# ╠═90841bf1-9ef2-4fb8-bfba-9c214467858a
# ╠═5ddd00f3-ca97-4b74-af75-b8a16783c666
# ╠═175d2884-a973-47e8-85be-17888a4eaa2c

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
Class, Labels = generate_academic_classes(J, Label, class_size=10);

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

# ╔═╡ 485c0121-b555-4b28-b8d7-d7e2525003a0
R = RadonTransform(256,128,0.0);

# ╔═╡ aded54a8-3850-4981-9676-10040ff7a9b5
RCDT = RadonCDT(256, R);

# ╔═╡ d5e67310-6f8d-4f38-91c4-9e4aa003f57d
aNRCDT = MeanNormRadonCDT(RCDT);

# ╔═╡ 22eeb170-0e9e-48d4-a67f-54889977815e
mNRCDT = MaxNormRadonCDT(RCDT);

# ╔═╡ bd13ec88-733a-4b51-afbb-6a6f713c7944
qClass = RCDT.(TClass);

# ╔═╡ 253befee-8f8f-432a-b374-a50883296d04
aqClass = aNRCDT.(TClass);

# ╔═╡ f7a43851-d5af-4a63-9264-d557b72b54ce
mqClass = mNRCDT.(TClass);

# ╔═╡ 31a688dc-6a49-400a-952f-05f926bfe75f
qTemp = RCDT.(J);

# ╔═╡ ecfef1eb-c149-40ce-994a-46e50ca84d77
mqTemp = mNRCDT.(J);

# ╔═╡ 0730ef64-5baa-4ddd-9b54-44d5de9cfe4b
aqTemp = aNRCDT.(J);

# ╔═╡ 2b4a9e41-ddb8-41c2-bb92-204db41dfa85
accuracy_k_nearest_neighbour(Array{Float64}.(J), Label, Array{Float64}.(TClass), Labels, "inf", ret=1);

# ╔═╡ dbcf0415-09ea-49b1-b637-74f4625d8343
accuracy_k_nearest_neighbour(Array{Float64}.(J), Label, Array{Float64}.(TClass), Labels, "euclidean", ret=1);

# ╔═╡ 548356e2-570f-45e8-856c-372afdc890f3
for angle in [1,2,4,8,16,32,64,128]
	R = RadonTransform(256,angle,0.0);
	RCDT = RadonCDT(256, R);
	NRCDT = NormRadonCDT(RCDT);
	mNRCDT = MaxNormRadonCDT(RCDT);
	aNRCDT = MeanNormRadonCDT(RCDT);
	qClass = RCDT.(TClass);
	qTemp = RCDT.(J);
	mqClass = mNRCDT.(TClass);
	mqTemp = mNRCDT.(J);
	aqClass = aNRCDT.(TClass);
	aqTemp = aNRCDT.(J);
	@info "number of equispaced angles:" angle
	accuracy_k_nearest_neighbour(qTemp, Label, qClass, Labels, "inf", ret=1);
	accuracy_k_nearest_neighbour(qTemp, Label, qClass, Labels, "euclidean", ret=1);
	accuracy_k_nearest_neighbour(mqTemp, Label, mqClass, Labels, "inf", ret=1);
	accuracy_k_nearest_neighbour(mqTemp, Label, mqClass, Labels, "euclidean", ret=1);
	accuracy_k_nearest_neighbour(aqTemp, Label, aqClass, Labels, "inf", ret=1);
	accuracy_k_nearest_neighbour(aqTemp, Label, aqClass, Labels, "euclidean", ret=1);
end

# ╔═╡ d1757123-7e1f-4de7-b2c6-67694f911d82
# ╠═╡ disabled = true
#=╠═╡
CC = accuracy_k_nearest_part_neighbour(20, 50, 100, 12, mqClass, Labels, "inf", K=5, ret=1);
  ╠═╡ =#

# ╔═╡ 3ce4e9e1-542a-4747-a5b7-3df22401f8d9
# ╠═╡ disabled = true
#=╠═╡
hh = heatmap(CC/20, size=(550,500), xticks=(1:12, 1:12), yticks=(1:12, 1:12))
  ╠═╡ =#

# ╔═╡ 5ddd00f3-ca97-4b74-af75-b8a16783c666
# ╠═╡ disabled = true
#=╠═╡
savefig(hh, "academic_entire_1NN_10_elastic.pdf")
  ╠═╡ =#

# ╔═╡ ee292de9-a64d-409f-8b30-97c2b9a84a05
# ╠═╡ disabled = true
#=╠═╡
aqMClass = mapreduce(permutedims, vcat, aqClass);
  ╠═╡ =#

# ╔═╡ 40f3c8d2-13a6-4581-b9a4-d4eb95de5bb9
# ╠═╡ disabled = true
#=╠═╡
aqJClass = aNRCDT.(J);
  ╠═╡ =#

# ╔═╡ d68ba277-aa95-4ad3-907c-fc4946ffd6d9
# ╠═╡ disabled = true
#=╠═╡
aqJMClass = mapreduce(permutedims, vcat, aqJClass);
  ╠═╡ =#

# ╔═╡ 6cbbc354-aa56-443e-8986-b9160bdce731
#=╠═╡
pca_model = fit(PCA, aqMClass'; maxoutdim=2);
  ╠═╡ =#

# ╔═╡ daa8335d-e110-40e5-922f-256148cd71df
#=╠═╡
pca_model_J = fit(PCA, aqJMClass'; maxoutdim=2);
  ╠═╡ =#

# ╔═╡ 9b9696d5-5aa2-4511-857c-25bc9304d97e
#=╠═╡
reduced_data = transform(pca_model, aqMClass');
  ╠═╡ =#

# ╔═╡ e0471d7d-d29d-4740-aa21-c11883085c8e
#=╠═╡
reduced_data_J = transform(pca_model_J, aqJMClass');
  ╠═╡ =#

# ╔═╡ ee7b0eb5-8870-43e3-8b1f-73434a76600c
#=╠═╡
pc1, pc2 = reduced_data[1, :], reduced_data[2, :];
  ╠═╡ =#

# ╔═╡ 656add7d-5f60-41ae-8a15-acdb28b65c3d
#=╠═╡
pcJ1, pcJ2 = reduced_data_J[1, :], reduced_data_J[2, :];
  ╠═╡ =#

# ╔═╡ cf93ea6e-2daf-4c01-af5d-529cb9e12b0f
# ╠═╡ disabled = true
#=╠═╡
unique_labels = unique(Labels)
  ╠═╡ =#

# ╔═╡ 7689530d-1ff5-47b0-ac69-01ff069c6d49
#=╠═╡
color_map = distinguishable_colors(length(unique_labels))
  ╠═╡ =#

# ╔═╡ bdbce52a-4200-4a17-80a3-fbcc4bb9ed13
#=╠═╡
label_colors = Dict(label => color_map[i] for (i, label) in enumerate(unique_labels))
  ╠═╡ =#

# ╔═╡ 7a89fc07-dfc4-42c9-92be-a70f2cd8dbda
#=╠═╡
colors = [label_colors[label] for label in Labels]
  ╠═╡ =#

# ╔═╡ 0f5047ec-fe23-4b17-8fc0-be957cf9d758
#=╠═╡
colorsJ = [label_colors[label] for label in Label]
  ╠═╡ =#

# ╔═╡ 4cf12be8-ed32-4996-a96b-0b98141a2e46
#=╠═╡
h = scatter(pc1, pc2, color=colors, alpha=0.33, markersize=5, xlabel="PC1", ylabel="PC2", label=false);
  ╠═╡ =#

# ╔═╡ e4e85b10-bf9a-4d27-b9e2-6f392f49b5f5
#=╠═╡
scatter!(h, pcJ1, pcJ2, color=colorsJ, markersize=5, xlabel="PC1", ylabel="PC2", label=false);
  ╠═╡ =#

# ╔═╡ c2e538f9-08b0-4ad7-acb4-34b393e29a94
#=╠═╡
for label in unique_labels
        scatter!(h, [], [], markershape=:circle, markersize=5, color=label_colors[label], label="Class $label", legend=true, framestyle=:box)
    end
  ╠═╡ =#

# ╔═╡ 65e7c0f0-e90c-4295-a08d-f9e2aed031c2
#=╠═╡
h
  ╠═╡ =#

# ╔═╡ 845da9b7-51b9-4eaa-b096-8ecb2491e448
#=╠═╡
savefig(h, "pca_cluster_academic_entire_salt.pdf")
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
# ╠═485c0121-b555-4b28-b8d7-d7e2525003a0
# ╠═aded54a8-3850-4981-9676-10040ff7a9b5
# ╠═d5e67310-6f8d-4f38-91c4-9e4aa003f57d
# ╠═22eeb170-0e9e-48d4-a67f-54889977815e
# ╠═bd13ec88-733a-4b51-afbb-6a6f713c7944
# ╠═253befee-8f8f-432a-b374-a50883296d04
# ╠═f7a43851-d5af-4a63-9264-d557b72b54ce
# ╠═31a688dc-6a49-400a-952f-05f926bfe75f
# ╠═ecfef1eb-c149-40ce-994a-46e50ca84d77
# ╠═0730ef64-5baa-4ddd-9b54-44d5de9cfe4b
# ╠═2b4a9e41-ddb8-41c2-bb92-204db41dfa85
# ╠═dbcf0415-09ea-49b1-b637-74f4625d8343
# ╠═548356e2-570f-45e8-856c-372afdc890f3
# ╠═d1757123-7e1f-4de7-b2c6-67694f911d82
# ╠═3ce4e9e1-542a-4747-a5b7-3df22401f8d9
# ╠═5ddd00f3-ca97-4b74-af75-b8a16783c666
# ╠═ee292de9-a64d-409f-8b30-97c2b9a84a05
# ╠═40f3c8d2-13a6-4581-b9a4-d4eb95de5bb9
# ╠═d68ba277-aa95-4ad3-907c-fc4946ffd6d9
# ╠═6cbbc354-aa56-443e-8986-b9160bdce731
# ╠═daa8335d-e110-40e5-922f-256148cd71df
# ╠═9b9696d5-5aa2-4511-857c-25bc9304d97e
# ╠═e0471d7d-d29d-4740-aa21-c11883085c8e
# ╠═ee7b0eb5-8870-43e3-8b1f-73434a76600c
# ╠═656add7d-5f60-41ae-8a15-acdb28b65c3d
# ╠═cf93ea6e-2daf-4c01-af5d-529cb9e12b0f
# ╠═7689530d-1ff5-47b0-ac69-01ff069c6d49
# ╠═bdbce52a-4200-4a17-80a3-fbcc4bb9ed13
# ╠═7a89fc07-dfc4-42c9-92be-a70f2cd8dbda
# ╠═0f5047ec-fe23-4b17-8fc0-be957cf9d758
# ╠═4cf12be8-ed32-4996-a96b-0b98141a2e46
# ╠═e4e85b10-bf9a-4d27-b9e2-6f392f49b5f5
# ╠═c2e538f9-08b0-4ad7-acb4-34b393e29a94
# ╠═65e7c0f0-e90c-4295-a08d-f9e2aed031c2
# ╠═845da9b7-51b9-4eaa-b096-8ecb2491e448

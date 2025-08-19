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
	using Random
	Random.seed!(42)
end

# ╔═╡ 2a981b3c-2765-49cf-8bbd-038f12904400
md"""
# arXiv:2506.08761 -- Table 2 (1st block), Figure 2
This pluto notebook reproduces the numerical experiment
for Table 2 (first block) and Figure 2 from

- Matthias Beckmann, Robert Beinert, Jonas Bresch, 
  'Normalized Radon Cummulative Distribution Transforms for Invariance and Robustness in Optimal Transport Based Image Classification',
  arXiv:2506.08761, 2025.
"""

# ╔═╡ 9997f3ac-b425-4881-9d46-579f3abe282b
md"""
## Templates
Generate the three templates
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

# ╔═╡ 47936e86-cfce-409f-8161-7755fff4536e
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
	shift_y=(-20, 20))

# ╔═╡ fb3629dc-1860-4a96-a75e-2b4402f847fe
Random.seed!(42); TClass = A.(Class)

# ╔═╡ 3ef98e3a-160f-42cc-a22a-1cba512697c0
md"""
## Max- and Mean-Normalized RCDT -- Figure 2
Setup the max- and mean-normalized RCDT,
and apply it 
to the dataset and templates.
"""

# ╔═╡ 7ebbd156-d1dc-49bb-a7f0-1b949a79ddac
R = RadonTransform(850,256,0.0)

# ╔═╡ 99fafed4-247c-4f9a-b945-02c39dd29c9c
RCDT = RadonCDT(64, R)

# ╔═╡ d6b09c3f-4383-4022-a2b9-3b05b533e3d0
NRCDT = NormRadonCDT(RCDT)

# ╔═╡ 33ed1571-93e2-4c1e-9c6b-9dbc962fdaed
mNRCDT = MaxNormRadonCDT(RCDT)

# ╔═╡ e6ce8adc-9c8a-4b38-92c6-ff713fdc72ed
aNRCDT = MeanNormRadonCDT(RCDT)

# ╔═╡ 540c0a0b-ed62-401b-a15d-e0d7d4d6228c
md"""
Reduce the set of templates and test images to those from class 5 and 12.
"""

# ╔═╡ 274784d5-088b-42d6-b4c2-4491c2f9ec2f
set = append!(collect(41:50), collect(111:120)); sett = [5,12];

# ╔═╡ 5f0a1b08-f2ae-4257-993b-cc3740775a3c
redTClass = TClass[set];

# ╔═╡ 5b6596f8-2f44-4624-817c-f2772a0ac250
redJ = J[sett];

# ╔═╡ 3611d48f-d50c-4b4e-9c89-e78c203f993b
redLabel = Label[sett];

# ╔═╡ 1b9dcd42-cb2c-4570-90fd-7592013b409b
redLabels = Labels[set];

# ╔═╡ 28ee2767-5a3d-4fac-a213-0df1a2673d05
mqredClass = mNRCDT.(redTClass);

# ╔═╡ c3dedd73-eebe-4403-b69f-ffa45334f471
aqredClass = aNRCDT.(redTClass);

# ╔═╡ d8c16a36-dd0b-4621-997b-f7156d3d1f88
mqredTemp = mNRCDT.(redJ);

# ╔═╡ 8d4bacec-6758-49ea-b694-fd1aee22a22b
aqredTemp = aNRCDT.(redJ);

# ╔═╡ 4e7c3707-5ed1-4888-a7f6-e490718f4692
md"""
Plot the computed max-normalized RCDTs.
"""

# ╔═╡ ed505ea7-c690-4813-9bc5-569e18c698b8
mp = plot_quantiles(mqredTemp, redLabel, mqredClass, redLabels)

# ╔═╡ 5bc61555-1ce4-49a2-8347-a53d79571760
md"""
Plot the computed mean-normalized RCDTs.
"""

# ╔═╡ 21f9c54c-dd3b-4e00-888f-4c91363d0385
ap = plot_quantiles(aqredTemp, redLabel, aqredClass, redLabels)

# ╔═╡ f199f6d0-970e-4fb9-b126-e4530f17671a
md"""
## Nearest Neighbour Classification -- Table 2
Use the nearest neighbour classification
with respect to the chosen templates
to classify the generated dataset.
The max- and mean-normalized RCDT is applied
with different numbers of used angles.
"""

# ╔═╡ 2b4a9e41-ddb8-41c2-bb92-204db41dfa85
accuracy_k_nearest_neighbour(Array{Float64}.(J), Label, Array{Float64}.(TClass), Labels, "inf", ret=1);

# ╔═╡ dbcf0415-09ea-49b1-b637-74f4625d8343
accuracy_k_nearest_neighbour(Array{Float64}.(J), Label, Array{Float64}.(TClass), Labels, "euclidean", ret=1);

# ╔═╡ 29c2d769-1bbb-4d8f-8bd8-874eaea96ac9
md"Short cut for the computations of the max- and mean-normalized RCDT by computing once the entire RCDT."

# ╔═╡ 548356e2-570f-45e8-856c-372afdc890f3
for angle in [1,2,4,8,16,32,64,128,256]
	R = RadonTransform(850,angle,0.0);
	RCDT = RadonCDT(64, R);
	NRCDT = NormRadonCDT(RCDT);
	#mNRCDT = MaxNormRadonCDT(RCDT);
	#aNRCDT = MeanNormRadonCDT(RCDT);
	qClass = RCDT.(TClass);
	qTemp = RCDT.(J);
	mqClass = max_normalization.(qClass);
	mqTemp = max_normalization.(qTemp);
	aqClass = mean_normalization.(qClass);
	aqTemp = mean_normalization.(qTemp);
	@info "number of equispaced angles:" angle
	accuracy_k_nearest_neighbour(qTemp, Label, qClass, Labels, "inf", ret=1);
	accuracy_k_nearest_neighbour(qTemp, Label, qClass, Labels, "euclidean", ret=1);
	accuracy_k_nearest_neighbour(mqTemp, Label, mqClass, Labels, "inf", ret=1);
	accuracy_k_nearest_neighbour(mqTemp, Label, mqClass, Labels, "euclidean", ret=1);
	accuracy_k_nearest_neighbour(aqTemp, Label, aqClass, Labels, "inf", ret=1);
	accuracy_k_nearest_neighbour(aqTemp, Label, aqClass, Labels, "euclidean", ret=1);
end

# ╔═╡ Cell order:
# ╟─2a981b3c-2765-49cf-8bbd-038f12904400
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╟─9997f3ac-b425-4881-9d46-579f3abe282b
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
# ╟─47936e86-cfce-409f-8161-7755fff4536e
# ╠═14864b75-d2e6-476a-bf63-5ffffa95a61d
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═fb3629dc-1860-4a96-a75e-2b4402f847fe
# ╟─3ef98e3a-160f-42cc-a22a-1cba512697c0
# ╠═7ebbd156-d1dc-49bb-a7f0-1b949a79ddac
# ╠═99fafed4-247c-4f9a-b945-02c39dd29c9c
# ╠═d6b09c3f-4383-4022-a2b9-3b05b533e3d0
# ╠═33ed1571-93e2-4c1e-9c6b-9dbc962fdaed
# ╠═e6ce8adc-9c8a-4b38-92c6-ff713fdc72ed
# ╟─540c0a0b-ed62-401b-a15d-e0d7d4d6228c
# ╠═274784d5-088b-42d6-b4c2-4491c2f9ec2f
# ╠═5f0a1b08-f2ae-4257-993b-cc3740775a3c
# ╠═5b6596f8-2f44-4624-817c-f2772a0ac250
# ╠═3611d48f-d50c-4b4e-9c89-e78c203f993b
# ╠═1b9dcd42-cb2c-4570-90fd-7592013b409b
# ╠═28ee2767-5a3d-4fac-a213-0df1a2673d05
# ╠═c3dedd73-eebe-4403-b69f-ffa45334f471
# ╠═d8c16a36-dd0b-4621-997b-f7156d3d1f88
# ╠═8d4bacec-6758-49ea-b694-fd1aee22a22b
# ╟─4e7c3707-5ed1-4888-a7f6-e490718f4692
# ╠═ed505ea7-c690-4813-9bc5-569e18c698b8
# ╟─5bc61555-1ce4-49a2-8347-a53d79571760
# ╠═21f9c54c-dd3b-4e00-888f-4c91363d0385
# ╠═f199f6d0-970e-4fb9-b126-e4530f17671a
# ╠═2b4a9e41-ddb8-41c2-bb92-204db41dfa85
# ╠═dbcf0415-09ea-49b1-b637-74f4625d8343
# ╠═29c2d769-1bbb-4d8f-8bd8-874eaea96ac9
# ╠═548356e2-570f-45e8-856c-372afdc890f3

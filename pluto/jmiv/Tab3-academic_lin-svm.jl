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
	Random.seed!(42)
end

# ╔═╡ da709924-bbbc-4233-80f8-03b4c2a22376
md"""
# SSVM 2025 -- Table 2 (right)
This pluto notebook reproduces the numerical experiment
for Table 2 (right) from

- Matthias Beckmann, Robert Beinert, Jonas Bresch, 
  'Max-Normalized Radon Cumulative Distribution
  Transform for Limited Data Classification',
  SSVM 2025.
"""

# ╔═╡ 6f2b943d-ccdf-4bd2-b414-d5aca34cbbbd
I₁ = render(OrbAndCross(Circle(),Star(1)), width=8);

# ╔═╡ 185aaebb-e97a-4c94-95e8-687719a93062
J₁ = extend_image(I₁, (256, 256))

# ╔═╡ 0771415d-ac67-47e1-b8dd-3b184c04cfc5
I₂ = render(OrbAndCross(Square(),Star(4)), width=8);

# ╔═╡ eb76e221-6b89-44d2-9f30-546fe0b326bf
J₂ = extend_image(I₂, (256, 256))

# ╔═╡ fc487267-fc96-4872-8d6f-8506598b1cb3
J = [J₁, J₂]; Label = [1, 5];

# ╔═╡ 5425372d-888c-4c65-97be-339ab8a5a466
Class, Labels = generate_academic_classes(J, Label, class_size=500);

# ╔═╡ 773832af-9099-4dcf-bd1b-c82baaa83424
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.75, 1.25), 
	scale_y = (0.75, 1.25),
	rotate=(-180.0, 180.0), 
	shear_x=(-45.0, 45.0),
	shear_y=(-45.0, 45.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))

# ╔═╡ fb3629dc-1860-4a96-a75e-2b4402f847fe
TClass = A.(Class)

# ╔═╡ 4a677cc5-ccdb-4d5e-9ba6-fdd925184dc9
# ╠═╡ disabled = true
#=╠═╡
R = RadonTransform(851,256,0.0)
  ╠═╡ =#

# ╔═╡ 1374b3cf-0e14-41e9-a2be-901aae16111b
# ╠═╡ disabled = true
#=╠═╡
RCDT = RadonCDT(64, R)
  ╠═╡ =#

# ╔═╡ b12b8ef1-0678-459f-9479-c21969989d42
# ╠═╡ disabled = true
#=╠═╡
qClass = RCDT.(TMLClass)
  ╠═╡ =#

# ╔═╡ 070cc31b-07c9-4120-86b9-24638c3c139a
# ╠═╡ disabled = true
#=╠═╡
mqClass = max_normalization.(qClass)
  ╠═╡ =#

# ╔═╡ 616228c8-1bd3-48a0-8328-7f6ca513f504
# ╠═╡ disabled = true
#=╠═╡
miqClass = maxmin_normalization.(qClass)
  ╠═╡ =#

# ╔═╡ dd0a47b4-9c14-4a5d-9899-8c8001a59b2a
# ╠═╡ disabled = true
#=╠═╡
tvqClass = tv_normalization.(qClass)
  ╠═╡ =#

# ╔═╡ 8b0121f4-9506-4af8-89ee-2fd2932c24a2
md"""
## Nearest Neighbour Classification -- Table 2
Use the nearest neighbour classification
with respect to five randomly chosen representatives
per class
to classify the generated dataset.
The max-normalized RCDT is applied
with different numbers of used angles.
Each experiment is repeated ten times.
"""

# ╔═╡ 348b704c-c1b5-456b-ad6b-d19a5057e84b
# ╠═╡ disabled = true
#=╠═╡
for angle in [2,4,8,16,32,64,128,256]
	R = RadonTransform(851,angle,0.0)
	RCDT = RadonCDT(64, R)
	qClass = RCDT.(TClass)
	mqClass = max_normalization.(qClass)
	miqClass = maxmin_normalization.(qClass)
	tvqClass = tv_normalization.(qClass)
	@info "number of equispaced angles:" angle
	accuracy_nearest_cross_neighbour(mqClass, Labels, "inf")
	accuracy_nearest_cross_neighbour(mqClass, Labels, "euclidean")
	accuracy_nearest_cross_neighbour(miqClass, Labels, "inf")
	accuracy_nearest_cross_neighbour(miqClass, Labels, "euclidean")
	accuracy_nearest_cross_neighbour(tvqClass, Labels, "inf")
	accuracy_nearest_cross_neighbour(tvqClass, Labels, "euclidean")
end
  ╠═╡ =#

# ╔═╡ 5f3194f7-b75c-4562-b475-2e724fccf2b4
for angle in [2,4,8,16]
	R = RadonTransform(301,angle,0.0)
	RCDT = RadonCDT(64, R)
	qClass = RCDT.(TClass)
	mqClass = max_normalization.(qClass)
	miqClass = maxmin_normalization.(qClass)
	tvqClass = tv_normalization.(qClass)
	maqClass = maxabs_normalization.(qClass)
	iaqClass = minabs_normalization.(qClass)
	miaqClass = maxminabs_normalization.(qClass)
	for prop in [11,25,50]
		@info "number of equispaced angles and split:" angle, prop
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, mqClass, Labels)
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, miqClass, Labels)
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, maqClass, Labels)
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, iaqClass, Labels)
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, miaqClass, Labels)
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, tvqClass, Labels)
	end
end

# ╔═╡ b09878e1-5ca1-43a4-8a82-358b5075dac6
for angle in [32,64,128,256]
	R = RadonTransform(301,angle,0.0)
	RCDT = RadonCDT(64, R)
	qClass = RCDT.(TClass)
	mqClass = max_normalization.(qClass)
	miqClass = maxmin_normalization.(qClass)
	tvqClass = tv_normalization.(qClass)
	maqClass = maxabs_normalization.(qClass)
	iaqClass = minabs_normalization.(qClass)
	miaqClass = maxminabs_normalization.(qClass)
	for prop in [11,25,50]
		@info "number of equispaced angles and split:" angle, prop
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, mqClass, Labels)
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, miqClass, Labels)
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, maqClass, Labels)
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, iaqClass, Labels)
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, miaqClass, Labels)
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, tvqClass, Labels)
	end
end

# ╔═╡ f9e1d846-611f-4c3c-9073-11777d660093
for angle in [2,4,8,16,32,64,128,256]
	R = RadonTransform(301,angle,0.0)
	RCDT = RadonCDT(64, R)
	qClass = RCDT.(TClass)
	for prop in [11,25,50]
		@info "number of equispaced angles and split:" angle, prop
		Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, qClass, Labels)
	end
end

# ╔═╡ bd5e8e66-306e-4a9a-869d-f255959ea2fa
for prop in [11,25,50]
	@info "number of equispaced angles and split:" angle, prop
	Random.seed!(42); accuracy_part_svm(20, prop, 500, 2, Array{Float64}.(TClass), Labels)
end

# ╔═╡ Cell order:
# ╟─da709924-bbbc-4233-80f8-03b4c2a22376
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╠═6f2b943d-ccdf-4bd2-b414-d5aca34cbbbd
# ╠═185aaebb-e97a-4c94-95e8-687719a93062
# ╠═0771415d-ac67-47e1-b8dd-3b184c04cfc5
# ╠═eb76e221-6b89-44d2-9f30-546fe0b326bf
# ╠═fc487267-fc96-4872-8d6f-8506598b1cb3
# ╠═5425372d-888c-4c65-97be-339ab8a5a466
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═fb3629dc-1860-4a96-a75e-2b4402f847fe
# ╠═4a677cc5-ccdb-4d5e-9ba6-fdd925184dc9
# ╠═1374b3cf-0e14-41e9-a2be-901aae16111b
# ╠═b12b8ef1-0678-459f-9479-c21969989d42
# ╠═070cc31b-07c9-4120-86b9-24638c3c139a
# ╠═616228c8-1bd3-48a0-8328-7f6ca513f504
# ╠═dd0a47b4-9c14-4a5d-9899-8c8001a59b2a
# ╟─8b0121f4-9506-4af8-89ee-2fd2932c24a2
# ╠═348b704c-c1b5-456b-ad6b-d19a5057e84b
# ╠═5f3194f7-b75c-4562-b475-2e724fccf2b4
# ╠═b09878e1-5ca1-43a4-8a82-358b5075dac6
# ╠═f9e1d846-611f-4c3c-9073-11777d660093
# ╠═bd5e8e66-306e-4a9a-869d-f255959ea2fa

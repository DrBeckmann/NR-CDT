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
	using Images
	using Plots
end

# ╔═╡ c9a1f57f-1874-40e4-b47f-d66f7dd4a064
I₁ = Gray{Float64}.(render(OrbAndCross(Circle(),Star(1))));

# ╔═╡ 237a6d1c-6d30-40a0-8eb6-aa0ae913b6d2
J₁ = DataTransformations.extend_image(I₁, (256, 256))

# ╔═╡ 8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
I₂ = Gray{Float64}.(render(OrbAndCross(Square(),Star(4))));

# ╔═╡ 29d43338-99a6-42ce-9cf6-eee91d3905b8
J₂ = DataTransformations.extend_image(I₂, (256, 256))

# ╔═╡ e6c9fb45-2ec7-4925-bc2c-efbed91caa46
I₃ = Gray{Float64}.(render(Shield(Triangle())));

# ╔═╡ 25220f99-8cbd-4387-b4fd-bb4a0e6fad96
J₃ = DataTransformations.extend_image(I₃, (256, 256))

# ╔═╡ c1959bcd-b5ce-40ae-9c41-142fca3027b6
# J = [J₁, J₂, J₃]; Label = [1, 2, 3];
# J = [J₂, J₃]; Label = [1, 2];
J = [J₁, J₂]; Label = [1, 2];

# ╔═╡ 14864b75-d2e6-476a-bf63-5ffffa95a61d
Class, Labels = DataTransformations.generate_academic_classes(J, class_size=90);

# ╔═╡ 773832af-9099-4dcf-bd1b-c82baaa83424
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.75, 1.25), 
	scale_y = (0.75, 1.25),
	rotate=(-90.0, 90.0), 
	shear_x=(-2.0, 2.0),
	shear_y=(-2.0, 2.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20)
	)

# ╔═╡ fb3629dc-1860-4a96-a75e-2b4402f847fe
# TClass = NormalizedRadonCDT.norm_data(A.(Class))
TClass = A.(Class)

# ╔═╡ 8fb1f5c3-386e-4117-9b87-dedb75c1ae1d
R = RadonTransform(floor(Int,sqrt(2)*256),4,0.0)

# ╔═╡ bbbcd04c-8b4f-4c44-958d-9e4089ada051
RCDT = RadonCDT(floor(Int,sqrt(2)*256), R)

# ╔═╡ 81fd55d8-24df-4047-b235-20468b2c111c
NRCDT = NormRadonCDT(RCDT)

# ╔═╡ 81e32395-78d9-4a5f-b6f0-ba2d6f01c8ee
mNRCDT = MaxNormRadonCDT(RCDT)

# ╔═╡ f956f9c4-ab3d-4c55-9eae-d0e082a9b188
rClass = RCDT.(TClass);

# ╔═╡ 553a0f34-84b4-4997-b00d-c90fdb1ae833
qClass = mNRCDT.(TClass);

# ╔═╡ 6f89e4df-7af6-4fdd-bf50-029b07ca82c2
qTemp = mNRCDT.(J);

# ╔═╡ e1d7ede5-1b44-4186-9b9e-21e6ddd29dda
NormalizedRadonCDT.mNRCDT_quantiles(qTemp, Label, qClass, Labels)

# ╔═╡ 2a6defcb-81de-44fa-936b-d7871a188d1a
NormalizedRadonCDT.classify_flatten_svm(TClass, Labels)

# ╔═╡ 67396c87-3bf4-4189-9a6b-9f861600a425
NormalizedRadonCDT.classify_flatten_svm(rClass, Labels)

# ╔═╡ ad89e85d-92a8-445a-8e6f-14ed1a570d06
NormalizedRadonCDT.classify_flatten_svm(qClass, Labels)

# ╔═╡ 457006d5-ae0d-4ee9-b58d-0c0c87d4b128
# ╠═╡ disabled = true
#=╠═╡
gClass, gLabels = DataTransformations.generate_academic_classes(J, class_size=270);
  ╠═╡ =#

# ╔═╡ c974aee4-ba38-4103-ab3d-b9d1a29066e9
# ╠═╡ disabled = true
#=╠═╡
gTClass = A.(gClass);
  ╠═╡ =#

# ╔═╡ 59720153-0f97-4144-befa-0cff41aee075
# ╠═╡ disabled = true
#=╠═╡
for class_size in [10,30,90,270]
	@info "class size:" class_size
	
	sTClass = gTClass[1:class_size];
	append!(sTClass, gTClass[271:270+class_size]);
	sLabels = gLabels[1:class_size];
	append!(sLabels, gLabels[271:270+class_size]);
	
	NormalizedRadonCDT.classify_flatten_svm(sTClass, sLabels)
	
	for angle in [2,4,8,16]
		R = RadonTransform(floor(Int,sqrt(2)*256),angle,0.0);
		RCDT = RadonCDT(floor(Int,sqrt(2)*256), R);
		NRCDT = NormRadonCDT(RCDT);
		mNRCDT = MaxNormRadonCDT(RCDT);
		rqClass = RCDT.(sTClass);
		mqClass = mNRCDT.(sTClass);
		@info "number of equispaced angles:" angle
		NormalizedRadonCDT.classify_flatten_svm(rqClass, sLabels)
		NormalizedRadonCDT.classify_flatten_svm(mqClass, sLabels)	
	end
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╠═c9a1f57f-1874-40e4-b47f-d66f7dd4a064
# ╠═237a6d1c-6d30-40a0-8eb6-aa0ae913b6d2
# ╠═8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
# ╠═29d43338-99a6-42ce-9cf6-eee91d3905b8
# ╠═e6c9fb45-2ec7-4925-bc2c-efbed91caa46
# ╠═25220f99-8cbd-4387-b4fd-bb4a0e6fad96
# ╠═c1959bcd-b5ce-40ae-9c41-142fca3027b6
# ╠═14864b75-d2e6-476a-bf63-5ffffa95a61d
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═fb3629dc-1860-4a96-a75e-2b4402f847fe
# ╠═8fb1f5c3-386e-4117-9b87-dedb75c1ae1d
# ╠═bbbcd04c-8b4f-4c44-958d-9e4089ada051
# ╠═81fd55d8-24df-4047-b235-20468b2c111c
# ╠═81e32395-78d9-4a5f-b6f0-ba2d6f01c8ee
# ╠═f956f9c4-ab3d-4c55-9eae-d0e082a9b188
# ╠═553a0f34-84b4-4997-b00d-c90fdb1ae833
# ╠═6f89e4df-7af6-4fdd-bf50-029b07ca82c2
# ╠═e1d7ede5-1b44-4186-9b9e-21e6ddd29dda
# ╠═2a6defcb-81de-44fa-936b-d7871a188d1a
# ╠═67396c87-3bf4-4189-9a6b-9f861600a425
# ╠═ad89e85d-92a8-445a-8e6f-14ed1a570d06
# ╠═457006d5-ae0d-4ee9-b58d-0c0c87d4b128
# ╠═c974aee4-ba38-4103-ab3d-b9d1a29066e9
# ╠═59720153-0f97-4144-befa-0cff41aee075

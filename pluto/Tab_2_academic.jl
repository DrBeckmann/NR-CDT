### A Pluto.jl notebook ###
# v0.20.4

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
# J = [J₁, J₂, J₃];
J = [J₂, J₃]; Label = [1,2];

# ╔═╡ 14864b75-d2e6-476a-bf63-5ffffa95a61d
Class, Labels = DataTransformations.generate_academic_classes(J, class_size=10);

# ╔═╡ 773832af-9099-4dcf-bd1b-c82baaa83424
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.75, 1.25), 
	scale_y = (0.75, 1.25),
	rotate=(-45.0, 45.0), 
	shear_x=(-2.0, 2.0),
	shear_y=(-2.0, 2.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))

# ╔═╡ fb3629dc-1860-4a96-a75e-2b4402f847fe
TClass = A.(Class)

# ╔═╡ 8fb1f5c3-386e-4117-9b87-dedb75c1ae1d
R = RadonTransform(256,32,0.0)

# ╔═╡ bbbcd04c-8b4f-4c44-958d-9e4089ada051
RCDT = RadonCDT(5*256, R)

# ╔═╡ 81fd55d8-24df-4047-b235-20468b2c111c
NRCDT = NormRadonCDT(RCDT)

# ╔═╡ 81e32395-78d9-4a5f-b6f0-ba2d6f01c8ee
mNRCDT = MaxNormRadonCDT(RCDT)

# ╔═╡ 553a0f34-84b4-4997-b00d-c90fdb1ae833
qClass = mNRCDT.(TClass);

# ╔═╡ 6f89e4df-7af6-4fdd-bf50-029b07ca82c2
qTemp = mNRCDT.(J);

# ╔═╡ e1d7ede5-1b44-4186-9b9e-21e6ddd29dda
NormalizedRadonCDT.mNRCDT_quantiles(qTemp, Label, qClass, Labels)

# ╔═╡ d3839077-85e3-42b7-ba3b-8b97c720ee9f
NormalizedRadonCDT.mNRCDT_nearest_neighbour(qTemp, Label, qClass, Labels)

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
# ╠═553a0f34-84b4-4997-b00d-c90fdb1ae833
# ╠═6f89e4df-7af6-4fdd-bf50-029b07ca82c2
# ╠═e1d7ede5-1b44-4186-9b9e-21e6ddd29dda
# ╠═d3839077-85e3-42b7-ba3b-8b97c720ee9f

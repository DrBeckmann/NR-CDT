### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 2c63867e-638a-11ef-3da0-a91c57c21253
begin
	import Pkg
	Pkg.activate("..")
	using Revise
	using NormalizedRadonCDT
	using Plots
end

# ╔═╡ b25b513b-b9f1-4572-b576-7c3dfb4f0f11
begin
	circle = NormalizedRadonCDT.TestImages.Circle(0,(1,1))
	image = NormalizedRadonCDT.TestImages.generate(circle)
	heatmap(image, ticks=nothing, border=:none)
end

# ╔═╡ 6590cd1e-3ac5-4bcf-9dfa-b8f024f93820
begin
	triangle = NormalizedRadonCDT.TestImages.Polygon(3, π/2, (1,1))
	image_triangle = NormalizedRadonCDT.TestImages.generate(triangle)
	heatmap(image_triangle, ticks=nothing, border=:none)
end

# ╔═╡ 86d9d7bc-0b0f-42bc-84d4-be8ea1cf4d4f
begin
	square = NormalizedRadonCDT.TestImages.Polygon(4, π/4, (1,1))
	image_square = NormalizedRadonCDT.TestImages.generate(square)
	heatmap(image_square, ticks=nothing, border=:none)
end

# ╔═╡ 978c6537-9d63-4366-8c5c-29555c7d5dfb
begin
	star = NormalizedRadonCDT.TestImages.Star(7, 0, (1,1))
	image_star = NormalizedRadonCDT.TestImages.generate(star)
	heatmap(image_star, ticks=nothing, border=:none)
end

# ╔═╡ 1aabbfed-2b4f-4acb-a287-eddb9f8497e5
begin
	orbandcross = NormalizedRadonCDT.TestImages.OrbAndCross(circle, star, 0, (1,1))
	image_orbandcross = NormalizedRadonCDT.TestImages.generate(orbandcross)
	heatmap(image_orbandcross, ticks=nothing, border=:none)
end

# ╔═╡ 656cab80-1731-4435-906e-a7dd57692ab7
begin
	shield = NormalizedRadonCDT.TestImages.Shield(circle, 0, (1,1))
	image_shield = NormalizedRadonCDT.TestImages.generate(shield)
	heatmap(image_shield, ticks=nothing, border=:none)
end

# ╔═╡ Cell order:
# ╠═2c63867e-638a-11ef-3da0-a91c57c21253
# ╠═b25b513b-b9f1-4572-b576-7c3dfb4f0f11
# ╠═6590cd1e-3ac5-4bcf-9dfa-b8f024f93820
# ╠═86d9d7bc-0b0f-42bc-84d4-be8ea1cf4d4f
# ╠═978c6537-9d63-4366-8c5c-29555c7d5dfb
# ╠═1aabbfed-2b4f-4acb-a287-eddb9f8497e5
# ╠═656cab80-1731-4435-906e-a7dd57692ab7

### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 8cbe0300-edff-11ef-2fad-d3b8cca171a9
begin
	import Pkg
	Pkg.activate("..")
	using Revise
	using NormalizedRadonCDT.TestImages
	using NormalizedRadonCDT.DataTransformations
	using Images
end

# ╔═╡ 8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
I = Gray{Float64}.(render(OrbAndCross(Circle(),Star(5))))

# ╔═╡ d1247be8-1628-43d0-9299-7471ec575c72
DataTransformations.scaling(I, 1.25, 1.25)

# ╔═╡ e4999db1-b28a-407c-9e67-f775a5a943f8
DataTransformations.rotate(I, 45.0)

# ╔═╡ 9ee9c8f4-2c1e-42dd-99d8-76eefd999658
DataTransformations.shear(I, 45.0, 0.0)

# ╔═╡ 373faeda-c158-48ba-889a-9be444f30373
DataTransformations.shear(I, 0.0, -20.0)

# ╔═╡ 0fa3ab95-f04d-439e-befe-a6d429530607
DataTransformations.translate(I, 20, 0)

# ╔═╡ 30e08937-a3f0-4913-91d8-b4fa50bf3de4
DataTransformations.translate(I, 0, 20)

# ╔═╡ 773832af-9099-4dcf-bd1b-c82baaa83424
A = DataTransformations.RandomAffineTransformation(rotate=(-45.0,45.0), shift_x=(-20,20))

# ╔═╡ 02e4f53d-992b-4701-9732-537c8023f1be
A(I)

# ╔═╡ 6bb6eee8-9510-478c-9886-994d45a26a66
A(A(A(I)))

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╠═8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
# ╠═d1247be8-1628-43d0-9299-7471ec575c72
# ╠═e4999db1-b28a-407c-9e67-f775a5a943f8
# ╠═9ee9c8f4-2c1e-42dd-99d8-76eefd999658
# ╠═373faeda-c158-48ba-889a-9be444f30373
# ╠═0fa3ab95-f04d-439e-befe-a6d429530607
# ╠═30e08937-a3f0-4913-91d8-b4fa50bf3de4
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═02e4f53d-992b-4701-9732-537c8023f1be
# ╠═6bb6eee8-9510-478c-9886-994d45a26a66

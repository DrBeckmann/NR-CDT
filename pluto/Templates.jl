### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 2c63867e-638a-11ef-3da0-a91c57c21253
# ╠═╡ show_logs = false
begin
	import Pkg
	Pkg.activate("..")
	using NormalizedRadonCDT.TestImages
end

# ╔═╡ bde12216-84b7-4029-808d-b184b6aae11e
md"""
# Test Images
The submodule `TestImages` allows to render different kind 
of basic and composed watermark-like shapes.
The shapes itself are implementet as `struct`
and are rendered as grey-level image
using the funciton `render`.
"""

# ╔═╡ 28f16420-6b87-43cb-b807-b4dd8ac908ff
md"""
## Basic Shapes
The submodule covers the following basic shapes:
- the `Circle`,
- the regular `Polygon`,
- and the regular `Star`.
Moreover,
the constructors `Triangle` and `Square` can be used
to construct specific regular polygons.
"""

# ╔═╡ b25b513b-b9f1-4572-b576-7c3dfb4f0f11
render(Circle())

# ╔═╡ 6590cd1e-3ac5-4bcf-9dfa-b8f024f93820
render(Triangle())

# ╔═╡ 86d9d7bc-0b0f-42bc-84d4-be8ea1cf4d4f
render(Square())

# ╔═╡ 7817f1c0-dadb-4ca6-b65e-3aefdb8e187b
render(Polygon(7))

# ╔═╡ 978c6537-9d63-4366-8c5c-29555c7d5dfb
render(Star(8))

# ╔═╡ 3c0c4c0c-bc27-4db5-94c0-fe9c02b5b86a
md"""
## Composed Shapes
Using the base shapes,
we can render the following composed shapes:
- the `OrbAndCross`
- and the `Shield` with emblem.
The components or emblems can be freely choosen
from the base shapes.
Drawing a single component can be avoided
by using the `Empty` shape.
"""

# ╔═╡ 1aabbfed-2b4f-4acb-a287-eddb9f8497e5
render(OrbAndCross(Circle(), Star(6)))

# ╔═╡ 656cab80-1731-4435-906e-a7dd57692ab7
render(Shield(Circle()))

# ╔═╡ Cell order:
# ╟─bde12216-84b7-4029-808d-b184b6aae11e
# ╠═2c63867e-638a-11ef-3da0-a91c57c21253
# ╟─28f16420-6b87-43cb-b807-b4dd8ac908ff
# ╠═b25b513b-b9f1-4572-b576-7c3dfb4f0f11
# ╠═6590cd1e-3ac5-4bcf-9dfa-b8f024f93820
# ╠═86d9d7bc-0b0f-42bc-84d4-be8ea1cf4d4f
# ╠═7817f1c0-dadb-4ca6-b65e-3aefdb8e187b
# ╠═978c6537-9d63-4366-8c5c-29555c7d5dfb
# ╟─3c0c4c0c-bc27-4db5-94c0-fe9c02b5b86a
# ╠═1aabbfed-2b4f-4acb-a287-eddb9f8497e5
# ╠═656cab80-1731-4435-906e-a7dd57692ab7

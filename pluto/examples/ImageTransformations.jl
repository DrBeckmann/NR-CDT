### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 8cbe0300-edff-11ef-2fad-d3b8cca171a9
begin
	import Pkg
	Pkg.activate("../..")
	using NormalizedRadonCDT.TestImages
	using NormalizedRadonCDT.DataTransformations
end

# ╔═╡ 20ce72b6-e372-4bc7-ba0e-cb63d32538dd
md"""
# Data Transformations
The submodule `DataTransformations` implements
- random affine transformations
- mikado noise
- salt noise
- bar noise
- elastic noise
for gray-valued images.
All Transformations are implemented as callable `struct`.
"""

# ╔═╡ c5dae360-774d-4c6a-a947-6fcbb7f65376
md"""
## Test Image
As test image,
we use the orb-and-cross shape 
from the submodule `TestImages`. 
"""

# ╔═╡ 8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
I = render(OrbAndCross(Circle(),Star(5)))

# ╔═╡ a6de063d-60dd-4e54-a795-ade268877b8f
md"""
## Random Affine Transformations
To calculate a random affine transformations,
we apply
1. a random anisotrop scaling
1. a random rotation
1. a random shearing in x-direction
1. a random shearing in y-direction
1. a random circular shift
The transformation itself is implemented as callable `struct`,
where the parameters
for the single transformations 
are given as tuples,
which indicate intervals
from which the actual parameter is randomly chosen.
"""

# ╔═╡ 773832af-9099-4dcf-bd1b-c82baaa83424
A = RandomAffineTransformation(rotate=(-45.0,45.0), shift_x=(-20,20), shear_y=(-20.0,20.0))

# ╔═╡ 7a720703-2d53-4458-a50f-153c618daf97
md"""
The random affine transformation can be directly applied to an image.
"""

# ╔═╡ 02e4f53d-992b-4701-9732-537c8023f1be
A(I)

# ╔═╡ 2caab0b1-f066-45f5-b77f-64323c44728e
md"""
## Mikado Noise
The mikado noise consists of short line segments
that are randomly added to an image.
For a fixed parameter set,
the noise is initialized via `MikadoNoise`.
The parameters consists of tuples
that indicate the range
where the actual parameter are chosen from.
"""

# ╔═╡ b342fa1b-5f2b-4985-afbf-e486c9c8f248
N = MikadoNoise(sticks=(10,15), length=(0.125,0.25), width=(1.5,2.5))

# ╔═╡ 26845920-3283-4210-b696-3f9fb7944b2f
md"""
To disturb an image,
the `struct` can be directly applied.
"""

# ╔═╡ de663f10-7987-4793-be78-52bc4f087fbb
N(I)

# ╔═╡ f08b2d07-eaf3-44ef-a42c-852fff1c17bd
md"""
## Salt Noise
The salt noise consists of a series of small dots
that are randonly added to an image.
For a fixed parameter set,
the noise is initialized via `SaltNoise`.
The parameter consits of tuples
that indicate the range 
from which the actual parameter are randomly chosen.
"""

# ╔═╡ fc0fb8f7-fc6a-4f72-9416-876edc07544f
S = SaltNoise(dots=(10,15), width=(3/128, 4/128))

# ╔═╡ 0b02c558-e684-484e-9564-fc75e06455a1
md"""
To disturb an image,
the `struct` can be directly applied.
"""

# ╔═╡ d5c7e146-db95-4b60-bc29-c2deae79e4bd
S(I)

# ╔═╡ bd182ab8-a38e-4e5a-a9a3-df3c5e70a6b9
md"""
## Bar Noise
The bar noise consists vertical black lines
that are randonly added to an image.
For a fixed parameter set,
the noise is initialized via `BarNoise`.
The parameter consits of tuples
that indicate the range 
from which the actual parameter are randomly chosen.
"""

# ╔═╡ 6ad576a6-3ade-4d81-912b-cf567b8c2f2d
B = BarNoise(bars=(2,6), width=(2.0, 3.0))

# ╔═╡ 9669eae5-7b31-44b3-8844-c9165348e053
md"""
To disturb an image,
the `struct` can be directly applied.
"""

# ╔═╡ 83b76372-30af-48e8-9cde-09d8d4ded334
B(I)

# ╔═╡ 972c65ea-1673-4ce2-a32d-65d921953f6c
md"""
## Elastic Noise
To simulate an small non-linear image transformations,
the columns and rows are circularly shifted
with respect to sine and cosine functions.
For a fixed parameter set,
the noise is initialized via `ElasticNoise`.
The parameter consits of tuples
that indicate the range 
from which the actual parameter are randomly chosen.
"""

# ╔═╡ 2349961b-3a2d-46a3-9225-84786050cbef
E = ElasticNoise(amplitude_x=(15.0, 15.0), amplitude_y=(5.0, 5.0))

# ╔═╡ 70a4fdc5-fd97-400e-a15a-2aed5e3932cd
md"""
To disturb an image,
the `struct` can be directly applied.
"""

# ╔═╡ b9d24f03-fe34-4b62-8709-b84b0b01c3c4
E(I)

# ╔═╡ a8e7dd91-dbef-4eb5-ab22-f3dd398e6f81
md"""
## Combining Noise Models
The single noise models and transformations can be applied 
in arbitrary order.
"""

# ╔═╡ 9b1fe40b-f34c-4cc3-99fa-29b2b72b1df2
N(S(B(A(E(I)))))

# ╔═╡ Cell order:
# ╟─20ce72b6-e372-4bc7-ba0e-cb63d32538dd
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╟─c5dae360-774d-4c6a-a947-6fcbb7f65376
# ╠═8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
# ╟─a6de063d-60dd-4e54-a795-ade268877b8f
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╟─7a720703-2d53-4458-a50f-153c618daf97
# ╠═02e4f53d-992b-4701-9732-537c8023f1be
# ╟─2caab0b1-f066-45f5-b77f-64323c44728e
# ╠═b342fa1b-5f2b-4985-afbf-e486c9c8f248
# ╟─26845920-3283-4210-b696-3f9fb7944b2f
# ╠═de663f10-7987-4793-be78-52bc4f087fbb
# ╟─f08b2d07-eaf3-44ef-a42c-852fff1c17bd
# ╠═fc0fb8f7-fc6a-4f72-9416-876edc07544f
# ╟─0b02c558-e684-484e-9564-fc75e06455a1
# ╠═d5c7e146-db95-4b60-bc29-c2deae79e4bd
# ╟─bd182ab8-a38e-4e5a-a9a3-df3c5e70a6b9
# ╠═6ad576a6-3ade-4d81-912b-cf567b8c2f2d
# ╟─9669eae5-7b31-44b3-8844-c9165348e053
# ╠═83b76372-30af-48e8-9cde-09d8d4ded334
# ╟─972c65ea-1673-4ce2-a32d-65d921953f6c
# ╠═2349961b-3a2d-46a3-9225-84786050cbef
# ╟─70a4fdc5-fd97-400e-a15a-2aed5e3932cd
# ╠═b9d24f03-fe34-4b62-8709-b84b0b01c3c4
# ╟─a8e7dd91-dbef-4eb5-ab22-f3dd398e6f81
# ╠═9b1fe40b-f34c-4cc3-99fa-29b2b72b1df2

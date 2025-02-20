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

# ╔═╡ b342fa1b-5f2b-4985-afbf-e486c9c8f248
N = DataTransformations.MikadoNoise((10,15), (0.125,0.25), (1.5,2.5))

# ╔═╡ de663f10-7987-4793-be78-52bc4f087fbb
N(I)

# ╔═╡ fc0fb8f7-fc6a-4f72-9416-876edc07544f
S = DataTransformations.SaltNoise((10,15), (3/128, 4/128))

# ╔═╡ d5c7e146-db95-4b60-bc29-c2deae79e4bd
S(I)

# ╔═╡ 6ad576a6-3ade-4d81-912b-cf567b8c2f2d
B = DataTransformations.BarNoise((2,6),(2.0, 3.0))

# ╔═╡ 83b76372-30af-48e8-9cde-09d8d4ded334
B(I)

# ╔═╡ 32ed4efd-3ecc-4ec5-a448-4340fcb42ed9
N(S(B(A(I))))

# ╔═╡ 2349961b-3a2d-46a3-9225-84786050cbef
E = DataTransformations.ElasticNoise(amplitude_x=(15.0, 15.0), amplitude_y=(5.0, 5.0))

# ╔═╡ b9d24f03-fe34-4b62-8709-b84b0b01c3c4
E(I)

# ╔═╡ 9b1fe40b-f34c-4cc3-99fa-29b2b72b1df2
N(S(B(A(E(I)))))

# ╔═╡ dbff450c-5836-462a-97ab-13a7137677db
J = DataTransformations.extend_image(I, (256, 128))

# ╔═╡ 7c59be83-d395-4283-a541-07fd5381ff49
N(S(B(A(E(J)))))

# ╔═╡ 62d921a9-ebdd-4d38-b8c0-df2be57134a6
I₁ = Gray{Float64}.(render(OrbAndCross(Circle(),Star(4)), width=4.0))

# ╔═╡ 649d8324-d91b-4295-8198-1f425f4e6acf
I₂ = Gray{Float64}.(render(Shield(Circle()), width=4.0))

# ╔═╡ 7f4a48e3-be97-450b-abe2-dabc62401ed7
J₁ = DataTransformations.extend_image(I₁, (256, 256))

# ╔═╡ c171a7a4-0815-4a2a-ac10-b8b799e2fc0f
J₂ = DataTransformations.extend_image(I₂, (256, 256))

# ╔═╡ e31b9e0b-ad11-431d-ae7d-1dd4b12a5b6a
JA = [J₁, J₂];

# ╔═╡ 24b8d1a6-216c-4614-b742-3e795ec747b9
JClass, JLabel = DataTransformations.generate_academic_classes(JA, 10);

# ╔═╡ 0593b84b-35fa-4ce3-9277-df1ba94a87de
A.(JClass)

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
# ╠═b342fa1b-5f2b-4985-afbf-e486c9c8f248
# ╠═de663f10-7987-4793-be78-52bc4f087fbb
# ╠═fc0fb8f7-fc6a-4f72-9416-876edc07544f
# ╠═d5c7e146-db95-4b60-bc29-c2deae79e4bd
# ╠═6ad576a6-3ade-4d81-912b-cf567b8c2f2d
# ╠═83b76372-30af-48e8-9cde-09d8d4ded334
# ╠═32ed4efd-3ecc-4ec5-a448-4340fcb42ed9
# ╠═2349961b-3a2d-46a3-9225-84786050cbef
# ╠═b9d24f03-fe34-4b62-8709-b84b0b01c3c4
# ╠═9b1fe40b-f34c-4cc3-99fa-29b2b72b1df2
# ╠═dbff450c-5836-462a-97ab-13a7137677db
# ╠═7c59be83-d395-4283-a541-07fd5381ff49
# ╠═62d921a9-ebdd-4d38-b8c0-df2be57134a6
# ╠═649d8324-d91b-4295-8198-1f425f4e6acf
# ╠═7f4a48e3-be97-450b-abe2-dabc62401ed7
# ╠═c171a7a4-0815-4a2a-ac10-b8b799e2fc0f
# ╠═e31b9e0b-ad11-431d-ae7d-1dd4b12a5b6a
# ╠═24b8d1a6-216c-4614-b742-3e795ec747b9
# ╠═0593b84b-35fa-4ce3-9277-df1ba94a87de

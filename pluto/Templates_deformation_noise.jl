### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 03f9300e-692e-11ef-145c-85ecce1e4c7f
begin
	import Pkg
	Pkg.activate("..")
	using Revise
	using NormalizedRadonCDT.TestImages
	using NormalizedRadonCDT
end

# ╔═╡ ea97ef35-3aa2-421c-a919-0c775266dbd5
using JLD, Images, Plots

# ╔═╡ 7a7eadb9-56b7-4145-940e-8e660aff12f8
using PythonOT

# ╔═╡ e2ba028e-6346-4a9b-8bd3-9223a9c537d9
templates = load("temp.jld")["temp"];

# ╔═╡ 43c617ec-3abd-42ef-80ef-62cc80d4dee7
label = range(1, size(templates)[1])

# ╔═╡ cd3f1705-772f-41d0-a37e-ba5cd345712d
temp, lab = NormalizedRadonCDT.gen_temp_ext([1,6], templates, label, 128)

# ╔═╡ 144852d9-16d8-46f7-81a2-300e1d8e3c30
temp_img = temp[1,:,:]; img_plot=plot(Gray.(temp_img))

# ╔═╡ 10196a6b-ae6b-4999-a9f8-a8761e8cd8b7
savefig(img_plot, "orig_img.pdf")

# ╔═╡ 4e012794-7f50-4e38-884f-65f4458c2f94
# temp_dis = NormalizedRadonCDT.nonlinear_distortion(temp_img, 10.0, 2.0);
# temp_dis = NormalizedRadonCDT.impulsive_distortion(temp_img, 1.0, 10, 10);
# temp_dis = NormalizedRadonCDT.temp_distortion(temp_img, [[10.0, 2.0], [2.0, 10, 10],[0]]);
# temp_dis = NormalizedRadonCDT.temp_distortion(temp_img, [[0], [0],[1]]);
# temp_dis = temp[2,:,:];
temp_dis = NormalizedRadonCDT.random_image_distortion(temp_img, 256, [0,0], [0,0], [0,0], [0,0], [0,0], [5,20,2,20], 42)

# ╔═╡ d9ae0610-cd8a-4cdb-b74a-49a307547409
dis_plot=plot(Gray.(temp_dis))

# ╔═╡ 4c3fa402-ac0f-41f7-a432-c285a56dbbd9
savefig(dis_plot, "dis_img.pdf")

# ╔═╡ 27fea2a5-2389-4cc5-b1a8-ce2f9ea17f97
nk_temp_img = findall(x->x>1e-6, temp_img); mk_temp_dis = findall(x->x>1e-6, temp_dis); 

# ╔═╡ 01afc537-717b-4af1-a7db-e9859d6a7acb
temp_img_s = temp_img[nk_temp_img]; temp_dis_s = temp_dis[mk_temp_dis];

# ╔═╡ c26e759e-30c0-4e80-a220-86343a4b1162
temp_img_v = vec(temp_img_s); temp_dis_v = vec(temp_dis_s);

# ╔═╡ 4d93083c-64a8-4746-8236-79f8cc897525
temp_img_flat = temp_img_v/sum(temp_img_v); temp_dis_flat = temp_dis_v/sum(temp_dis_v);

# ╔═╡ 6b228522-2498-425c-8d70-7de7473d6570
n = size(temp_img_flat)[1]; m = size(temp_dis_flat)[1];

# ╔═╡ 071901d2-fedf-4d3a-a270-7abe23f4d733
C = zeros(n,m);

# ╔═╡ a699c2aa-efad-43a3-9bfb-f62867089dab
for i in 1:n
	# for j in 1:n
		for k in 1:m
			# for l in 1:m
				a = nk_temp_img[i][1]/256
				b = nk_temp_img[i][2]/256
				c = mk_temp_dis[k][1]/256
				d = mk_temp_dis[k][2]/256
				C[i, k] = (a - c)^2 + (b - d)^2
			# end
		end
	# end
end

# ╔═╡ 7d197833-aee6-4012-8eaf-e597891bb476
P = emd(temp_img_flat, temp_dis_flat, C);

# ╔═╡ 80690413-7709-4556-920f-eaa7006ebc7f
heatmap(C/maximum(C))

# ╔═╡ 3ef25658-b833-4950-9575-241f54b532f5
cP = emd2(temp_img_flat, temp_dis_flat, C)

# ╔═╡ abe4accd-4819-4be2-9969-2bfdba580be9
l_x = []; l_y = []; l_u = []; l_v = []; l_s = []; l_w = [];

# ╔═╡ 42583760-0733-4638-bc77-9bb021585a89
for i in 1:n
	# for j in 1:m
		# if temp_img[i,j] > 1e-6
			xy_pos = findall(x->x>1e-6, P[i,:])
			s_xy_pos = size(xy_pos)[1]
			for l in 1:s_xy_pos
				append!(l_s, s_xy_pos)
				
				append!(l_x, nk_temp_img[i][1])
				append!(l_y, nk_temp_img[i][2])

				append!(l_u, mk_temp_dis[xy_pos[l]][1])
				append!(l_v, mk_temp_dis[xy_pos[l]][2])

				append!(l_w, P[i,xy_pos[l]])
			end
		# end
	# end
end

# ╔═╡ f24e1609-bc89-447b-ad74-ca6767a953e3
q = quiver(l_y,256*ones(size(l_v)[1])-l_x, quiver=(l_v-l_y,-l_u+l_x), width=1.2*l_w/maximum(l_w), 
	# marker=(:circle, 0.1, :green), 
	aspect_ratio=:equal,
	# la=1
	la=0.1,
	color=:gray,
	dpi=250
	)

# ╔═╡ cfabffe2-3b9a-4858-8ed3-d79bcc08936b
savefig(q, "movement_particals.pdf")

# ╔═╡ 7fe07976-f180-4c47-a3b6-cf636901ebe9
s = scatter(l_y, 256*ones(size(l_x)[1])-l_x, color=:green, ms=1, ma=0.2, label = "original image");

# ╔═╡ bad075bd-35f7-43ab-87b5-6017df2d5b8b
plot(scatter(s, l_v, 256*ones(size(l_u)[1])-l_u, color=:red, ms=1, ma=0.2, label = "disturbed image"));

# ╔═╡ 648d5829-352f-4a3f-ba9a-6803d4065f27
qq = quiver!(l_y, 256*ones(size(l_x)[1])-l_x, quiver=(l_v-l_y, -l_u+l_x), width=1.2*l_w/maximum(l_w), 
	# marker=(:circle, 0.1, :green), 
	aspect_ratio=:equal,
	# la=1
	color=:gray,
	la=0.2,
	dpi=250
	)

# ╔═╡ 1418d1c6-b027-428a-a508-ad556c2c551d
savefig(qq, "movement_img_particals.pdf")

# ╔═╡ Cell order:
# ╠═03f9300e-692e-11ef-145c-85ecce1e4c7f
# ╠═ea97ef35-3aa2-421c-a919-0c775266dbd5
# ╠═e2ba028e-6346-4a9b-8bd3-9223a9c537d9
# ╠═43c617ec-3abd-42ef-80ef-62cc80d4dee7
# ╠═cd3f1705-772f-41d0-a37e-ba5cd345712d
# ╠═144852d9-16d8-46f7-81a2-300e1d8e3c30
# ╠═10196a6b-ae6b-4999-a9f8-a8761e8cd8b7
# ╠═4e012794-7f50-4e38-884f-65f4458c2f94
# ╠═d9ae0610-cd8a-4cdb-b74a-49a307547409
# ╠═4c3fa402-ac0f-41f7-a432-c285a56dbbd9
# ╠═27fea2a5-2389-4cc5-b1a8-ce2f9ea17f97
# ╠═01afc537-717b-4af1-a7db-e9859d6a7acb
# ╠═c26e759e-30c0-4e80-a220-86343a4b1162
# ╠═4d93083c-64a8-4746-8236-79f8cc897525
# ╠═6b228522-2498-425c-8d70-7de7473d6570
# ╠═071901d2-fedf-4d3a-a270-7abe23f4d733
# ╠═a699c2aa-efad-43a3-9bfb-f62867089dab
# ╠═7a7eadb9-56b7-4145-940e-8e660aff12f8
# ╠═7d197833-aee6-4012-8eaf-e597891bb476
# ╠═80690413-7709-4556-920f-eaa7006ebc7f
# ╠═3ef25658-b833-4950-9575-241f54b532f5
# ╠═abe4accd-4819-4be2-9969-2bfdba580be9
# ╠═42583760-0733-4638-bc77-9bb021585a89
# ╠═f24e1609-bc89-447b-ad74-ca6767a953e3
# ╠═cfabffe2-3b9a-4858-8ed3-d79bcc08936b
# ╠═7fe07976-f180-4c47-a3b6-cf636901ebe9
# ╠═bad075bd-35f7-43ab-87b5-6017df2d5b8b
# ╠═648d5829-352f-4a3f-ba9a-6803d4065f27
# ╠═1418d1c6-b027-428a-a508-ad556c2c551d

require 'cunn'
require 'nn'
require 'gnuplot'
require 'image'
dofile './provider.lua'
-- require 'itorch'

local loss = torch.Tensor(50000,92)


for i=1,92 do
	loss[{{},{i}}] = torch.load('tensors/loss_'..i..'.t7')
end

local pr = torch.Tensor(50000,92)
for i=1,92 do
	pr[{{},{i}}] = torch.load('tensors/prediction_'..i..'.t7')
end
-- -- p = 0
-- -- for i = 1,50000 do
-- -- 	if pr[i][56] == 0 then
-- -- 		p = i
-- -- 		break
-- -- 	end
-- -- end 
-- -- gnuplot.plot(pr[p])
-- -- gnuplot.plot(t[p])
-- -- print(p,t[p][20])

-- --here pr is the predictions tensor, and t is the losses tensor  

-- local slope = torch.Tensor(50000):zero()

-- --calculating the averages of X and Y i.e Xb and Yb
-- Xb = 46.5
-- Xb = torch.Tensor(50000):fill(Xb)
-- Yb = torch.sum(loss:clone(),2) --Yb has 50000 dimension
-- Yb = torch.div(Yb,92)


-- -- print({Yb})

-- for i = 1,50000 do
-- 	local numer = 0
-- 	local denom = 0
	
-- 	for j = 1,92 do
-- 		-- print({(loss[i][j]-Yb[i])})
-- 		numer  = numer + (j-Xb[i])*(loss[i][j]-Yb[i]) 

-- 		denom = denom + (j-Xb[i])*(j-Xb[i])
-- 	end
-- 	-- print({numer})
	
-- 	slope[i] = numer/denom
-- end


-- torch.save('tensors/slope.t7',slope)

slope = torch.load('tensors/slope.t7')
val , ind = torch.sort(slope)
torch.save('tensors/indices.t7',ind)
print(val, ind[1], ind[-1])
gnuplot.plot(loss[ind[-1]])
-- print(c.blue '==>' ..' loading data')
provider = torch.load 'provider.t7'
print(provider.trainData.labels[ind[-1]])
-- image.display(provider.trainData.data[ind[-1]])
-- gnuplot.imagesc(provider.trainData.data[ind[1]])

-- provider.trainData.data[]
require 'cunn'
require 'nn'

-- torch.setdefaulttensortype('torch.CudaTensor')
local t = torch.Tensor(200):fill(0)
local model = nn.Sequential()
model:add(nn.Linear(200,100,true)):add(nn.Tanh()):add(nn.Linear(100,50,true)):add(nn.Tanh()):add(nn.Linear(50,2,true)):add(nn.SoftMax())

for k,v in pairs(model:findModules('nn.Linear')) do
	print({v})
	v.bias:fill(0)
	if k == 3 then
		v.bias:fill(2)
	end
	v.weight:normal(0,0.1)
end
-- model = model:cuda()
local output = model:forward(t)

print(output)

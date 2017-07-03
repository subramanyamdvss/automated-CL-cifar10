require 'xlua'
require 'optim'
require 'nn'
dofile './provider.lua'
require 'cudnn'
require 'cunn'
require 'image'
local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "logs3/vgg")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 1)       learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default vgg_bn_drop)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default cudnn)            backend
   --type                     (default cuda)          cuda/float/cl
   --startfrom                (default 0)             from which epoch should I start the training
   --eps                      (default 0.05)            epsilon for greedy policy 
   --eta                      (default 0.001)           eta during updating weights
   --beta                     (default 0)               beta while calculating reward bar
   --loadprev                 (default 0)          load previous  epoch
]]


print(opt)

do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i]:float(), input[i]:float()) end
      end
    end
    self.output:set(input:float())
    return self.output
  end
end

local function cast(t)
   if opt.type == 'cuda' then
      require 'cunn'
      return t:cuda()
   elseif opt.type == 'float' then
      return t:float()
   elseif opt.type == 'cl' then
      require 'clnn'
      return t:cl()
   else
      error('Unknown type '..opt.type)
   end
end


-- local model = nn.Sequential()
-- model:add(nn.BatchFlip():float())
-- model:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
-- model:add(cast(dofile('models/'..opt.model..'.lua')))
-- model:get(2).updateGradInput = function(input) return end

-- if opt.backend == 'cudnn' then
--    require 'cudnn'
--    cudnn.benchmark=true
--    cudnn.convert(model:get(3), cudnn)
-- end
local model = nn.Sequential()
if  paths.filep('/home/surya/cifar-10/cifar.torch/'..opt.save..'/model_'..opt.startfrom..'.net')  then
  model:add(nn.BatchFlip():float())
  model:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
  print('loading model from' ,'/home/surya/cifar-10/cifar.torch/'..opt.save..'/model_'..opt.startfrom..'.net')
  local net = torch.load('/home/surya/cifar-10/cifar.torch/'..opt.save..'/model_'..opt.startfrom..'.net')
  model:add(cast(net))
  model:get(2).updateGradInput = function(input) return end
  -- model:read(torch.DiskFile('/home/surya/cifar-10/cifar.torch/logs/vgg/model_37.net'))
  print(c.blue '==>' ..' configuring model')
  if opt.backend == 'cudnn' then
    require 'cudnn'
    cudnn.benchmark=true
    cudnn.convert(model:get(3), cudnn)
  end
  print(model)
else
  opt.startfrom = 0
  model:add(nn.BatchFlip():float())
  model:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
  model:add(cast(dofile('models/'..opt.model..'.lua')))
  model:get(2).updateGradInput = function(input) return end

  if opt.backend == 'cudnn' then
    require 'cudnn'
    cudnn.benchmark=true
    cudnn.convert(model:get(3), cudnn)
  end
  print(model)
end
print(c.blue '==>' ..' loading data')
provider = torch.load 'provider.t7'
provider.trainData.data = provider.trainData.data:cuda():index(1,torch.load('tensors/indices.t7'):long())
provider.testData.data = provider.testData.data:cuda()
-- provider.trainData.data = cast(provider.trainData.data)
-- provider.testData.data = cast(provider.testData.data)
confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model:getParameters()


print(c.blue'==>' ..' setting criterion')
criterion = cast(nn.CrossEntropyCriterion())


print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}


local poli = cast(torch.Tensor(50)):zero()
local wv = cast(torch.Tensor(50)):zero()

if paths.filep('tensors2/wv_'..opt.startfrom..'.t7') then
  wv = torch.load('tensors2/wv_'..opt.startfrom..'.t7'):cuda()
end
local polf = nn.Sequential()
polf:add(nn.SoftMax())
polf:add(nn.MulConstant(1-opt.eps)):add(nn.AddConstant(opt.eps/50))
polf = polf:cuda()
cudnn.convert(polf,cudnn)



function policy(t)
  local probs = polf:forward(wv)
  poli = probs
  -- print(probs)
  local gumb = torch.rand(50):double():cuda()
  gumb = -torch.log(-torch.log(gumb))
  probs:log()

  val , ind = torch.max((probs+gumb),1)
  -- if ind == nil then ind = 27 end
  -- print('in policy',ind,val , probs)
  -- print('index', ind[1])
  -- print('in policy',probs[1])
  return {val , ind}
end

function modifywv(t,reward,ind,prob)
  prob = prob[1]
  ind = ind[1]
  -- print(prob)
  local rbart =  torch.CudaTensor(50):fill(opt.beta/prob)
  rbart[ind] = (opt.beta+reward)/prob
  local wvt = wv
  local temp = torch.exp(wvt+opt.eta*rbart)
  local sum = temp:sum()
  local alpha = 1/t
  local const = alpha/(50-1)
  local wvtp = (1-50*const)*temp + const*sum
  -- print('in modifywv',wv)
  wv = torch.log(wvtp) 
  -- print('modifywv',wv[t+1])
end

local reservoir = torch.CudaTensor(5000):zero()
local ql = -1;
local qh = 1;
local reward = 0;
local indices;



if paths.filep('tensors2/indices_'..opt.startfrom..'.t7') then
  indices = torch.load('tensors2/indices_.t7'):long():cuda():split(1000)

else
  indices = torch.randperm(provider.trainData.data:size(1)):long()
  torch.save('tensors2/indices_.t7',indices)
  indices = torch.load('tensors2/indices_.t7'):long():cuda():split(1000)
end


function train(ep)
  model:training()
  epoch = epoch or opt.startfrom+1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local targets = cast(torch.FloatTensor(opt.batchSize))
  
  -- local indices = torch.Tensor():long():split(1000)
  -- remove last element so that all the batches have equal size
  -- indices[#indices] = nil

  local tic = torch.tic()
  for t = 1,390 do
    xlua.progress(t, 390)
    local probt = policy(t)
    local prob = probt[1]
    -- print('prob',{prob})
    local ind = probt[2]
    local indmask = torch.randperm(1000):long()
    -- indices[ind] = indices[ind]
    local tmp = torch.CudaTensor(1000)
    -- print(indices[1])
      -- print({ind})
    tmp = indices[ind[1]]
    -- print(#indices[ind[1]])
    tmp = tmp:index(1,indmask):long():split(opt.batchSize)
    -- print(tmp[1])
    local inputs = provider.trainData.data:index(1,tmp[1])
    targets:copy(provider.trainData.labels:index(1,tmp[1]))
    -- print(targets)
    collectgarbage()
    local prevl;
    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      f = criterion:forward(outputs, targets)
      prevl = f
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      confusion:batchAdd(outputs, targets)

      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)
    local outputs = model:forward(inputs)
    local diff = prevl - criterion:forward(outputs, targets)
    local rcap = diff
    if 390*(ep-1)+t <= 5000 then
      if 390*(ep-1)+t == 1 or 390*ep+t == 2  then
        reward = rcap
        if rcap >=1 then
          reward = 1
        
        elseif rcap <= -1 then
          reward = -1
        end
        ql = -1
        qh = 1
        reservoir[390*(ep-1)+t] = rcap
      end
      if 390*(ep-1)+t >= 3 then
        reward = 2*(rcap-ql)/(qh-ql) - 1
        if rcap >=qh then
          reward = 1
        
        elseif rcap <= ql then
          reward = -1
        end
        local temp = torch.Tensor((390*(ep-1)+t))
        reservoir[390*(ep-1)+t] = rcap
        temp = reservoir[{{1,(390*(ep-1)+t)}}]
        val, indx = torch.sort(temp)
        reservoir[{{1,(390*(ep-1)+t)}}] = val
        ql = reservoir[math.floor((390*(ep-1)+t)*4/5)]
        qh = reservoir[math.floor((390*(ep-1)+t)*2/5)]
        
      end
      
    
    else 
      reward = 2*(rcap-ql)/(qh-ql) - 1
      if rcap>=qh then
        reward = 1
      elseif rcap <= qh then
        reward = -1
      end
      local idx = torch.randperm(390*(ep-1)+t)[1]
      if idx <= 5000 then
        reservoir[idx] = rcap
      end
      val, indx = torch.sort(reservoir)
      reservoir = val
      qh = reservoir[4000]
      ql = reservoir[2000]
      
    end
  -- print('reward bro ' ,reward,'qh',qh,'ql',ql)
  modifywv(t,reward,ind,prob)
  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  confusion:zero()
  epoch = epoch + 1

end


function test(ep)
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = 250
  for i=1,provider.testData.data:size(1),bs do
    local outputs = model:forward(provider.testData.data:narrow(1,i,bs))
    confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
  end

  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, confusion.totalValid * 100}
    testLogger:style{'-','-'}
    testLogger:plot()

    if paths.filep(opt.save..'/test.log.eps') then
      local base64im
      do
        os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
        os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
        local f = io.open(opt.save..'/test.base64')
        if f then base64im = f:read'*all' end
      end

      local file = io.open(opt.save..'/report.html','w')
      file:write(([[
      <!DOCTYPE html>
      <html>
      <body>
      <title>%s - %s</title>
      <img src="data:image/png;base64,%s">
      <h4>optimState:</h4>
      <table>
      ]]):format(opt.save,epoch,base64im))
      for k,v in pairs(optimState) do
        if torch.type(v) == 'number' then
          file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
        end
      end
      file:write'</table><pre>\n'
      file:write(tostring(confusion)..'\n')
      file:write(tostring(model)..'\n')
      file:write'</pre></body></html>'
      file:close()
    end
  end

  -- save model every 50 epochs
  if epoch % 1 == 0 then
    local filename = paths.concat(opt.save, 'model_'..ep..'.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(3):clearState())
    print('==> saving model to '..'tensors2/indices'..ep..'.t7')
    print('tensors2/wv_'..ep..'.t7')
  end

  confusion:zero()
end

function savepred(ep)
  model:evaluate()
  local correct = 0
  print('saving predictions of training set...')
  local bs = 250
  local pred = torch.Tensor(50000):zero()
  local loss = torch.Tensor(50000):zero()
  local probs = torch.Tensor(50000,10):zero()
  for i=1,provider.trainData.data:size(1),bs do
    xlua.progress(i,provider.trainData.data:size(1))
    local outputs = model:forward(provider.trainData.data:narrow(1,i,bs)) --outputs have size bsx10 so 
    probs[{{i,i+bs-1}}] = outputs:double()
    -- print({probs[{{i,i+bs-1}}]}) 
    -- print({outputs})
    for k = 1,bs do
      loss[i+k-1] = criterion:forward(outputs[k],provider.trainData.labels[i+k-1])
    end
    -- confusion:batchAdd(outputs, provider.trainData.labels:narrow(1,i,bs))
    y,ind = torch.max(outputs,2)
    -- print({y},{ind})
    for j = i,i+bs-1 do
      if ind[j-i+1][1]== provider.trainData.labels[j] then
        pred[j] = 1
        correct = correct + 1
      end
    end
  end
  print('train accuracy:', 100*correct/50000 )
  torch.save('./tensors/prediction_'.. ep ..'.t7' ,pred)
  torch.save('./tensors/loss_'.. ep ..'.t7',loss)
  torch.save('./tensors/probs_'.. ep ..'.t7',loss)
end

for ep=opt.startfrom+1,opt.max_epoch do
  
  train(ep)
  -- print('policy probabilities', poli)
  -- print('weights',wv)
  test(ep)


  -- savepred(ep) 
  
end



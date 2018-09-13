import numpy as np 
import os 
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

from torch.autograd.variable import Variable

class Model(nn.Module):

    def __init__(self):
	super(Model, self).__init__()
        #self.affine1 = nn.Linear(39, 128)
        self.affine1 = nn.Linear(29, 128) #for prep_state2
	self.affine2 = nn.Linear(128,128)
        self.affine3 = nn.Linear(128, 9)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
    	x = F.relu(self.affine1(x))
    	x = F.relu(self.affine2(x))
	action_scores = self.affine3(x)
        #print("ACTION",action_scores)
        #action_scores = F.normalize(action_scores)
        return action_scores


model = Model()
def prepare_train_X():

	cur_folder = os.getcwd()
	with open('State_info_trail_no2','rb') as fp:
		itemlist = pickle.load(fp)

	return itemlist



def prepare_train_Y():
    move_list = [(1,1) , (1,-1) , (1, 0) , (0,1) , (0,-1),(0,0) , (-1,1),(-1,0),(-1,-1)]
    trainY = []
    with open('Trial_no_2','rb') as fp:

    	itemlist = pickle.load(fp)
	#print("Itemlist",itemlist)
	for i in itemlist:

       	    move_arr = np.zeros(9)
            i = i/10
            tup = (i[0],i[1])
            for j in range(9):
          	if move_list[j]==tup:
                    break
            #move_arr[j] = 1
            trainY.append(j)
    return trainY

trainx = prepare_train_X()
trainy = prepare_train_Y()

print(trainy)

def train(train_x, train_y,batch_size = 200):
	fig = plt.figure()
	plt.clf()
	#print(type(train_x))
	#print(type(train_y))
	xs = []
	ys = []
	counter = 0
	loss_fn = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(),lr=1e-2)
	for epoch in range(39000):
		#print("Starting epoch :",epoch)
		running_loss = 0.0
		deno = 1
		for i in range(0,len(train_x),batch_size):
			counter+=1
			if (i+batch_size)<len(train_x):
				mini_x = train_x[i:i+batch_size]
				mini_y = train_y[i:i+batch_size]
				deno = batch_size
			else:
				mini_x = train_x[i:len(train_x)]
				mini_y = train_y[i:len(train_x)]
				deno = len(train_x)-i
			y_pred = model(mini_x)
			#print(deno)
			#print(type(y_pred))
			#print(type(train_y))
			#print(y_pred)
			#print(mini_y)
			#break
			loss = loss_fn(y_pred,mini_y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
        		if counter % 2000 == 0:    # print every 2000 mini-batches
            			print('[%d, %5d] loss: %.5f' %
                  		(epoch + 1, i + 1, running_loss / deno))
            			
				xs.append(counter)
            			ys.append(running_loss/deno)
				running_loss = 0.0
            			plt.plot(xs,ys,color='green')
            			plt.draw()
            			plt.title("Loss:")
				plt.pause(.0001)
            			fig.show()
		if epoch+1%13000==0:
			torch.save(model.state_dict(),'./stored_models/supervised/episode_{}.pth'.format(epoch))
	print("finished training")
        torch.save(model.state_dict(),'./stored_models/supervised/episode_{}.pth'.format(epoch))
	print("Model saved!!")

trainx = prepare_train_X()
l = len(trainx)
train_nparr = np.zeros((l,trainx[0].shape[0]))
for i in range(l):
    train_nparr[i] = trainx[i]
train_X = torch.from_numpy(train_nparr).float()


trainy = prepare_train_Y()
l = len(trainy)
train_np_yarr = np.zeros((l))
for i in range(l):
    train_np_yarr[i] = trainy[i]
train_Y = torch.from_numpy(train_np_yarr).long()
print(train_Y.size())
#for i in train_Y:
	#print i
train(train_X,train_Y)












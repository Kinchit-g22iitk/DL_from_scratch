import numpy as np
import os
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
file_1 = '/content/drive/MyDrive/cifar-10-batches-py/data_batch_1'
data_batch_1 = unpickle(file_1)
data_batch_1.keys()
file_2 = '/content/drive/MyDrive/cifar-10-batches-py/data_batch_1'
data_batch_2 = unpickle(file_2)
data_batch_2.keys()
file_3 = '/content/drive/MyDrive/cifar-10-batches-py/data_batch_1'
data_batch_3 = unpickle(file_3)
data_batch_3.keys()
file_4 = '/content/drive/MyDrive/cifar-10-batches-py/data_batch_1'
data_batch_4 = unpickle(file_4)
data_batch_4.keys()
file_5 = '/content/drive/MyDrive/cifar-10-batches-py/data_batch_1'
data_batch_5 = unpickle(file_5)
data_batch_5.keys()
file = '/content/drive/MyDrive/cifar-10-batches-py/test_batch'
data_test = unpickle(file)
data_batch_1[b'data'] = np.reshape(data_batch_1[b'data'],(10000,32,32,3))
data_batch_2[b'data'] = np.reshape(data_batch_2[b'data'],(10000,32,32,3))
data_batch_3[b'data'] = np.reshape(data_batch_3[b'data'],(10000,32,32,3))
data_batch_4[b'data'] = np.reshape(data_batch_4[b'data'],(10000,32,32,3))
data_batch_5[b'data'] = np.reshape(data_batch_5[b'data'],(10000,32,32,3))
x_1 = data_batch_1[b'data']
x_2 = data_batch_2[b'data']
x_3 = data_batch_3[b'data']
x_4 = data_batch_4[b'data']
x_5 = data_batch_5[b'data']
y_1 = data_batch_1[b'labels']
y_2 = data_batch_2[b'labels']
y_3 = data_batch_3[b'labels']
y_4 = data_batch_4[b'labels']
y_5 = data_batch_5[b'labels']
y_t = data_test[b'labels']
y = y_1+y_2+y_3+y_4+y_5
con = np.concatenate((x_1,x_2,x_3,x_4,x_5),axis=0)
con = np.reshape(con,(50000,3,32,32))
con.shape
test = data_test[b'data']
test = np.reshape(test,(10000,3,32,32))
(train_X,train_y),(test_X,test_y) = (con,y),(test,y_t)
import random
train_X = train_X/255
test_X = test_X/255
train_data=list(zip(train_X,train_y))
test_data=list(zip(test_X,test_y))
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
def sigmoid(z):
    z = 1/(1+np.exp(-z))
    return z
def tanh(z):
    z = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    return z
class Network():
    def __init__(self,l):
        w = np.ones((3,32,32))
        b = np.zeros((32,32))
        weights = []
        bias = []
        weights.append(w)
        bias.append(b)
        weights.append(np.random.randn(l[0],32**2))
        bias.append(np.random.randn(l[0],1))
        for i in range(len(l)-1):
            weights.append(np.random.randn(l[i+1],l[i]))
            bias.append(np.random.randn(l[i+1],1))
        weights.append(np.random.randn(10,l[-1]))
        bias.append(np.random.randn(10,1))
        self.weights = weights
        self.bias = bias
        self.l = l
        path1 = './expt'
        path2 = './save'
        if not os.path.exists(path1):
            os.mkdir(path1)
        if not os.path.exists(path2):
            os.mkdir(path2)
        file_path_1 = os.path.join(path1, "log_train.txt")
        file_path_2 = os.path.join(path1, "log_test.txt")
        file_path_3 = os.path.join(path2, "parameters.txt")
        self.file_path_1 = file_path_1
        self.file_path_2 = file_path_2
        self.file_path_3 = file_path_3
        
    def flatten(self,x):
        sum = np.zeros((x.shape[1],x.shape[1]))
        for i in range(3):
            sum += np.matmul(x[i],self.weights[0][i].T)
        sum += self.bias[0]
        sum = sum/3
        a = sum.reshape((sum.shape[0]**2,1))
        self.a = a
        return a
    def forwardpropagation(self,a):
        a = self.flatten(a)
        for b,w in zip(self.bias[1:], self.weights[1:]):
            a=sigmoid(np.matmul(w,a)+b)
            # print(a.shape)
        return a

    def backpropagation(self,x,y,fun):
        if fun == "sigmoid":
        
            y_t = np.zeros((len(y), 10))
            y_t[np.arange(len(y)), y] = 1
            y_t= y_t.T
            
            nabla_b=[np.zeros(b.shape) for b in self.bias]
            nabla_w=[np.zeros(w.shape) for w in self.weights]
            x = self.flatten(x)
   
            activation=x
            activation_list=[x]
            for w,b in zip(self.weights[1:],self.bias[1:]):
                activation= sigmoid(np.matmul(w,activation)+b)
                activation_list.append(activation)            
            delta = (activation_list[-1]-y_t)*activation_list[-1]*(1-activation_list[-1]) #dc/dz3
            m=len(y_t)    
            nabla_w[-1]=np.matmul(delta,activation_list[-2].T)
            
            nabla_b[-1] = delta   
    

            for j in range(2,(len(self.l))+2):
                sig_der = activation_list[-j]*(1-activation_list[-j])
                delta= np.matmul(self.weights[-j+1].T,delta)*sig_der
                
                nabla_b[-j]= (delta)
                nabla_w[-j]= np.matmul(delta,activation_list[-j-1].T)
    
            return (nabla_b,nabla_w)
        elif fun == "tanh":
            y_t = np.zeros((len(y), 10))
            y_t[np.arange(len(y)), y] = 1
            y_t= y_t.T
           
            # Doing one hot encoding in these lines
        
            nabla_b=[np.zeros(b.shape) for b in self.bias]
            nabla_w=[np.zeros(w.shape) for w in self.weights]
            x = self.flatten(x)
            activation=x
            activation_list=[x]

            for w,b in zip(self.weights[1:],self.bias[1:]):
                activation= tanh(np.matmul(w,activation)+b)
                activation_list.append(activation)

            
            delta = (activation_list[-1]-y_t)*(1-activation_list[-1]**2) #dc/dz3
            m=len(y_t)

            nabla_w[-1]=np.matmul(delta,activation_list[-2].T)
            
            nabla_b[-1] = delta
            for j in range(2,(len(self.l))+2):
                sig_der = (1-activation_list[-j]**2)
                delta= np.matmul(self.weights[-j+1].T,delta)*sig_der

                
                nabla_b[-j]= (delta)
                nabla_w[-j]= np.matmul(delta,activation_list[-j-1].T)
            return (nabla_b,nabla_w)
    def update_mini_batch_gd(self,mini_batch):
        nabla_b=[np.zeros(b.shape) for b in self.bias]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        for (x,y) in mini_batch:
            (delta_b,delta_w) = self.backpropagation(x,y,self.fun)
            nabla_b=[nb + db for nb,db in zip(nabla_b,delta_b)]
            nabla_w=[nw + dw for nw,dw in zip(nabla_w,delta_w)]

        self.weights=[w- self.lr*nw/len(mini_batch) for w,nw in zip(self.weights,nabla_w)]
        self.bias=[b-self.lr*nb/len(mini_batch) for b,nb in zip(self.bias,nabla_b)]
    def update_mini_batch_mom(self,mini_batch):
        nabla_b=[np.zeros(b.shape) for b in self.bias]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        delta_w=[np.zeros(w.shape) for w in self.weights]
        delta_b=[np.zeros(b.shape) for b in self.bias]

        i = 0 
        for (x,y) in mini_batch:
            i += 1
            (delta_b_i,delta_w_i) = self.backpropagation(x,y,self.fun)
            delta_b = [nb+db for nb,db in zip(delta_b,delta_b_i)]                 
            delta_w = [nw+dw for nw,dw in zip(delta_w,delta_w_i)]  
            if i%self.mini_batch_size==0:               
                nabla_b=[self.beta*nb + (self.lr*db)/i for nb,db in zip(nabla_b,delta_b)]
                nabla_w=[self.beta*nw + (self.lr*dw)/i for nw,dw in zip(nabla_w,delta_w)]
                self.weights=[w- nw for w,nw in zip(self.weights,nabla_w)]  #averaging over many example
                self.bias=[b-nb for b,nb in zip(self.bias,nabla_b)]
    def update_mini_batch_nag(self, mini_batch):
        delta_b = [np.zeros(b.shape) for b in self.bias]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        i  = 0
        
        for (x, y) in mini_batch:
            (delta_b, delta_w) = self.backpropagation(x, y,self.fun)
            i += 1
            
            nabla_b = [self.beta * nb + (self.lr * db) / i for nb, db in zip(nabla_b, delta_b)]
            nabla_w = [self.beta * nw + (self.lr * dw) / i for nw, dw in zip(nabla_w, delta_w)]
            
            if i % self.mini_batch_size == 0:
                self.weights = [w - self.beta * vw for w, vw in zip(self.weights, nabla_w)]
                self.bias = [b - self.beta * vb for b, vb in zip(self.bias, nabla_b)]
                
                (delta_b, delta_w) = self.backpropagation(x, y, self.fun)
                nabla_b = [self.beta * nb + (self.lr * db) / i for nb, db in zip(nabla_b, delta_b)]
                nabla_w = [self.beta * nw + (self.lr * dw) / i for nw, dw in zip(nabla_w, delta_w)]
                
                self.weights = [w - nw for w, nw in zip(self.weights, nabla_w)]  # averaging over many examples
                self.bias = [b - nb for b, nb in zip(self.bias, nabla_b)] 
    def update_mini_batch_adam(self,mini_batch,beta1=0.9,beta2=0.999,epsi=1e-8):
        i = 0
        delta_b = [np.zeros(b.shape) for b in self.bias]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        acc_w = [np.zeros(w.shape) for w in self.weights]
        acc_b = [np.zeros(b.shape) for b in self.bias]
        for (x,y) in mini_batch:
            (delta_b,delta_w) = self.backpropagation(x,y,self.fun)
            i+=1
            acc_w = [beta1*nw + (1-beta1)*dw for nw,dw in zip(nabla_w,delta_w)]
            acc_b = [beta1*nb + (1-beta1)*db for nb,db in zip(nabla_b,delta_b)]
            nabla_w = [beta2*nw + (1-beta2)*np.power(dw,2) for nw,dw in zip(nabla_w,delta_w)]
            nabla_b = [beta2*nb + (1-beta2)*np.power(db,2) for nb,db in zip(nabla_b,delta_b)]
            if i%self.mini_batch_size==0:
                acc_w = [(1.0/(1.0 - np.power(beta1,i)))*mw for mw in acc_w]
                acc_b = [(1.0/(1.0 - np.power(beta1,i)))*mb for mb in acc_b]
                nabla_w = [(1.0/(1.0-np.power(beta2,i)))*nw for nw in nabla_w]
                nabla_b = [(1.0/(1.0-np.power(beta2,i)))*nb for nb in nabla_b]
                for i in range(len(self.weights)):
                    self.weights[i] -= (self.lr/np.sqrt(nabla_w[i]+epsi))*acc_w[i]
                    self.bias[i] -= (self.lr/np.sqrt(nabla_b[i]+epsi))*acc_b[i]
    def anneal(self,lr):
        lr = lr/2
    def cross_entropy_loss(self,y,y_hat):
        sum = 0
        for i in range(len(y)):
            sum -= (y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
        return sum
    def squared_error_loss(self,y,y_hat):
        sum = 0
        for i in range(len(y)):
            sum += (y-y_hat)**2
        return sum
    def training_batch_loss(self,loss,batch):
        sum = 0
        for (x,y) in batch:
            self.flatten(x)
            for y_hat in self.forward_propagation(x):
                if loss=="sq":
                    sum+= self.squared_error_loss(y,y_hat)
                elif loss == "ce":
                    sum+= self.cross_entropy_loss(y,y_hat)
        return sum
    def validation_loss(self,loss,batch):
        sum = 0
        for (x,y) in batch:
            for y_hat in self.forward_propagation(x):
                if loss=="sq":
                    sum+= self.squared_error_loss(y,y_hat)
                elif loss == "ce":
                    sum+= self.cross_entropy_loss(y,y_hat)
        return sum           
    def mini_batch(self,optimizer):
        n_train= len(self.train_data)
        for i in range(self.epochs):
            np.random.shuffle(self.train_data)
            mini_batches = [self.train_data[k:k+ self.mini_batch_size] for k in range(0,n_train,self.mini_batch_size)]
            if optimizer == "gd":
                for mini_batch in mini_batches:
                    self.update_mini_batch_gd(mini_batch)
            elif optimizer == "momentum":
                for mini_batch in mini_batches:
                    self.update_mini_batch_mom(mini_batch)
            elif optimizer == "nag":
                for mini_batch in mini_batches:
                    self.update_mini_batch_nag(mini_batch)
            elif optimizer=="adam":
                for mini_batch in mini_batches:
                    self.update_mini_batch_adam(mini_batch)
            num = 0 
            
            for j in range(len(self.train_data)):
                a = self.predict(self.train_data[j])
                if a==1:
                    num += 1
                if j+1%100 == 0:
                    with open(self.file_path_1,'a') as f:
                        f.write(f"Epoch {i+1}, Step {j+1}, Loss: {self.training_batch_loss(self.loss,self.train_data[j-99:j+1])}, Error:{(1-num/j+1)*100}, lr: {self.lr}\n")

            if self.anneal == True:
                self.anneal(self.lr)
        with open(self.file_path_3,'a') as f:
            f.write(f"Weigths arrays for different layers {self.weights[1:]} \n and Biases arrays for different layers {self.bias[1:]}")
        
    def optimization(self,train_data,epochs,optimizer,mini_batch_size,lr,beta,fun,loss,test_data,anneal):
        self.train_data = train_data
        self.epochs = epochs
        self.optimizer = optimizer
        self.mini_batch_size = mini_batch_size
        self.lr = lr
        self.fun = fun
        self.loss = loss
        self.beta = beta
        self.test_data = test_data
        self.anneal = anneal
        self.mini_batch(self.optimizer)
        self.test_log()
    def predict(self,test_data):
        
        test_results = [(np.argmax(self.forwardpropagation(x)),y) for x,y in test_data]
        
        num = sum(int (x==y) for (x,y) in test_results)
        return num
    def test_log(self):
        num = 0
        for j in range(len(self.test_data)+1):
            a = self.predict(self.test_data[j])
            if a==1:
                num += 1
            if j+1%100 ==0:
                with open(self.file_path_2,'a') as f:
                    f.write(f"Step {j+1}, Loss: {self.validation_batch_loss(self.loss,self.test_data[j-99:j+1])}, Error:{(1-num/j+1)*100}, lr: {self.lr}\n")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate for gradient descent based algorithms')
parser.add_argument('--momentum', type=float, default=0.5, help='momentum to be used by momentum based algorithms')
parser.add_argument('--num_hidden', type=int, default=3, help='number of hidden layers')
parser.add_argument('--sizes', type=int, default=[100,100,100],nargs='+', help='comma separated list for the size of each hidden layer')
parser.add_argument('--activation', choices=['tanh', 'sigmoid'], default='sigmoid', help='choice of activation function')
parser.add_argument('--loss', choices=['sq', 'ce'], default='sq', help='choice of loss function')
parser.add_argument('--opt', choices=['gd', 'momentum', 'nag', 'adam'], default='adam', help='optimization algorithm to be used')
parser.add_argument('--batch_size', type=int, choices=[1, 5, 10, 15, 20], default=20, help='batch size to be used')
parser.add_argument('--anneal', type=bool, default=True, help='whether to halve the learning rate if validation loss decreases')
parser.add_argument('--save_dir', type=str, default='pa1/', help='directory to save the pickled model')
parser.add_argument('--expt_dir', type=str, default='pa1/exp1/', help='directory to save log files')
parser.add_argument('--train', type=str, default='train.csv', help='path to the training dataset')
parser.add_argument('--test', type=str, default='test.csv', help='path to the test dataset')

args = parser.parse_args()
print(args.sizes)
if __name__== '__main__':
    # sizes = list(map(int, args.sizes.split(',')))
    sizes = args.sizes
    model = Network(sizes)
    # model = Network(args.sizes)
    model.optimization(train_data,10,args.opt,args.batch_size,args.lr,args.momentum,args.activation,args.loss,test_data,args.anneal)
    
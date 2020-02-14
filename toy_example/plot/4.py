import numpy as np
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import matplotlib.pyplot as pltlegend
from tqdm import tqdm

x0=0
y0=0
Converge_condtion=-1


def gx(x,y):
	return  (2 * x * (y ** 2))
def gy(x,y):
	return  (2 * (x ** 2) * y )
def f(x,y):
	return  (x ** 2) * (y ** 2)


 

def dist(x,y):
	return (x - x0) ** 2 + (y - y0) **2



def GDA_step(x,y,eta):
	x_n= np.clip(x + eta * gx(x,y), -2, 2)
	y_n= np.clip(y - eta * gy(x,y), -2, 2)
	return x_n,y_n


def SGLD_step(x,y,eta,psi):
	x_n= np.clip(x + eta * gx(x,y)+np.sqrt(2 * eta) * np.random.normal(0,1,1)*psi, -2, 2)
	y_n= np.clip(y - eta * gy(x,y)+np.sqrt(2 * eta) * np.random.normal(0,1,1)*psi, -2, 2)
	return x_n,y_n

def SGLD_DA_step(x,y,eta,psi,beta,k):
	x_=x
	y_=y
	x_b=x
	y_b=y

	for i in range(k):
		x__, y__ = SGLD_step (x_,y_,eta,psi)
		x_b= (1-beta) * x_b + beta * x__
		y_b= (1-beta) * y_b + beta * y__
		x_=x__
		y_=y__

	x_n= (1-beta)* x + beta * x_b
	y_n= (1-beta)* y + beta * y_b

	return x_n, y_n 


def GDA(x,y,eta,Max_iteration_time):
	dlist=np.zeros(Max_iteration_time)
	dlistx=np.zeros(Max_iteration_time)
	dlisty=np.zeros(Max_iteration_time)
	d=dist(x,y)
	dlist[0]=f(x,y)
	dlistx[0]=x
	dlisty[0]=y
	for i in range(Max_iteration_time-1):
		x,y=GDA_step(x,y,eta)
		d=dist(x,y)
		dlistx[i+1]=x
		dlisty[i+1]=y
		d=f(x,y)
		dlist[i+1]=d
	return dlist,dlistx,dlisty

def SGLD_DA(x,y,eta=0.1,psi=0.1,beta=0.9,k=50,Max_iteration_time=1000):
	dlist=np.zeros(Max_iteration_time)
	dlistx=np.zeros(Max_iteration_time)
	dlisty=np.zeros(Max_iteration_time)
	d=dist(x,y)
	dlist[0]=f(x,y)
	dlistx[0]=x
	dlisty[0]=y
	for i in range(Max_iteration_time-1):
		x,y= SGLD_DA_step(x,y,eta,psi,beta,k)
		d=dist(x,y)
		dlistx[i+1]=x
		dlisty[i+1]=y
		d=f(x,y)
		dlist[i+1]=d
	return dlist,dlistx,dlisty




def OGDA_step(x,y,x_o,y_o,eta):
	x_n=np.clip(x+2*eta*gx(x,y)-eta*gx(x_o,y_o),-2, 2)
	y_n=np.clip(y-2*eta*gy(x,y)+eta*gy(x_o,y_o),-2, 2)

	return x_n,y_n

def OGDA(x,y,eta,Max_iteration_time):
	dlist=np.zeros(Max_iteration_time)
	dlistx=np.zeros(Max_iteration_time)
	dlisty=np.zeros(Max_iteration_time)
	d=dist(x,y)
	dlist[0]=d
	dlistx[0]=x
	dlisty[0]=y
	x_o=x
	y_o=y

	for i in range(Max_iteration_time-1):
		x_n,y_n= OGDA_step(x,y,x_o,y_o,eta)
		x_o=x
		y_o=y
		x=x_n
		y=y_n
		d=dist(x,y)
		dlistx[i+1]=x
		dlisty[i+1]=y
		d=f(x,y)
		dlist[i+1]=d
		#print(x,y,d)
	return dlist,dlistx,dlisty



def EG_step(x,y,eta,c=2):
	x_h=np.clip(x+eta*gx(x,y),-c, c)
	y_h=np.clip(y-eta*gy(x,y),-c, c)
	x_n=np.clip(x+eta*gx(x_h,y_h),-c, c)
	y_n=np.clip(y-eta*gy(x_h,y_h),-c, c)
	return x_n,y_n


def EG(x,y,eta,Max_iteration_time):
	dlist=np.zeros(Max_iteration_time)
	dlistx=np.zeros(Max_iteration_time)
	dlisty=np.zeros(Max_iteration_time)
	d=dist(x,y)
	dlist[0]=f(x,y)
	dlistx[0]=x
	dlisty[0]=y
	print(gx(x,y),gy(x,y),x,y)
	for i in range(Max_iteration_time-1):
		x,y= EG_step(x,y,eta)
		
		d=dist(x,y)
		#print(gx(x,y),gy(x,y),x,y,f(x,y))

		dlistx[i+1]=x
		dlisty[i+1]=y
		d=f(x,y)
		dlist[i+1]=d
	return dlist,dlistx,dlisty













if __name__ == '__main__':

	N=1
	Max_iteration_time=200
	xlist=range(Max_iteration_time)
	d_GDA=np.zeros([N,Max_iteration_time])
	d_OGDA=np.zeros([N,Max_iteration_time])
	d_EG=np.zeros([N,Max_iteration_time])
	d_SGLD1=np.zeros([N,Max_iteration_time])
	d_SGLD2=np.zeros([N,Max_iteration_time])
	for ii in range(N):
		np.random.seed(ii+487)
		x_init=np.random.uniform(0,0.5,1)
		y_init=np.random.uniform(0,0.5,1)
		x_init=x_init+np.random.choice([-0.5,0.5])
		y_init=y_init+np.random.choice([-0.5,0.5])
		#x_init=0.92628269
		#y_init=0.94097207
		#x_init=0.2
		#y_init=1.2
		print(x_init,y_init)
		dlist,x1,y1=GDA(x_init,y_init,0.05,Max_iteration_time)
		d_GDA[ii,:]=dlist
		dlist,x2,y2=OGDA(x_init,y_init,0.05,Max_iteration_time)
		d_OGDA[ii,:]=dlist
		#dlist,x3,y3=EG(x_init,y_init,0.47,Max_iteration_time)
		dlist,x3,y3=EG(x_init,y_init,0.1,Max_iteration_time)
		d_EG[ii,:]=dlist
		
		dlist,x4,y4=SGLD_DA(x_init,y_init,eta=0.04,psi=0.1,beta=1,k=1,Max_iteration_time=Max_iteration_time)
		d_SGLD1[ii,:]=dlist

		dlist,x5,y5=SGLD_DA(x_init,y_init,eta=0.1,psi=0.01,beta=0.5,k=100,Max_iteration_time=Max_iteration_time)
		d_SGLD2[ii,:]=dlist


	x=np.arange(-100,1000,1)
	y=x*0 
	plt1.plot(x,y,'--',label='NE',color='cornflowerblue',linewidth=2)

	plt1.plot(xlist,np.mean(d_GDA,axis=0),label='GDA',marker=">",color='C1',markevery=5)
	plt1.plot(xlist,np.mean(d_EG,axis=0),label='EG',marker=">",color='blue',markevery=5)
	plt1.plot(xlist,np.mean(d_SGLD2,axis=0),label='MixedNE-LD',marker=">",color='red',markevery=5)

	plt1.ylabel(r'$\mathrm{f}(\omega,\theta)$')
	plt1.xlabel('t')
	
	
	plt1.plot(0,f(x_init,y_init),'.',marker='o',markersize=10,label='Start',color='black')
	plt1.xlim(-10, 210)
	plt1.savefig('./maxminx2y2_1.pdf')
	plt1.show()

#	pltlegend.figure(figsize = (3.6,0.5))

	#pltlegend.figlegend(*ax.get_legend_handles_labels(),ncol=3,loc ='center')
	#pltlegend.savefig('legend.pdf')


	#pltlegend.show()


	#plt3.plot(xlist,x1,'-',label='GDA-x',color='green')
	#plt3.plot(xlist,y1,'--',label='GDA-y',color='green')
	#plt3.plot(xlist,x3,'-',label='EG-x',color='blue')
	#plt3.plot(xlist,y3,'--',label='EG-y',color='blue')
	#plt3.plot(xlist,x5,'-',label='MixedNE-LD-x',color='red')
	#plt3.plot(xlist,y5,'--',label='MixedNE-LD-y',color='red')
	#plt3.title(r'$\max_x \min_y \mathrm{f}(x,y)=x^2y^2$')
	#plt3.xlabel('step')
	#plt3.legend()
	#plt3.savefig('./maxminx2y2_2_2.pdf')
	
	#plt3.show()

	xx = np.arange(-0.7,-0.3,0.08)
	yy = np.arange(0,0.8,0.2)
	X, Y = np.meshgrid(xx, yy)
	u = gx(X,Y)
	v = -gy(X,Y)
	plt2.quiver(X,Y,u,v,scale=10,color='gray',width=0.005,minlength=2,alpha=0.5)





	x=np.arange(-0.75,-0.25,0.05)
	y=0*x 
	plt2.plot(x,y,'--',label=r'$y=0$',color='cornflowerblue')



	plt2.plot(x1,y1,'-',label='GDA',color='C1')
	d=1
	x_=x1[:-1][0::d]
	y_=y1[:-1][0::d]
	x=x1[1:][0::d]
	y=y1[1:][0::d]
	plt2.quiver(x1[:-1], y1[:-1], x1[1:]-x1[:-1], y1[1:]-y1[:-1],scale_units='xy', angles='xy',scale=1,headwidth=5,color='C1')
	#plt2.quiver(x1[:-1], y1[:-1], x1[1:]-x1[:-1], y1[1:]-y1[:-1],scale_units='xy', angles='xy',scale=1,headwidth=5,color='C1')
	plt2.plot(x3,y3,'-',label='EG',color='blue')


	x_=x3[:-1][0::d]
	y_=y3[:-1][0::d]
	x=x3[1:][0::d]
	y=y3[1:][0::d]
	plt2.quiver(x3[:-1], y3[:-1], x3[1:]-x3[:-1], y3[1:]-y3[:-1],scale_units='xy', angles='xy',scale=1,headwidth=5,color='blue')

	plt2.plot(x5,y5,'-',label='MixedNE-LD',color='red')
	plt2.plot(x_init,y_init,'.',marker='o',markersize=10,label='Start',color='black')

	x_=x5[:-1][0::d]
	y_=y5[:-1][0::d]
	x=x5[1:][0::d]
	y=y5[1:][0::d]
	plt2.quiver(x5[:-1], y5[:-1], x5[1:]-x5[:-1], y5[1:]-y5[:-1], scale_units='xy', angles='xy',scale=1,headwidth=5,color='red')


	
	#plt2.title(r'$\max_x \min_y \mathrm{f}(x,y)=x^2y^2$')
	plt2.ylabel(r'$\theta$')
	plt2.xlabel(r'$\omega$')
	#plt2.legend()

	plt2.savefig('./maxminx2y2_2.pdf')
	plt2.show()


	



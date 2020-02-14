import os


inp=input("This script is for plotting the toy example in 'Robust Reinforcement Learning via Adversarial training with Langevin Dynamics'.\n\
For Figure 1. (a)(b), please enter 1.\n\
For Figure 1. (c)(d), please enter 2.\n\
For Figure 2. (a)(b), please enter 3.\n\
For Figure 2. (c)(d), please enter 4.\n\
For Figure 3. (a)(b), please enter 3.\n\
For Figure 3. (c)(d), please enter 4.\n")

try:
	f=open("./plot/"+inp+".py")
except IOError:
   print ("Invalid input")
else:
	f.close()
	os.system("python ./plot/"+inp+".py")
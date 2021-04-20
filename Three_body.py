
import numpy as np
#import matplotlib and associated modules for 3D andanimations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


#Define universal gravitation constant
G=6.67408e-11 #N-m2/kg2
#Reference quantities
m_nd=1.989e+30 #kg #mass of the sun
r_nd=5.326e+12 #m #distance between stars in Alpha Centauri
v_nd=30000 #m/s #relative velocity of earth around the sun
t_nd=79.91*365*24*3600*0.51 #s #orbital period of Alpha Centauri
#Net constants
K1=G*t_nd*m_nd/(r_nd**2*v_nd)
K2=v_nd*t_nd/r_nd

#Define masses
m1=1.1 #Alpha Centauri A
m2=0.907 #Alpha Centauri B
m3=1.0

#Define initial position vectors
r1=[-0.5,0,0] #m
r2=[0.5,0,0] #m
r3=[0,1,0]

#Convert pos vectors to arrays
r1=np.array(r1,dtype="float64")
r2=np.array(r2,dtype="float64")
r3=np.array(r3,dtype="float64")

#Find Centre of Mass
r_com=(m1*r1+m2*r2+m3*r3)/(m1+m2+m3)

#Define initial velocities
v1=[0.01,0.01,0] #m/s
v2=[-0.05,0,-0.1] #m/s
v3=[0,-0.01,0]

#Convert velocity vectors to arrays
v1=np.array(v1,dtype="float64")
v2=np.array(v2,dtype="float64")
v3=np.array(v3,dtype="float64")


#Find velocity of COM
v_com=(m1*v1+m2*v2+m3*v3)/(m1+m2+m3)
#A function defining the equations of motion
def TwoBodyEquations(w,t,G,m1,m2):
    r1=w[:3]
    r2=w[3:6]
    v1=w[6:9]
    v2=w[9:12]
    r=np.linalg.norm(r2-r1) #Calculate magnitude or norm of vector
    dv1bydt=K1*m2*(r2-r1)/r**3
    dv2bydt=K1*m1*(r1-r2)/r**3
    dr1bydt=K2*v1
    dr2bydt=K2*v2
    r_derivs=np.concatenate((dr1bydt,dr2bydt))
    derivs=np.concatenate((r_derivs,dv1bydt,dv2bydt))
    return derivs


def ThreeBodyEquations(w, t, G, m1, m2, m3):
    r1 = w[:3]
    r2 = w[3:6]
    r3 = w[6:9]
    v1 = w[9:12]
    v2 = w[12:15]
    v3 = w[15:18]

    r12 = np.linalg.norm(r2 - r1)
    r13 = np.linalg.norm(r3 - r1)
    r23 = np.linalg.norm(r3 - r2)

    dv1bydt = K1 * m2 * (r2 - r1) / r12 ** 3 + K1 * m3 * (r3 - r1) / r13 ** 3
    dv2bydt = K1 * m1 * (r1 - r2) / r12 ** 3 + K1 * m3 * (r3 - r2) / r23 ** 3
    dv3bydt = K1 * m1 * (r1 - r3) / r13 ** 3 + K1 * m2 * (r2 - r3) / r23 ** 3
    dr1bydt = K2 * v1
    dr2bydt = K2 * v2
    dr3bydt = K2 * v3

    r12_derivs = np.concatenate((dr1bydt, dr2bydt))
    r_derivs = np.concatenate((r12_derivs, dr3bydt))
    v12_derivs = np.concatenate((dv1bydt, dv2bydt))
    v_derivs = np.concatenate((v12_derivs, dv3bydt))
    derivs = np.concatenate((r_derivs, v_derivs))
    return derivs
#Package initial parameters
# #双星
# init_params=np.array([r1,r2,v1,v2]) #create array of initial params
# init_params=init_params.flatten() #flatten array to make it 1D
# time_span=np.linspace(0,8,500) #8 orbital periods and 500 points
# #Run the ODE solver
# import scipy.integrate
# two_body_sol=scipy.integrate.odeint(TwoBodyEquations,init_params,time_span,args=(G,m1,m2))
#
# r1_sol=two_body_sol[:,:3]
# r2_sol=two_body_sol[:,3:6]

# #Create figure
# fig=plt.figure(1,figsize=(15,15))
# #Create 3D axes
# ax=fig.add_subplot(111,projection="3d")
# #Plot the orbits
# ax.plot(r1_sol[:,0],r1_sol[:,1],r1_sol[:,2],color="darkblue")
# ax.plot(r2_sol[:,0],r2_sol[:,1],r2_sol[:,2],color="tab:red")
# #Plot the final positions of the stars
# ax.scatter(r1_sol[-1,0],r1_sol[-1,1],r1_sol[-1,2],color="darkblue",marker="o",s=100,label="AlphaCentauri A")
# ax.scatter(r2_sol[-1,0],r2_sol[-1,1],r2_sol[-1,2],color="tab:red",marker="o",s=100,label="AlphaCentauri B")
# #Add a few more bells and whistles
# ax.set_xlabel("x-coordinate",fontsize=14)
# ax.set_ylabel("y-coordinate",fontsize=14)
# ax.set_zlabel("z-coordinate",fontsize=14)
# ax.set_title("Visualization of orbits of stars in a two-bodysystem\n",fontsize=14)
# ax.legend(loc="upper left",fontsize=14)


# #Find location of COM
# rcom_sol=(m1*r1_sol+m2*r2_sol)/(m1+m2)
#
# #Find location of Alpha Centauri A w.r.t COM
# r1com_sol=r1_sol-rcom_sol
#
# #Find location of Alpha Centauri B w.r.t COM
# r2com_sol=r2_sol-rcom_sol

#Package initial parameters
init_params=np.array([r1,r2,r3,v1,v2,v3]) #Initial parameters
init_params=init_params.flatten() #Flatten to make 1D array
time_span=np.linspace(0,20,1000) #20 orbital periods and 500 points

#Run the ODE solver
import scipy.integrate
three_body_sol=scipy.integrate.odeint(ThreeBodyEquations,init_params,time_span,args=(G,m1,m2,m3))
r1_sol=three_body_sol[:,:3]
r2_sol=three_body_sol[:,3:6]
r3_sol=three_body_sol[:,6:9]
# #Create figure
# fig=plt.figure(2,figsize=(15,15))
# #Create 3D axes
# ax=fig.add_subplot(111,projection="3d")
# #Plot the orbits
# ax.plot(r1_sol[:,0],r1_sol[:,1],r1_sol[:,2],color="darkblue")
# ax.plot(r2_sol[:,0],r2_sol[:,1],r2_sol[:,2],color="tab:red")
# ax.plot(r3_sol[:,0],r3_sol[:,1],r3_sol[:,2],color="green")
# #Plot the final positions of the stars
# ax.scatter(r1_sol[-1,0],r1_sol[-1,1],r1_sol[-1,2],color="darkblue",marker="o",s=100,label="AlphaCentauri A")
# ax.scatter(r2_sol[-1,0],r2_sol[-1,1],r2_sol[-1,2],color="tab:red",marker="o",s=100,label="AlphaCentauri B")
# ax.scatter(r3_sol[-1,0],r3_sol[-1,1],r3_sol[-1,2],color="green",marker="o",s=100,label="AlphaCentauri C")
# #Add a few more bells and whistles
# ax.set_xlabel("x-coordinate",fontsize=14)
# ax.set_ylabel("y-coordinate",fontsize=14)
# ax.set_zlabel("z-coordinate",fontsize=14)
# ax.set_title("Visualization of orbits of stars in a three-bodysystem\n",fontsize=14)
# ax.legend(loc="upper left",fontsize=14)


plt.ion()
for i in range(len(r1_sol[:,0])):
    fig = plt.figure(1, figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(r1_sol[:i,0],r1_sol[:i,1],r1_sol[:i,2],color="darkblue")
    ax.plot(r2_sol[:i,0],r2_sol[:i,1],r2_sol[:i,2], color="tab:red")
    ax.plot(r3_sol[:i,0],r3_sol[:i,1],r3_sol[:i,2],color="green")
    ax.scatter(r1_sol[i, 0], r1_sol[i, 1], r1_sol[i, 2], color="darkblue", marker="o", s=100,label="AlphaCentauri A")
    ax.scatter(r2_sol[i, 0], r2_sol[i, 1], r2_sol[i, 2], color="tab:red", marker="o", s=100, label="AlphaCentauri B")
    ax.scatter(r3_sol[i,0],r3_sol[i,1],r3_sol[i,2],color="green",marker="o",s=100,label="AlphaCentauri C")
    ax.set_xlabel("x-coordinate", fontsize=14)
    ax.set_ylabel("y-coordinate",fontsize=14)
    ax.set_zlabel("z-coordinate",fontsize=14)
    ax.set_title("Visualization of orbits of stars in a three-bodysystem\n", fontsize=14)
    ax.legend(loc="upper left",fontsize=14)
    plt.show()
    plt.pause(0.1)
    if i == len(r1_sol[:,0])-1:
        plt.savefig('三体运动轨迹.jpg')
    plt.clf()


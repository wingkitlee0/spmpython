"""
2nd order wave equation on Chebyshev grid
based on Program 19 in Spectral methods in MATLAB. Lloyd

eq.: 
    u_tt = c^2 u_xx => u_t = c v_x, v_t = c u_x

"""


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from chebfft import chebfft, chebfft_complex

def initcond(x):
    return np.exp(-200*x**2)

def initcond1(r,r0=5.0):
    return np.exp(-200*((r-r0)/drx)**2)


rmin = 1.0
rmax = 10.0
drx = 0.5*(rmax-rmin)
cr = drx
cc = 0.25

N = 80
x = np.cos(np.pi*np.arange(0,N+1)/N)
r = rmin + (x+1.0)*drx

dt = 16.0/N**2
print("# dt = ", dt)

#U = initcond1(r,0.5*(rmin+rmax))
U = initcond1(r,3)
#Uold = initcond1(r)
#vold = initcond1(r-cc*dt*drx)
#U = initcond(x-0.1)
#Uold = initcond(x-cc*dt) # x is descending, so this is inward-propagating in r
V = 0.0 * x
Vx0 = np.zeros_like(x)

tmax = 4
tplot = 0.1
plotgap = int(np.round(tplot/dt))
dt = tplot/plotgap
nplots = int(np.round(tmax/tplot))
plotdata = np.vstack((U, np.zeros((nplots,N+1))))
tdata = 0
for i in range(0,nplots):
    for n in range(0,plotgap):
        
        # Heun's method
        Ux0 = cc* chebfft(U).T   # Ux from current U
        
        Vx0 = cc* chebfft(V).T # Vx from current V
        Vx0[0] = 0.0 # u_t = 0 at edges
        Vx0[N] = 0.0
        
        Ustar = U + dt * Vx0  # full step of U
        Uxstar = cc* chebfft(Ustar).T # Vx from current V
        
        Vnew = V + dt * 0.5*(Ux0+Uxstar) # V at i+1
        
        
        Vx1 = cc* chebfft(Vnew).T       
        Vx1[0] = 0.0 # u_t = 0 at edges
        Vx1[N] = 0.0
        
        Unew = U + dt * 0.5 *(Vx0 + Vx1) # U at i+1
                
        U = Unew
        V = Vnew
    plotdata[i+1,:] = U
    tdata = np.vstack((tdata, dt*i*plotgap))

# Plot results

fig = plt.figure(figsize=(6,6))
ax = axes3d.Axes3D(fig)
X, Y = np.meshgrid(r, tdata)
#ax.plot_surface(X,Y,plotdata, rstride=1, cstride=1, cmap="viridis", alpha=0.3)
ax.plot_wireframe(X,Y,plotdata,color='k',lw=0.5)

ax.set_xlim(rmin, rmax) #ax.set_xlim(-1, 1)
ax.set_ylim(0, tmax)
ax.set_zlim(-2, 2)
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("v")
plt.show()
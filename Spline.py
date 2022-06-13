from scipy.interpolate import splprep, splev
from scipy.interpolate import BSpline
import numpy as np
import time
import matplotlib.pyplot as plt

class AKBSpline:
    def __init__(self, curve_sequence:np.array, num_knots:np.array = 10, degree:int = 3) -> None:
        assert curve_sequence.shape[0] >= degree + 2
        assert num_knots >= (degree + 1) * 2
        # self.curve_sequence = self._normalize(curve_sequence,True)
        self.curve_sequence = curve_sequence
        self.u = self._get_u(curve_sequence)
        p = degree + 1
        self.p = p
        self.num_knots = num_knots
        self.r = self.num_knots - 2 * degree
        # calculate p-th derivative of the input curve
        deri = []
        for pp in range(p):
            if pp == 0:
                next_deri = self._next_derivative(self.curve_sequence,self.u)
            else:
                next_deri = self._next_derivative(deri[pp-1][0], deri[pp-1][1])
            deri.append(next_deri)
        self.f, self.u_f = self._get_feature_function(deri[-1][0],deri[-1][1])
        self.F = self._integral_f()
        u_array = self._cal_auto_knots()
        self.knots = np.zeros((self.num_knots))
        self.knots[degree:-degree] = u_array
        self.knots[-degree:] = 1
        tcku, fp, ier, msg = splprep(self.curve_sequence.T, t=self.knots, k = degree, task=-1, full_output=True, quiet=True)
        
        self.tcku = tcku
        self.fp = fp

    def _normalize(self, seq:np.array, keep_aspect = False) -> np.array:
        if not keep_aspect:
            min_seq = np.min(seq, axis = 0, keepdims=True)
            max_seq = np.max(seq, axis = 0, keepdims=True)
        else: 
            min_seq = np.min(seq)
            max_seq = np.max(seq)
        seq = (seq - min_seq) / (max_seq - min_seq)
        return seq

    def _get_u(self, coords:np.array):
        dist = np.sqrt(np.sum((coords[1:] - coords[:-1]) ** 2,axis=1))
        acc_dist = np.zeros((coords.shape[0],))
        acc_dist[1:] = np.add.accumulate(dist)
        return acc_dist/ dist.sum()

    def _get_feature_function(self, p_th_deri:np.array, u:np.array):
        f = np.zeros((len(u) + 2))
        u_f = np.zeros_like(f)
        u_f[1:-1] = u
        u_f[-1] = 1
        f[1:-1] = np.linalg.norm(p_th_deri,axis=1) ** (1/(self.p))
        return f, u_f

    def _integral_f(self):
        f = self.f 
        u_f = self.u_f
        eps = 1e-10
        F = np.zeros_like(f)
        f_trap = np.zeros_like(f)
        acc = 0
        f_trap[1:] = 0.5 * (f[:-1] + f[1:] + eps) * (u_f[1:] - u_f[:-1])
        acc = f_trap.sum()
        del_f = acc / (self.r-1)
        acc = 0
        f_trap_del = np.where(f_trap<del_f,f_trap,del_f)
        F = np.add.accumulate(f_trap_del)
        return F

    def _cal_auto_knots(self):
        F = self.F
        u_f = self.u_f
        F_array = np.linspace(0,F[-1],self.r )
        idx = 0
        i = 0
        u_array = np.zeros_like(F_array)
        while True:
            interval = F[idx+1] - F[idx]
            if F_array[i] > F[idx+1]:
                idx += 1
                continue
            else:
                alpha = (F_array[i] - F[idx]) / interval
                u_array[i] = (1-alpha) * u_f[idx] + alpha * u_f[idx+1]
            i += 1
            if i == len(F_array):
                break
        return u_array

    def _next_derivative(self, sequence, u):
        xyz_diff = sequence[1:] - sequence[:-1]
        u_diff = u[1:] - u[:-1]
        new_u = (u[:-1] + u[1:]) / 2
        deri = xyz_diff / u_diff[:,None]
        return [deri, new_u]


    def _dist(self, coord1:np.array, coord2:np.array):
        return np.sqrt(np.sum((coord1 - coord2) ** 2))

if __name__ == "__main__":
    l = 30
    k = 3
    n_knots = 50
    tra = np.load('trajectories.npy')
    min_coord = np.array([   0. ,        0.,         0. ,     -999.64136, -999.7137,  -997.5717 ], dtype=np.float32)[None,None,:]
    max_coord = np.array([6.9077554e+00, 9.9213535e-01, 6.2831855e+00, 9.9986377e+02, 9.9980994e+02, 9.9779254e+02], dtype=np.float32)[None,None,:]
    #normalization
    tra = (tra - min_coord) / (max_coord - min_coord)
    length = np.load('tra_len.npy')
    trace = tra[0,:length[0],3:]
    spline = AKBSpline(trace,n_knots,k)
    tck, u = spline.tcku
    print(spline.knots)
    print(spline.F)
    new_points = splev(u, tck)
    
    fig = plt.figure(figsize=(18, 10), dpi=80)
    ax = fig.add_subplot(1,2,1,projection='3d')
    ax2 = fig.add_subplot(1,2,2,projection='3d')
    # ax.scatter(new_points[0][:l], new_points[1][:l], new_points[2][:l], '.',c=rgb_cycle)
    ax.plot(new_points[0][:l],new_points[1][:l],new_points[2][:l])

    # ax3.scatter(trace[:l,0], trace[:l,1], trace[:l,2], 'o',c=rgb_cycle)
    ax2.plot(trace[:l,0], trace[:l,1], trace[:l,2])
    plt.show()
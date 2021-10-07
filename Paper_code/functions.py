import numpy as np

def T_inf_ramp(t, idx_low, idx_high):
    a = 0.3
    b = (273 + 21) - t[idx_low] * a

    T_inf = np.zeros(t.shape[0])
    T_inf = a*t+b
    T_inf[0:idx_low] = 273 + 21
    T_inf[idx_high:] = 273 + 21
    return T_inf


def T_cone_ramp(t, idx_low, idx_high):
    a = .5
    b = 273 + 21 -t[idx_low]*a 

    T_inf = np.zeros(t.shape[0])
    T_inf = a*t+b
    T_inf[0:idx_low] = 273 + 21
    T_inf[idx_high:] = T_inf[idx_high]
    return T_inf

def Cone_Temp(time, t1, t2, T_cone, T_inf):
    T = ((time > t1) & (time < t2))*(T_cone)
    T[T==0] = T_inf
    return T

def generate_u_lin(t):
    decrease_or_increase = np.random.randint(2)
    if decrease_or_increase == 0:
        a = np.random.uniform(0.005,0.015)
        b = np.random.uniform(1,2)
    else:
        a = np.random.uniform(-0.005, -0.02)
        b = np.random.uniform(10, 12)

    return a*t+b


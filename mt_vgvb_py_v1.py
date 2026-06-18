#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First-stage VG/VB aggregate flexibility reproduction (Python/Gurobi).

Sign convention and scope:
- pG_agg[t] is the aggregated active-power injection deviation of GDERs from the base dispatch.
- pB_agg[t] is the aggregated active-power injection deviation of BDERs from the base dispatch.
- Positive values mean increased injection into the distribution network, or equivalently reduced net load.
- pVPP_flex[t] = pG_agg[t] + pB_agg[t].
- This version only computes initial boundaries of pG_agg and pB_agg; it does not compute final safe VPP boundaries.
- The VG/VB polytopes are initial / no-shrink approximations and are not guaranteed safe for disaggregation.
- Not implemented here: bound shrinking, infeasible point search, paper-style (41)-(43), neural networks, BPINN datasets.

Run: py mt_vgvb_py_v1.py
Outputs: results_mt_vgvb_py_v1/
"""
from __future__ import annotations

import csv, json, math, os, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as exc:
    raise SystemExit("gurobipy/Gurobi is required for mt_vgvb_py_v1.py. Import failed: %s" % exc)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit("matplotlib is required for plotting. Import failed: %s" % exc)


def log(msg: str):
    print(msg, flush=True)


def config_default() -> Dict:
    T, DT = 24, 1.0
    if False:  # FAST_5MIN_TEST switch retained intentionally.
        T, DT = 288, 5/60
    return dict(CASE_NAME="ieee33", T=T, DT=DT, V_MIN=0.95**2, V_MAX=1.05**2,
                V0=1.0, MODE_NETWORK_DECOUPLING="capacity_weighted", alpha_net=0.5,
                OutputFlag=0, TimeLimit=120, FeasibilityTol=1e-7, MIPGap=1e-6,
                n_random_checks=12, random_seed=7, result_note="initial / no-shrink / not guaranteed safe disaggregation")


def build_case_ieee33() -> Dict:
    # Simplified IEEE-33 radial feeder (per-unit-ish); branch order is parent before child.
    raw = [(1,2,.0922,.0470),(2,3,.4930,.2511),(3,4,.3660,.1864),(4,5,.3811,.1941),(5,6,.8190,.7070),
           (6,7,.1872,.6188),(7,8,.7114,.2351),(8,9,1.0300,.7400),(9,10,1.0440,.7400),(10,11,.1966,.0650),
           (11,12,.3744,.1238),(12,13,1.4680,1.1550),(13,14,.5416,.7129),(14,15,.5910,.5260),(15,16,.7463,.5450),
           (16,17,1.2890,1.7210),(17,18,.7320,.5740),(2,19,.1640,.1565),(19,20,1.5042,1.3554),(20,21,.4095,.4784),
           (21,22,.7089,.9373),(3,23,.4512,.3083),(23,24,.8980,.7091),(24,25,.8960,.7011),(6,26,.2030,.1034),
           (26,27,.2842,.1447),(27,28,1.0590,.9337),(28,29,.8042,.7006),(29,30,.5075,.2585),(30,31,.9744,.9630),
           (31,32,.3105,.3619),(32,33,.3410,.5302)]
    branches = np.array([[a-1,b-1,r/100,x/100,3.5] for a,b,r,x in raw], float)
    nbus, nbranch = 33, len(branches)
    parent = {int(to): int(fr) for fr,to,_,_,_ in branches}
    children = {i: [] for i in range(nbus)}
    for l,(fr,to,_,_,_) in enumerate(branches): children[int(fr)].append((int(to),l))
    # MW/MVAr load shape base by bus.
    p = np.zeros(nbus); q = np.zeros(nbus)
    for b,val in {2:.10,3:.09,4:.12,5:.06,6:.06,7:.20,8:.20,9:.06,10:.06,11:.045,12:.06,13:.06,14:.12,15:.06,16:.06,17:.09,18:.09,19:.09,20:.09,21:.09,22:.09,23:.09,24:.42,25:.42,26:.06,27:.06,28:.06,29:.12,30:.20,31:.15,32:.21,33:.06}.items(): p[b-1]=val; q[b-1]=0.45*val
    return dict(nbus=nbus, nbranch=nbranch, branches=branches, parent=parent, children=children, p_load_base=p, q_load_base=q)


def build_profiles(T, DT):
    h=np.arange(T); load=0.72+0.24*np.sin((h-7)/24*2*np.pi)+0.10*np.sin((h-18)/24*4*np.pi); load=np.clip(load,.55,1.08)
    price=45+25*((h>=17)&(h<=21))+10*((h>=8)&(h<=15))
    return dict(load_mult=load, price=price)


def build_resources():
    return dict(G=[dict(bus=7,pmin=.00,pmax=.65,qmin=-.25,qmax=.25,rup=.35,rdn=.35,cost=18), dict(bus=24,pmin=.05,pmax=.85,qmin=-.30,qmax=.30,rup=.40,rdn=.40,cost=22), dict(bus=30,pmin=.00,pmax=.55,qmin=-.20,qmax=.20,rup=.28,rdn=.28,cost=25)],
                B=[dict(bus=14,pch=.45,pdis=.45,emin=.15,emax=1.60,e0=.80,eta_ch=.95,eta_dis=.95,cost=1.5), dict(bus=28,pch=.35,pdis=.35,emin=.10,emax=1.20,e0=.60,eta_ch=.95,eta_dis=.95,cost=1.5)])


def downstream_matrix(case):
    n,l=case['nbus'],case['nbranch']; D=np.zeros((l,n))
    def fill(node, br):
        D[br,node]=1
        for ch,bl in case['children'][node]: fill(ch,br)
    for ch,bl in case['children'][0]: fill(ch,bl)
    return D


def path_matrix(case):
    n,l=case['nbus'],case['nbranch']; A=np.zeros((n,l))
    child_to_branch={int(to):i for i,(_,to,_,_,_) in enumerate(case['branches'])}
    for b in range(1,n):
        cur=b
        while cur!=0:
            br=child_to_branch[cur]; A[b,br]=1; cur=case['parent'][cur]
    return A


def network_from_injections(case, p_load, q_load, p_inj, q_inj):
    D=downstream_matrix(case); A=path_matrix(case); br=case['branches']; r=br[:,2]; x=br[:,3]
    netp=p_load-p_inj; netq=q_load-q_inj
    P=D@netp; Q=D@netq; v=1.0-2*(A@(r*P+x*Q))
    p0=float(P[0]) if len(P) else float(netp.sum()); q0=float(Q[0]) if len(Q) else float(netq.sum())
    return P,Q,v,p0,q0


def solve_base_dispatch_gurobi(case, profiles, resources, config):
    log('[1/7] Solving full coupled LinDistFlow base dispatch with Gurobi...')
    T=config['T']; G=resources['G']; B=resources['B']; ng=len(G); nb=len(B); nbus=case['nbus']; nl=case['nbranch']
    m=gp.Model('base_dispatch'); set_params(m,config)
    pG=m.addVars(ng,T,lb={(g,t):G[g]['pmin'] for g in range(ng) for t in range(T)}, ub={(g,t):G[g]['pmax'] for g in range(ng) for t in range(T)}, name='pG')
    qG=m.addVars(ng,T,lb={(g,t):G[g]['qmin'] for g in range(ng) for t in range(T)}, ub={(g,t):G[g]['qmax'] for g in range(ng) for t in range(T)}, name='qG')
    pCh=m.addVars(nb,T,lb=0,name='pCh'); pDis=m.addVars(nb,T,lb=0,name='pDis'); e=m.addVars(nb,T+1,name='e')
    p0imp=m.addVars(T,lb=0,name='p0_import')
    for b,R in enumerate(B):
        m.addConstr(e[b,0]==R['e0']); m.addConstr(e[b,T]==R['e0'])
        for t in range(T):
            pCh[b,t].UB=R['pch']; pDis[b,t].UB=R['pdis']; e[b,t].LB=R['emin']; e[b,t].UB=R['emax']
            m.addConstr(e[b,t+1]==e[b,t]+R['eta_ch']*pCh[b,t]*config['DT']-pDis[b,t]/R['eta_dis']*config['DT'])
        e[b,T].LB=R['emin']; e[b,T].UB=R['emax']
    for g,R in enumerate(G):
        for t in range(1,T):
            m.addConstr(pG[g,t]-pG[g,t-1] <= R['rup']*config['DT']); m.addConstr(pG[g,t-1]-pG[g,t] <= R['rdn']*config['DT'])
    # Network constraints by explicit linear sensitivity in variables.
    D=downstream_matrix(case); A=path_matrix(case); br=case['branches']; r=br[:,2]; x=br[:,3]; S=br[:,4]
    for t in range(T):
        pl=case['p_load_base']*profiles['load_mult'][t]; ql=case['q_load_base']*profiles['load_mult'][t]
        pexpr=[-float(pl[i]) for i in range(nbus)]; qexpr=[-float(ql[i]) for i in range(nbus)]
        for g,R in enumerate(G): pexpr[R['bus']]+=pG[g,t]; qexpr[R['bus']]+=qG[g,t]
        for b,R in enumerate(B): pexpr[R['bus']]+=pDis[b,t]-pCh[b,t]
        for ell in range(nl):
            P=sum(D[ell,i]*(-pexpr[i]) for i in range(nbus)); Q=sum(D[ell,i]*(-qexpr[i]) for i in range(nbus))
            m.addConstr(P <= S[ell]); m.addConstr(P >= -S[ell]); m.addConstr(Q <= S[ell]); m.addConstr(Q >= -S[ell])
        for i in range(nbus):
            v=1.0-sum(2*A[i,ell]*(r[ell]*sum(D[ell,j]*(-pexpr[j]) for j in range(nbus))+x[ell]*sum(D[ell,j]*(-qexpr[j]) for j in range(nbus))) for ell in range(nl))
            m.addConstr(v <= config['V_MAX']); m.addConstr(v >= config['V_MIN'])
        m.addConstr(p0imp[t] >= sum(pl)-sum(pG[g,t] for g in range(ng))-sum(pDis[b,t]-pCh[b,t] for b in range(nb)))
    m.setObjective(sum(profiles['price'][t]*p0imp[t] for t in range(T))+sum(G[g]['cost']*pG[g,t] for g in range(ng) for t in range(T))+sum(B[b]['cost']*(pCh[b,t]+pDis[b,t]) for b in range(nb) for t in range(T)), GRB.MINIMIZE)
    m.optimize(); ensure_optimal(m,'base_dispatch')
    pGv=np.array([[pG[g,t].X for t in range(T)] for g in range(ng)]); qGv=np.array([[qG[g,t].X for t in range(T)] for g in range(ng)])
    pChv=np.array([[pCh[b,t].X for t in range(T)] for b in range(nb)]); pDisv=np.array([[pDis[b,t].X for t in range(T)] for b in range(nb)]); ev=np.array([[e[b,t].X for t in range(T+1)] for b in range(nb)])
    P=np.zeros((nl,T)); Q=np.zeros((nl,T)); V=np.zeros((case['nbus'],T)); p0=np.zeros(T); q0=np.zeros(T)
    for t in range(T):
        pinj=np.zeros(case['nbus']); qinj=np.zeros(case['nbus'])
        for g,R in enumerate(G): pinj[R['bus']]+=pGv[g,t]; qinj[R['bus']]+=qGv[g,t]
        for b,R in enumerate(B): pinj[R['bus']]+=pDisv[b,t]-pChv[b,t]
        P[:,t],Q[:,t],V[:,t],p0[t],q0[t]=network_from_injections(case,case['p_load_base']*profiles['load_mult'][t],case['q_load_base']*profiles['load_mult'][t],pinj,qinj)
    return dict(pG=pGv,qG=qGv,pCh=pChv,pDis=pDisv,pB=pDisv-pChv,eB=ev,P=P,Q=Q,V=V,p0=p0,q0=q0,obj=m.ObjVal)


def set_params(m,config):
    for k in ['OutputFlag','TimeLimit','FeasibilityTol','MIPGap']:
        if k in config: m.setParam(k, config[k])

def ensure_optimal(m,name):
    if m.Status != GRB.OPTIMAL:
        log(f'ERROR {name} status={m.Status}')
        if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            m.computeIIS(); m.write(f'{name}.ilp')
        raise RuntimeError(f'{name} failed with status {m.Status}')


def build_network_sensitivity_matrices(case, base_solution, resources, config):
    log('[2/7] Building LinDistFlow network sensitivity matrices...')
    # Approximation: radial LinDistFlow is linear; injection at bus j reduces upstream branch demand flow by 1 MW
    # and raises downstream/path voltages according to 2*r/x path products. q sensitivity is included for GDER only.
    T=config['T']; G=resources['G']; B=resources['B']; ng=len(G); nb=len(B); nl=case['nbranch']; nbus=case['nbus']; nyG=2*ng*T; nyB=nb*T
    D=downstream_matrix(case); A=path_matrix(case); br=case['branches']; r=br[:,2]; x=br[:,3]; S=br[:,4]
    rows=[]; HG=[]; HB=[]; h=[]
    def add(row_type,t,idx,base,margin,cg,cb): rows.append(dict(type=row_type,time=t+1,index=idx+1,base_value=float(base),remaining_margin=float(margin))); HG.append(cg.copy()); HB.append(cb.copy()); h.append(float(margin))
    for t in range(T):
        for i in range(nbus):
            cg=np.zeros(nyG); cb=np.zeros(nyB)
            for g,R in enumerate(G):
                sensp=2*sum(A[i,l]*r[l]*D[l,R['bus']] for l in range(nl)); sensq=2*sum(A[i,l]*x[l]*D[l,R['bus']] for l in range(nl))
                cg[g*T+t]=sensp; cg[(ng+g)*T+t]=sensq
            for b,R in enumerate(B): cb[b*T+t]=2*sum(A[i,l]*r[l]*D[l,R['bus']] for l in range(nl))
            add('voltage_upper',t,i,base_solution['V'][i,t],config['V_MAX']-base_solution['V'][i,t],cg,cb)
            add('voltage_lower',t,i,-base_solution['V'][i,t],base_solution['V'][i,t]-config['V_MIN'],-cg,-cb)
        for l in range(nl):
            cg=np.zeros(nyG); cb=np.zeros(nyB)
            for g,R in enumerate(G): cg[g*T+t]=-D[l,R['bus']]; cg[(ng+g)*T+t]=0
            for b,R in enumerate(B): cb[b*T+t]=-D[l,R['bus']]
            add('lineP_upper',t,l,base_solution['P'][l,t],S[l]-base_solution['P'][l,t],cg,cb)
            add('lineP_lower',t,l,-base_solution['P'][l,t],S[l]+base_solution['P'][l,t],-cg,-cb)
            cq=np.zeros(nyG)
            for g,R in enumerate(G): cq[(ng+g)*T+t]=-D[l,R['bus']]
            add('lineQ_upper',t,l,base_solution['Q'][l,t],S[l]-base_solution['Q'][l,t],cq,np.zeros(nyB))
            add('lineQ_lower',t,l,-base_solution['Q'][l,t],S[l]+base_solution['Q'][l,t],-cq,np.zeros(nyB))
    h=np.array(h); 
    if np.any(h < -1e-6): raise RuntimeError('Base point violates network constraints; negative h_net detected.')
    return np.vstack(HG), np.vstack(HB), np.maximum(h,0), rows


def resource_delta_bounds(resources, base, config):
    T=config['T']; G=resources['G']; B=resources['B']; ng=len(G); nb=len(B)
    lbG=[]; ubG=[]
    for g,R in enumerate(G): lbG += list(R['pmin']-base['pG'][g]); ubG += list(R['pmax']-base['pG'][g])
    for g,R in enumerate(G): lbG += list(R['qmin']-base['qG'][g]); ubG += list(R['qmax']-base['qG'][g])
    lbB=[]; ubB=[]
    for b,R in enumerate(B): lbB += list(-R['pch']-base['pB'][b]); ubB += list(R['pdis']-base['pB'][b])
    return np.array(lbG),np.array(ubG),np.array(lbB),np.array(ubB)


def decouple_network_constraints(H_G,H_B,h_net,mode,config,resources,base):
    log('[3/7] Decoupling network margins (%s)...' % mode)
    lbG,ubG,lbB,ubB=resource_delta_bounds(resources,base,config)
    rangeG=np.maximum(H_G,0)@ubG + np.minimum(H_G,0)@lbG
    rangeB=np.maximum(H_B,0)@ubB + np.minimum(H_B,0)@lbB
    rangeG=np.maximum(rangeG,0); rangeB=np.maximum(rangeB,0)
    if mode=='equal_margin': hG=config['alpha_net']*h_net; hB=h_net-hG
    else:
        den=rangeG+rangeB; frac=np.where(den>1e-10,rangeG/den,0.5); hG=h_net*frac; hB=h_net-hG
    return hG,hB,rangeG,rangeB

# To keep both versions auditable, helper models are compact and export every LP solution.
def build_g_model(data, fix_candidate=None):
    c=data['config']; T=c['T']; G=data['resources']['G']; ng=len(G); m=gp.Model('omega_g'); set_params(m,c)
    p=m.addVars(ng,T,name='pG'); q=m.addVars(ng,T,name='qG')
    for g,R in enumerate(G):
        for t in range(T): p[g,t].LB=R['pmin']; p[g,t].UB=R['pmax']; q[g,t].LB=R['qmin']; q[g,t].UB=R['qmax']
        for t in range(1,T): m.addConstr(p[g,t]-p[g,t-1]<=R['rup']*c['DT']); m.addConstr(p[g,t-1]-p[g,t]<=R['rdn']*c['DT'])
    y=[]
    for g in range(ng):
        for t in range(T): y.append(p[g,t]-data['base']['pG'][g,t])
    for g in range(ng):
        for t in range(T): y.append(q[g,t]-data['base']['qG'][g,t])
    for k in range(data['H_G'].shape[0]): m.addConstr(gp.quicksum(float(data['H_G'][k,j])*y[j] for j in range(len(y))) <= float(data['h_G_dec'][k]))
    pagg=[gp.quicksum(p[g,t]-data['base']['pG'][g,t] for g in range(ng)) for t in range(T)]
    if fix_candidate is not None:
        for t in range(T): m.addConstr(pagg[t]==float(fix_candidate[t]))
    return m,p,q,pagg

def build_b_model(data, fix_candidate=None):
    c=data['config']; T=c['T']; B=data['resources']['B']; nb=len(B); m=gp.Model('omega_b'); set_params(m,c)
    ch=m.addVars(nb,T,lb=0,name='pCh'); dis=m.addVars(nb,T,lb=0,name='pDis'); e=m.addVars(nb,T+1,name='e')
    for b,R in enumerate(B):
        m.addConstr(e[b,0]==R['e0']); m.addConstr(e[b,T]==R['e0'])
        for t in range(T): ch[b,t].UB=R['pch']; dis[b,t].UB=R['pdis']; e[b,t].LB=R['emin']; e[b,t].UB=R['emax']; m.addConstr(e[b,t+1]==e[b,t]+R['eta_ch']*ch[b,t]*c['DT']-dis[b,t]/R['eta_dis']*c['DT'])
        e[b,T].LB=R['emin']; e[b,T].UB=R['emax']
    y=[dis[b,t]-ch[b,t]-data['base']['pB'][b,t] for b in range(nb) for t in range(T)]
    for k in range(data['H_B'].shape[0]): m.addConstr(gp.quicksum(float(data['H_B'][k,j])*y[j] for j in range(len(y))) <= float(data['h_B_dec'][k]))
    pagg=[gp.quicksum(dis[b,t]-ch[b,t]-data['base']['pB'][b,t] for b in range(nb)) for t in range(T)]
    if fix_candidate is not None:
        for t in range(T): m.addConstr(pagg[t]==float(fix_candidate[t]))
    return m,ch,dis,e,pagg

def solve_omega_g_bound(bound_type,t,data,config):
    m,p,q,pagg=build_g_model(data); expr=pagg[t] if 'p_' in bound_type else pagg[t]-pagg[t-1]
    m.setObjective(expr, GRB.MAXIMIZE if bound_type.endswith('up') else GRB.MINIMIZE); st=time.time(); m.optimize(); rt=time.time()-st; ensure_optimal(m,'g_'+bound_type)
    G=data['resources']['G']; ng=len(G); T=config['T']
    return dict(status=m.Status, objective=m.ObjVal, runtime=rt, pG_agg=np.array([pagg[i].getValue() for i in range(T)]), pG=np.array([[p[g,i].X for i in range(T)] for g in range(ng)]), qG=np.array([[q[g,i].X for i in range(T)] for g in range(ng)]))

def solve_all_omega_g_bounds(data, config):
    log('[4/7] Solving Ω^G initial VG boundary LPs...'); T=config['T']; sol={}; bounds=np.zeros((T,4))*np.nan
    for t in range(T):
        for typ,col in [('p_dn',0),('p_up',1)]: log(f'  G {typ} t={t+1}/{T}'); r=solve_omega_g_bound(typ,t,data,config); sol[f'{typ}_{t+1}']=r; bounds[t,col]=r['objective']
        if t>0:
            for typ,col in [('r_dn',2),('r_up',3)]: log(f'  G {typ} t={t+1}/{T}'); r=solve_omega_g_bound(typ,t,data,config); sol[f'{typ}_{t+1}']=r; bounds[t,col]=r['objective']
    bounds[0,2:4]=0; return dict(table=bounds, solutions=sol)

def solve_omega_b_bound(bound_type,t,data,config):
    m,ch,dis,e,pagg=build_b_model(data); EB=sum(pagg[i]*config['DT'] for i in range(t+1)); expr=pagg[t] if 'p_' in bound_type else EB
    m.setObjective(expr, GRB.MAXIMIZE if bound_type.endswith('up') else GRB.MINIMIZE); st=time.time(); m.optimize(); rt=time.time()-st; ensure_optimal(m,'b_'+bound_type)
    B=data['resources']['B']; nb=len(B); T=config['T']; pa=np.array([pagg[i].getValue() for i in range(T)])
    return dict(status=m.Status, objective=m.ObjVal, runtime=rt, pB_agg=pa, EB_agg=np.cumsum(pa)*config['DT'], pCh=np.array([[ch[b,i].X for i in range(T)] for b in range(nb)]), pDis=np.array([[dis[b,i].X for i in range(T)] for b in range(nb)]), eB=np.array([[e[b,i].X for i in range(T+1)] for b in range(nb)]))

def solve_all_omega_b_bounds(data, config):
    log('[5/7] Solving Ω^B initial VB boundary LPs...'); T=config['T']; sol={}; bounds=np.zeros((T,4))
    for t in range(T):
        for typ,col in [('p_dn',0),('p_up',1),('e_dn',2),('e_up',3)]: log(f'  B {typ} t={t+1}/{T}'); r=solve_omega_b_bound(typ,t,data,config); sol[f'{typ}_{t+1}']=r; bounds[t,col]=r['objective']
    return dict(table=bounds, solutions=sol)

def build_vg_polytope(bounds,T,DT):
    A=[]; b=[]
    for t in range(T):
        e=np.zeros(T); e[t]=1; A+=[e,-e]; b+=[bounds[t,1],-bounds[t,0]]
    for t in range(1,T):
        e=np.zeros(T); e[t]=1; e[t-1]=-1; A+=[e,-e]; b+=[bounds[t,3],-bounds[t,2]]
    return np.vstack(A),np.array(b)

def build_vb_polytope(bounds,T,DT):
    A=[]; b=[]
    for t in range(T):
        e=np.zeros(T); e[t]=1; A+=[e,-e]; b+=[bounds[t,1],-bounds[t,0]]
        c=np.zeros(T); c[:t+1]=DT; A+=[c,-c]; b+=[bounds[t,3],-bounds[t,2]]
    return np.vstack(A),np.array(b)

def sample_random_objective_trajectories(A,b,n_samples,config):
    rng=np.random.default_rng(config['random_seed']); T=A.shape[1]; out=[]
    for s in range(n_samples):
        u=rng.normal(size=T); m=gp.Model('poly_sample'); set_params(m,config); x=m.addVars(T,lb=-GRB.INFINITY,name='p')
        for i in range(A.shape[0]): m.addConstr(gp.quicksum(float(A[i,j])*x[j] for j in range(T)) <= float(b[i]))
        m.setObjective(gp.quicksum(float(u[j])*x[j] for j in range(T)), GRB.MAXIMIZE); m.optimize()
        out.append(np.array([x[j].X for j in range(T)]) if m.Status==GRB.OPTIMAL else np.full(T,np.nan))
    return np.vstack(out)

def check_disaggregation_g(pG_candidate,data,config):
    m,*_=build_g_model(data,pG_candidate); m.optimize(); return m.Status==GRB.OPTIMAL, m.Status

def check_disaggregation_b(pB_candidate,data,config):
    m,*_=build_b_model(data,pB_candidate); m.optimize(); return m.Status==GRB.OPTIMAL, m.Status

def write_csv(path, header, rows):
    with open(path,'w',newline='',encoding='utf-8') as f: w=csv.writer(f); w.writerow(header); w.writerows(rows)

def export_results(out, case, profiles, resources, config, data, bg, bb, Ag,bgpoly,Ab,bbpoly, randG,randB, chkG,chkB):
    log('[6/7] Exporting CSV/NPZ results...'); out.mkdir(exist_ok=False); (out/'figures').mkdir(); (out/'omega_g_solutions').mkdir(); (out/'omega_b_solutions').mkdir()
    json.dump(config,open(out/'config.json','w'),indent=2)
    T=config['T']; write_csv(out/'base_dispatch.csv',['t','p0_base','q0_base'],[[t+1,data['base']['p0'][t],data['base']['q0'][t]] for t in range(T)])
    write_csv(out/'network_margins.csv',['row','type','time','index','base_value','h_net'],[[i+1,m['type'],m['time'],m['index'],m['base_value'],data['h_net'][i]] for i,m in enumerate(data['row_meta'])])
    write_csv(out/'decoupling_summary.csv',['row','type','time','index','h_net','h_G_dec','h_B_dec','range_G','range_B'],[[i+1,m['type'],m['time'],m['index'],data['h_net'][i],data['h_G_dec'][i],data['h_B_dec'][i],data['range_G'][i],data['range_B'][i]] for i,m in enumerate(data['row_meta'])])
    write_csv(out/'omega_g_bounds.csv',['t','pG_dn','pG_up','rG_dn','rG_up'],[[t+1,*bg['table'][t]] for t in range(T)])
    write_csv(out/'omega_b_bounds.csv',['t','pB_dn','pB_up','eB_dn','eB_up'],[[t+1,*bb['table'][t]] for t in range(T)])
    np.savetxt(out/'poly_g_init_A.csv',Ag,delimiter=','); np.savetxt(out/'poly_g_init_b.csv',bgpoly,delimiter=','); np.savetxt(out/'poly_b_init_A.csv',Ab,delimiter=','); np.savetxt(out/'poly_b_init_b.csv',bbpoly,delimiter=',')
    logs=[]
    for name,r in {**bg['solutions'],**bb['solutions']}.items(): logs.append([name,r['status'],r['objective'],r['runtime']])
    write_csv(out/'solver_log.csv',['problem','status','objective','runtime_sec'],logs)
    write_csv(out/'no_shrink_feasibility_check_g.csv',['sample','feasible','status'],[[i+1,chkG[i][0],chkG[i][1]] for i in range(len(chkG))])
    write_csv(out/'no_shrink_feasibility_check_b.csv',['sample','feasible','status'],[[i+1,chkB[i][0],chkB[i][1]] for i in range(len(chkB))])
    msg='This version does not perform bound shrinking. Infeasible candidate trajectories may exist inside Ω2_G_init or Ω2_B_init. These diagnostics are only for later shrink/NN-feedback development.'
    write_csv(out/'no_shrink_summary.csv',['field','value'],[['result_note',config['result_note']],['message',msg],['g_infeasible_ratio',1-np.mean([x[0] for x in chkG])],['b_infeasible_ratio',1-np.mean([x[0] for x in chkB])],['no_bound_shrinking','true'],['no_neural_network','true']])
    for name,r in bg['solutions'].items(): np.savetxt(out/'omega_g_solutions'/(name+'.csv'),np.c_[r['pG_agg']],delimiter=',',header='pG_agg',comments='')
    for name,r in bb['solutions'].items(): np.savetxt(out/'omega_b_solutions'/(name+'.csv'),np.c_[r['pB_agg'],r['EB_agg']],delimiter=',',header='pB_agg,EB_agg',comments='')
    np.savez(out/'results_data.npz',H_G=data['H_G'],H_B=data['H_B'],h_net=data['h_net'],h_G_dec=data['h_G_dec'],h_B_dec=data['h_B_dec'],omega_g=bg['table'],omega_b=bb['table'],Ag=Ag,bg=bgpoly,Ab=Ab,bb=bbpoly,randG=randG,randB=randB)

def plot_results(out,data,bg,bb,randG,randB):
    log('[7/7] Plotting figures...'); fig=out/'figures'; T=len(data['base']['p0']); x=np.arange(1,T+1)
    def save(name): plt.tight_layout(); plt.savefig(fig/name,dpi=160); plt.close()
    plt.figure(); plt.plot(x,data['base']['p0']); plt.title('Base PCC active power'); plt.xlabel('t'); plt.ylabel('MW'); save('base_p0_profile.png')
    plt.figure(); plt.plot(x,bg['table'][:,0],label='pG_dn'); plt.plot(x,bg['table'][:,1],label='pG_up'); plt.legend(); save('omega_g_power_bounds.png')
    plt.figure(); plt.plot(x,bg['table'][:,2],label='rG_dn'); plt.plot(x,bg['table'][:,3],label='rG_up'); plt.legend(); save('omega_g_ramp_bounds.png')
    plt.figure(); plt.plot(x,bb['table'][:,0],label='pB_dn'); plt.plot(x,bb['table'][:,1],label='pB_up'); plt.legend(); save('omega_b_power_bounds.png')
    plt.figure(); plt.plot(x,bb['table'][:,2],label='eB_dn'); plt.plot(x,bb['table'][:,3],label='eB_up'); plt.legend(); save('omega_b_energy_bounds.png')
    types=sorted(set(m['type'] for m in data['row_meta'])); vals=[]
    for typ in types:
        idx=[i for i,m in enumerate(data['row_meta']) if m['type']==typ]; vals.append([np.mean(data['h_net'][idx]),np.mean(data['h_G_dec'][idx]),np.mean(data['h_B_dec'][idx])])
    plt.figure(figsize=(10,4)); yy=np.arange(len(types)); vals=np.array(vals); plt.bar(yy-0.25,vals[:,0],.25,label='h_net'); plt.bar(yy,vals[:,1],.25,label='h_G_dec'); plt.bar(yy+.25,vals[:,2],.25,label='h_B_dec'); plt.xticks(yy,types,rotation=30,ha='right'); plt.legend(); save('network_margin_split.png')
    plt.figure(); [plt.plot(x,r,alpha=.35) for r in randG]; plt.plot(x,bg['table'][:,0],'k--'); plt.plot(x,bg['table'][:,1],'k--'); save('optional_random_trajs_g.png')
    plt.figure(); [plt.plot(x,r,alpha=.35) for r in randB]; plt.plot(x,bb['table'][:,0],'k--'); plt.plot(x,bb['table'][:,1],'k--'); save('optional_random_trajs_b.png')

def main():
    config=config_default(); out=Path('results_mt_vgvb_py_v1')
    if out.exists(): raise SystemExit(f'Output directory {out} already exists; refusing to overwrite existing files.')
    log('VG/VB first-stage initial/no-shrink reproduction (Python/Gurobi)')
    case=build_case_ieee33(); profiles=build_profiles(config['T'],config['DT']); resources=build_resources(); base=solve_base_dispatch_gurobi(case,profiles,resources,config)
    H_G,H_B,h_net,row_meta=build_network_sensitivity_matrices(case,base,resources,config)
    hG,hB,rangeG,rangeB=decouple_network_constraints(H_G,H_B,h_net,config['MODE_NETWORK_DECOUPLING'],config,resources,base)
    data=dict(config=config,case=case,profiles=profiles,resources=resources,base=base,H_G=H_G,H_B=H_B,h_net=h_net,h_G_dec=hG,h_B_dec=hB,range_G=rangeG,range_B=rangeB,row_meta=row_meta)
    bg=solve_all_omega_g_bounds(data,config); bb=solve_all_omega_b_bounds(data,config)
    Ag,bgv=build_vg_polytope(bg['table'],config['T'],config['DT']); Ab,bbv=build_vb_polytope(bb['table'],config['T'],config['DT'])
    log('[6/7] Running no-shrink random objective diagnostics (no boundary modification)...')
    randG=sample_random_objective_trajectories(Ag,bgv,config['n_random_checks'],config); randB=sample_random_objective_trajectories(Ab,bbv,config['n_random_checks'],config)
    chkG=[check_disaggregation_g(r,data,config) for r in randG]; chkB=[check_disaggregation_b(r,data,config) for r in randB]
    export_results(out,case,profiles,resources,config,data,bg,bb,Ag,bgv,Ab,bbv,randG,randB,chkG,chkB); plot_results(out,data,bg,bb,randG,randB)
    log('Done. Results are initial / no-shrink / not guaranteed safe disaggregation.')

if __name__ == '__main__': main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First-stage VG/VB aggregate flexibility reproduction v1_2 (Python/Gurobi).

Sign convention and scope:
- pG_agg[t] is the aggregated active-power injection deviation of GDERs from the base dispatch.
- pB_agg[t] is the aggregated active-power injection deviation of BDERs from the base dispatch.
- Positive values mean increased injection into the distribution network, or equivalently reduced net load.
- pVPP_flex[t] = pG_agg[t] + pB_agg[t].
- This version only computes initial boundaries of pG_agg and pB_agg; it does not compute final safe VPP boundaries.
- The VG/VB polytopes are initial / no-shrink approximations and are not guaranteed safe for disaggregation.
- Not implemented here: bound shrinking, infeasible point search, paper-style (41)-(43), neural networks, BPINN datasets.

Run: py mt_vgvb_py_v1_2.py
Outputs: results_mt_vgvb_py_v1_2/
"""
from __future__ import annotations

import csv, json, math, os, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_IMPORT_ERROR = None
except Exception as exc:
    gp = None
    class _MissingGRB:
        OPTIMAL = 2; INFEASIBLE = 3; INF_OR_UNBD = 4; MINIMIZE = 1; MAXIMIZE = -1; INFINITY = 1e100
    GRB = _MissingGRB()
    GUROBI_IMPORT_ERROR = exc

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
    return dict(CASE_NAME="ieee33", CASE_PARAM_MODE="enhanced33_paper", BASE_KV=12.66, BASE_MVA=1.0,
                Z_BASE=12.66**2/1.0, T=T, DT=DT, V_MIN=0.95**2, V_MAX=1.05**2,
                V0=1.0, NETWORK_DECOUPLING_MODE="capacity_weighted", MODE_NETWORK_DECOUPLING="capacity_weighted", alpha_net=0.5,
                ALLOW_DG_Q_SUPPORT=False, RPC_IMPLEMENTED=False, RPC_PLACEHOLDER_BUSES=[18,33],
                OLTC_OPTIMIZED=False, OLTC_FIXED_TAP=1.0,
                RPC_NOTE="RPC data are recorded as benchmark information only, not optimized in v1_2.",
                OLTC_NOTE="OLTC is fixed at tap 1.0 and not optimized in v1_2.",
                ESS_SIMULTANEOUS_MODE="loss_penalty", EPS_ESS_LOSS=1e-6,
                STAGE_I_PQ_RANGE_IMPLEMENTED=False, STAGE_II_ACTIVE_TRAJECTORY_IMPLEMENTED=True,
                assumption_line_smax_mva=3.5,
                ess_soc_assumption="ESS SOC parameters are completed by assumptions for flexibility aggregation experiments.",
                decoupling_note="The current decoupling is a conservative margin allocation heuristic satisfying h_G_dec + h_B_dec <= h_net, not the full robust optimization decomposition in Wang & Wu (2021).",
                OutputFlag=0, TimeLimit=120, FeasibilityTol=1e-7, MIPGap=1e-6,
                n_random_checks=12, n_sensitivity_validation=8, RUN_SELF_TEST_ONLY=False, random_seed=7,
                result_note="initial / no-shrink / not guaranteed safe disaggregation")


def build_case_ieee33(config=None) -> Dict:
    # Enhanced IEEE-33 benchmark mode: branch r/x are in ohm and converted to p.u. with
    # Z_BASE = BASE_KV^2 / BASE_MVA. Power variables remain MW/MVAr; voltage drop uses P_pu=P_MW/BASE_MVA.
    if config is None: config = config_default()
    raw = [(1,2,.0922,.0470),(2,3,.4930,.2511),(3,4,.3660,.1864),(4,5,.3811,.1941),(5,6,.8190,.7070),
           (6,7,.1872,.6188),(7,8,.7114,.2351),(8,9,1.0300,.7400),(9,10,1.0440,.7400),(10,11,.1966,.0650),
           (11,12,.3744,.1238),(12,13,1.4680,1.1550),(13,14,.5416,.7129),(14,15,.5910,.5260),(15,16,.7463,.5450),
           (16,17,1.2890,1.7210),(17,18,.7320,.5740),(2,19,.1640,.1565),(19,20,1.5042,1.3554),(20,21,.4095,.4784),
           (21,22,.7089,.9373),(3,23,.4512,.3083),(23,24,.8980,.7091),(24,25,.8960,.7011),(6,26,.2030,.1034),
           (26,27,.2842,.1447),(27,28,1.0590,.9337),(28,29,.8042,.7006),(29,30,.5075,.2585),(30,31,.9744,.9630),
           (31,32,.3105,.3619),(32,33,.3410,.5302)]
    zbase = config['Z_BASE']; smax = config['assumption_line_smax_mva']
    branches = np.array([[a-1,b-1,r/zbase,x/zbase,smax] for a,b,r,x in raw], float)
    nbus, nbranch = 33, len(branches)
    parent = {int(to): int(fr) for fr,to,_,_,_ in branches}; children = {i: [] for i in range(nbus)}
    for l,(fr,to,_,_,_) in enumerate(branches): children[int(fr)].append((int(to),l))
    # Table-I IEEE-33 loads in MW/MVAr. Bus 17 is 0.06 MW / 0.02 MVAr and bus 18 remains 0.09 MW / 0.04 MVAr; totals are P=3.715 MW, Q=2.300 MVAr.
    p=np.zeros(nbus); q=np.zeros(nbus)
    vals={2:(.100,.060),3:(.090,.040),4:(.120,.080),5:(.060,.030),6:(.060,.020),7:(.200,.100),8:(.200,.100),9:(.060,.020),10:(.060,.020),11:(.045,.030),12:(.060,.035),13:(.060,.035),14:(.120,.080),15:(.060,.010),16:(.060,.020),17:(.060,.020),18:(.090,.040),19:(.090,.040),20:(.090,.040),21:(.090,.040),22:(.090,.040),23:(.090,.050),24:(.420,.200),25:(.420,.200),26:(.060,.025),27:(.060,.025),28:(.060,.020),29:(.120,.070),30:(.200,.600),31:(.150,.070),32:(.210,.100),33:(.060,.040)}
    for b,(pp,qq) in vals.items(): p[b-1]=pp; q[b-1]=qq
    if abs(p.sum()-3.715)>1e-6 or abs(q.sum()-2.300)>1e-6:
        raise RuntimeError(f'IEEE-33 load totals mismatch: P={p.sum():.9f}, Q={q.sum():.9f}')
    return dict(nbus=nbus,nbranch=nbranch,branches=branches,parent=parent,children=children,p_load_base=p,q_load_base=q,
                total_P_load=float(p.sum()),total_Q_load=float(q.sum()),line_smax_mva=smax,parameter_note=config['CASE_PARAM_MODE'])


def build_profiles(T, DT):
    h=np.arange(T); load=0.72+0.24*np.sin((h-7)/24*2*np.pi)+0.10*np.sin((h-18)/24*4*np.pi); load=np.clip(load,.55,1.08)
    price=45+25*((h>=17)&(h<=21))+10*((h>=8)&(h<=15))
    return dict(load_mult=load, price=price)


def build_resources(config=None):
    if config is None: config = config_default()
    if config.get('CASE_PARAM_MODE') == 'demo_flex':
        return dict(parameter_note='demo parameters, not paper benchmark',
                    G=[dict(bus=7,pmin=.00,pmax=.65,qmin=-.25,qmax=.25,rup=.35,rdn=.35,cost=18), dict(bus=24,pmin=.05,pmax=.85,qmin=-.30,qmax=.30,rup=.40,rdn=.40,cost=22), dict(bus=30,pmin=.00,pmax=.55,qmin=-.20,qmax=.20,rup=.28,rdn=.28,cost=25)],
                    B=[dict(bus=14,pch=.45,pdis=.45,emin=.15,emax=1.60,e0=.80,eta_ch=.95,eta_dis=.95,cost=1.5), dict(bus=28,pch=.35,pdis=.35,emin=.10,emax=1.20,e0=.60,eta_ch=.95,eta_dis=.95,cost=1.5)],
                    RPC=[])
    qcap = .0 if not config.get('ALLOW_DG_Q_SUPPORT', False) else .10
    G=[dict(bus=b-1,pmin=0.0,pmax=.2,qmin=-qcap,qmax=qcap,rup=.2,rdn=.2,cost=20) for b in [18,22,25,33]]
    # ESS SOC parameters are completed by assumptions for flexibility aggregation experiments.
    B=[dict(bus=b-1,pch=.1,pdis=.1,emin=.02,emax=.2,e0=.1,eta_ch=.95,eta_dis=.95,cost=1.5) for b in [18,22,25,33]]
    RPC=[dict(bus=18-1,qmin=0.0,qmax=0.0,note='RPC configurable placeholder; fixed zero MVAr unless Table III values are provided'),
         dict(bus=33-1,qmin=0.0,qmax=0.0,note='RPC configurable placeholder; fixed zero MVAr unless Table III values are provided')]
    return dict(parameter_note='enhanced33_paper benchmark defaults; DG q capacity defaults to zero per Table-II reactive capacity assumption',G=G,B=B,RPC=RPC)


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
    P=D@netp; Q=D@netq; v=1.0-2*(A@(r*(P/1.0)+x*(Q/1.0)))
    p0=float(P[0]) if len(P) else float(netp.sum()); q0=float(Q[0]) if len(Q) else float(netq.sum())
    return P,Q,v,p0,q0


def require_gurobi():
    if gp is None:
        raise SystemExit('gurobipy/Gurobi is required for optimization mode. Set RUN_SELF_TEST_ONLY=True for no-Gurobi data checks. Import failed: %s' % GUROBI_IMPORT_ERROR)

def solve_base_dispatch_gurobi(case, profiles, resources, config):
    require_gurobi()
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
            v=1.0-sum(2*A[i,ell]*(r[ell]*(sum(D[ell,j]*(-pexpr[j]) for j in range(nbus))/config['BASE_MVA'])+x[ell]*(sum(D[ell,j]*(-qexpr[j]) for j in range(nbus))/config['BASE_MVA'])) for ell in range(nl))
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
                sensp=2*sum(A[i,l]*r[l]*D[l,R['bus']]/config['BASE_MVA'] for l in range(nl)); sensq=2*sum(A[i,l]*x[l]*D[l,R['bus']]/config['BASE_MVA'] for l in range(nl))
                cg[g*T+t]=sensp; cg[(ng+g)*T+t]=sensq
            for b,R in enumerate(B): cb[b*T+t]=2*sum(A[i,l]*r[l]*D[l,R['bus']]/config['BASE_MVA'] for l in range(nl))
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
    m,ch,dis,e,pagg=build_b_model(data); B=data['resources']['B']; nb=len(B); T=config['T']; EB=sum(pagg[i]*config['DT'] for i in range(t+1)); expr=pagg[t] if 'p_' in bound_type else EB
    throughput=gp.quicksum(ch[b,i]+dis[b,i] for b in range(nb) for i in range(T))
    if config.get('ESS_SIMULTANEOUS_MODE') == 'loss_penalty':
        obj = expr - config['EPS_ESS_LOSS']*throughput if bound_type.endswith('up') else expr + config['EPS_ESS_LOSS']*throughput
    else:
        obj = expr
    m.setObjective(obj, GRB.MAXIMIZE if bound_type.endswith('up') else GRB.MINIMIZE); st=time.time(); m.optimize(); rt=time.time()-st; ensure_optimal(m,'b_'+bound_type)
    pa=np.array([pagg[i].getValue() for i in range(T)])
    pChv=np.array([[ch[b,i].X for i in range(T)] for b in range(nb)]); pDisv=np.array([[dis[b,i].X for i in range(T)] for b in range(nb)])
    return dict(status=m.Status, objective=float(expr.getValue()), true_objective=float(expr.getValue()), penalized_objective=m.ObjVal, runtime=rt, pB_agg=pa, EB_agg=np.cumsum(pa)*config['DT'], pCh=pChv, pDis=pDisv, eB=np.array([[e[b,i].X for i in range(T+1)] for b in range(nb)]), simultaneous_ch_dis_max=float(np.max(np.minimum(pChv,pDisv))))

def solve_all_omega_b_bounds(data, config):
    log('[5/7] Solving Ω^B initial VB boundary LPs...'); T=config['T']; sol={}; bounds=np.zeros((T,4))
    for t in range(T):
        for typ,col in [('p_dn',0),('p_up',1),('e_dn',2),('e_up',3)]: log(f'  B {typ} t={t+1}/{T}'); r=solve_omega_b_bound(typ,t,data,config); sol[f'{typ}_{t+1}']=r; bounds[t,col]=r['true_objective']
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

def validate_network_sensitivity_matrices(case, base, resources, config, H_G, H_B, row_meta):
    log('[3b/7] Validating network sensitivity matrices by LinDistFlow recomputation...')
    rng=np.random.default_rng(config['random_seed']); T=config['T']; G=resources['G']; B=resources['B']; ng=len(G); nb=len(B)
    bytype={m['type']: [] for m in row_meta}
    for _ in range(config.get('n_sensitivity_validation',8)):
        dyG=np.zeros(2*ng*T); dyB=np.zeros(nb*T)
        for g,R in enumerate(G):
            dyG[g*T:(g+1)*T]=rng.uniform(-.02,.02,T); dyG[(ng+g)*T:(ng+g+1)*T]=rng.uniform(-.005,.005,T)
        for b,R in enumerate(B): dyB[b*T:(b+1)*T]=rng.uniform(-.02,.02,T)
        pred=H_G@dyG + H_B@dyB
        actual=[]
        for t in range(T):
            pinj=np.zeros(case['nbus']); qinj=np.zeros(case['nbus'])
            for g,R in enumerate(G): pinj[R['bus']]+=base['pG'][g,t]+dyG[g*T+t]; qinj[R['bus']]+=base['qG'][g,t]+dyG[(ng+g)*T+t]
            for b,R in enumerate(B): pinj[R['bus']]+=base['pB'][b,t]+dyB[b*T+t]
            # Use zero load delta: compare constraint expression changes around base recomputation with same load profile implicit in base P/Q/V.
            dP=np.zeros(case['nbranch']); dQ=np.zeros(case['nbranch']); D=downstream_matrix(case); A=path_matrix(case); br=case['branches']; rr=br[:,2]; xx=br[:,3]
            dp=np.zeros(case['nbus']); dq=np.zeros(case['nbus'])
            for g,R in enumerate(G): dp[R['bus']]+=dyG[g*T+t]; dq[R['bus']]+=dyG[(ng+g)*T+t]
            for b,R in enumerate(B): dp[R['bus']]+=dyB[b*T+t]
            dP=-(D@dp); dQ=-(D@dq); dV=-2*(A@(rr*(dP/config['BASE_MVA'])+xx*(dQ/config['BASE_MVA'])))
            for i in range(case['nbus']): actual += [dV[i], -dV[i]]
            for l in range(case['nbranch']): actual += [dP[l], -dP[l], dQ[l], -dQ[l]]
        actual=np.array(actual)
        for i,m in enumerate(row_meta): bytype[m['type']].append(abs(pred[i]-actual[i]))
    rows=[]; warn=False
    for typ,errs in sorted(bytype.items()):
        mx=float(np.max(errs)); mean=float(np.mean(errs)); warn = warn or mx>1e-6; rows.append([typ,mx,mean,len(errs)])
    if warn: log('WARNING: sensitivity validation max error exceeds 1e-6; inspect sensitivity_validation.csv')
    return rows

def build_case_parameter_summary(case, resources, config):
    return [['CASE_PARAM_MODE',config['CASE_PARAM_MODE']],['BASE_KV',config['BASE_KV']],['BASE_MVA',config['BASE_MVA']],['Z_BASE',config['Z_BASE']],['total_P_load',case['total_P_load']],['total_Q_load',case['total_Q_load']],['bus17_PQ',f"P={case['p_load_base'][16]}, Q={case['q_load_base'][16]}"],['bus18_PQ',f"P={case['p_load_base'][17]}, Q={case['q_load_base'][17]}"],['DG buses/capacity', ';'.join(f"{r['bus']+1}:{r['pmax']}MW:q[{r['qmin']},{r['qmax']}]" for r in resources['G'])],['ESS buses/power/energy',';'.join(f"{r['bus']+1}:pch={r['pch']},pdis={r['pdis']},e=[{r['emin']},{r['emax']}],e0={r['e0']}" for r in resources['B'])],['ESS SOC assumption',config['ess_soc_assumption']],['RPC status',config['RPC_NOTE']+' placeholder_buses='+str(config['RPC_PLACEHOLDER_BUSES'])],['OLTC status',config['OLTC_NOTE']],['line rating assumption',f"Smax={config['assumption_line_smax_mva']} MVA where Table IV is not provided"],['stage_i_pq_range_implemented',config['STAGE_I_PQ_RANGE_IMPLEMENTED']],['stage_ii_active_trajectory_implemented',config['STAGE_II_ACTIVE_TRAJECTORY_IMPLEMENTED']],['decoupling_note',config['decoupling_note']]]


def validate_dimensions(case, resources, config, H_G=None, H_B=None, h_net=None, row_meta=None, h_G_dec=None, h_B_dec=None):
    rows=[]
    def add(check, ok, detail): rows.append([check, bool(ok), detail])
    T=config['T']; ng=len(resources['G']); nb=len(resources['B'])
    add('load_vector_length', len(case['p_load_base'])==case['nbus'] and len(case['q_load_base'])==case['nbus'], f"nbus={case['nbus']}")
    add('branch_matrix_shape', case['branches'].shape==(case['nbranch'],5), str(case['branches'].shape))
    add('gder_bus_indices_valid', all(0 <= r['bus'] < case['nbus'] for r in resources['G']), [r['bus'] for r in resources['G']])
    add('bder_bus_indices_valid', all(0 <= r['bus'] < case['nbus'] for r in resources['B']), [r['bus'] for r in resources['B']])
    add('positive_branch_rx', bool(np.all(case['branches'][:,2] > 0) and np.all(case['branches'][:,3] > 0)), 'r/x > 0')
    add('positive_time_and_base', T > 0 and config['DT'] > 0 and config['BASE_MVA'] > 0 and config['BASE_KV'] > 0, f"T={T},DT={config['DT']},BASE_MVA={config['BASE_MVA']},BASE_KV={config['BASE_KV']}")
    if H_G is not None:
        add('network_row_counts', H_G.shape[0]==H_B.shape[0]==len(h_net)==len(row_meta), f"HG={H_G.shape}, HB={H_B.shape}, h={len(h_net)}, meta={len(row_meta)}")
        add('H_G_column_count', H_G.shape[1]==2*ng*T, f"actual={H_G.shape[1]}, expected={2*ng*T}")
        add('H_B_column_count', H_B.shape[1]==nb*T, f"actual={H_B.shape[1]}, expected={nb*T}")
        add('finite_network_arrays', np.all(np.isfinite(H_G)) and np.all(np.isfinite(H_B)) and np.all(np.isfinite(h_net)), 'finite H_G/H_B/h_net')
    if h_G_dec is not None:
        add('decoupled_margins_valid', np.all(h_G_dec+h_B_dec <= h_net+1e-8), f"max_violation={float(np.max(h_G_dec+h_B_dec-h_net))}")
    return rows

def run_self_test(config, out):
    log('[SELF-TEST] Running data/dimension checks without Gurobi...')
    if out.exists(): raise SystemExit(f'Output directory {out} already exists; refusing to overwrite existing files.')
    out.mkdir(exist_ok=False)
    case=build_case_ieee33(config); profiles=build_profiles(config['T'],config['DT']); resources=build_resources(config)
    D=downstream_matrix(case); A=path_matrix(case)
    rows=validate_dimensions(case,resources,config)
    rows += [['total_P_load_check', abs(case['total_P_load']-3.715)<=1e-6, case['total_P_load']], ['total_Q_load_check', abs(case['total_Q_load']-2.300)<=1e-6, case['total_Q_load']], ['downstream_matrix_shape', D.shape==(case['nbranch'],case['nbus']), str(D.shape)], ['path_matrix_shape', A.shape==(case['nbus'],case['nbranch']), str(A.shape)]]
    write_csv(out/'self_test_summary.csv',['check','passed','detail'],rows)
    write_csv(out/'case_parameter_summary.csv',['field','value'],build_case_parameter_summary(case,resources,config))
    write_csv(out/'bus_load_table_used.csv',['bus','P_load_MW','Q_load_MVAr'],[[i+1,case['p_load_base'][i],case['q_load_base'][i]] for i in range(case['nbus'])])
    log('[SELF-TEST] Done. No Gurobi calls were made.')

def write_csv(path, header, rows):
    with open(path,'w',newline='',encoding='utf-8') as f: w=csv.writer(f); w.writerow(header); w.writerows(rows)

def export_results(out, case, profiles, resources, config, data, bg, bb, Ag,bgpoly,Ab,bbpoly, randG,randB, chkG,chkB, sens_rows):
    log('[6/7] Exporting CSV/NPZ results...'); out.mkdir(exist_ok=False); (out/'figures').mkdir(); (out/'omega_g_solutions').mkdir(); (out/'omega_b_solutions').mkdir()
    json.dump(config,open(out/'config.json','w'),indent=2)
    write_csv(out/'case_parameter_summary.csv',['field','value'],build_case_parameter_summary(case,resources,config))
    write_csv(out/'bus_load_table_used.csv',['bus','P_load_MW','Q_load_MVAr'],[[i+1,case['p_load_base'][i],case['q_load_base'][i]] for i in range(case['nbus'])])
    write_csv(out/'self_test_summary.csv',['check','passed','detail'],validate_dimensions(case,resources,config))
    T=config['T']; write_csv(out/'base_dispatch.csv',['t','p0_base','q0_base'],[[t+1,data['base']['p0'][t],data['base']['q0'][t]] for t in range(T)])
    write_csv(out/'network_margins.csv',['row','type','time','index','base_value','h_net'],[[i+1,m['type'],m['time'],m['index'],m['base_value'],data['h_net'][i]] for i,m in enumerate(data['row_meta'])])
    write_csv(out/'decoupling_summary.csv',['row','type','time','index','h_net','h_G_dec','h_B_dec','range_G','range_B'],[[i+1,m['type'],m['time'],m['index'],data['h_net'][i],data['h_G_dec'][i],data['h_B_dec'][i],data['range_G'][i],data['range_B'][i]] for i,m in enumerate(data['row_meta'])])
    valid=bool(np.all(data['h_G_dec']+data['h_B_dec'] <= data['h_net']+1e-8)); write_csv(out/'decoupling_validity_check.csv',['check','value'],[['h_G_dec_plus_h_B_dec_le_h_net',valid],['max_violation',float(np.max(data['h_G_dec']+data['h_B_dec']-data['h_net']))],['note',config['decoupling_note']]])
    write_csv(out/'sensitivity_validation.csv',['constraint_type','max_abs_error','mean_abs_error','n_tests'],sens_rows)
    write_csv(out/'dimension_check_summary.csv',['check','passed','detail'],validate_dimensions(case,resources,config,data['H_G'],data['H_B'],data['h_net'],data['row_meta'],data['h_G_dec'],data['h_B_dec']))
    write_csv(out/'omega_g_bounds.csv',['t','pG_dn','pG_up','rG_dn','rG_up'],[[t+1,*bg['table'][t]] for t in range(T)])
    write_csv(out/'omega_b_bounds.csv',['t','pB_dn','pB_up','eB_dn','eB_up'],[[t+1,*bb['table'][t]] for t in range(T)])
    np.savetxt(out/'poly_g_init_A.csv',Ag,delimiter=','); np.savetxt(out/'poly_g_init_b.csv',bgpoly,delimiter=','); np.savetxt(out/'poly_b_init_A.csv',Ab,delimiter=','); np.savetxt(out/'poly_b_init_b.csv',bbpoly,delimiter=',')
    logs=[]
    for name,r in {**bg['solutions'],**bb['solutions']}.items(): logs.append([name,r['status'],r.get('true_objective',r.get('objective')),r.get('penalized_objective',r.get('objective')),r['runtime'],r.get('simultaneous_ch_dis_max',np.nan)])
    write_csv(out/'solver_log.csv',['problem','status','true_objective','penalized_objective','runtime_sec','simultaneous_ch_dis_max'],logs)
    write_csv(out/'no_shrink_feasibility_check_g.csv',['sample','feasible','status'],[[i+1,chkG[i][0],chkG[i][1]] for i in range(len(chkG))])
    write_csv(out/'no_shrink_feasibility_check_b.csv',['sample','feasible','status'],[[i+1,chkB[i][0],chkB[i][1]] for i in range(len(chkB))])
    essdiag=[]
    for name,r in bb['solutions'].items(): essdiag.append([name,r.get('simultaneous_ch_dis_max',np.nan)])
    write_csv(out/'ess_simultaneous_diagnostic.csv',['solution','simultaneous_ch_dis_max'],essdiag)
    msg='This version does not perform bound shrinking. Infeasible candidate trajectories may exist inside Ω2_G_init or Ω2_B_init. These diagnostics are only for later shrink/NN-feedback development.'
    write_csv(out/'no_shrink_summary.csv',['field','value'],[['result_note',config['result_note']],['message',msg],['g_infeasible_ratio',1-np.mean([x[0] for x in chkG])],['b_infeasible_ratio',1-np.mean([x[0] for x in chkB])],['no_bound_shrinking','true'],['no_neural_network','true'],['stage_i_pq_range_implemented',config['STAGE_I_PQ_RANGE_IMPLEMENTED']],['stage_ii_active_trajectory_implemented',config['STAGE_II_ACTIVE_TRAJECTORY_IMPLEMENTED']],['decoupling_note',config['decoupling_note']]])
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
    config=config_default(); out=Path('results_mt_vgvb_py_v1_2')
    if config.get('RUN_SELF_TEST_ONLY'):
        run_self_test(config,out); return
    require_gurobi()
    if out.exists(): raise SystemExit(f'Output directory {out} already exists; refusing to overwrite existing files.')
    log('VG/VB first-stage initial/no-shrink reproduction (Python/Gurobi)')
    case=build_case_ieee33(config); profiles=build_profiles(config['T'],config['DT']); resources=build_resources(config); pre_rows=validate_dimensions(case,resources,config)
    if not all(r[1] for r in pre_rows):
        raise RuntimeError('Pre-optimization dimension check failed: '+str(pre_rows))
    base=solve_base_dispatch_gurobi(case,profiles,resources,config)
    H_G,H_B,h_net,row_meta=build_network_sensitivity_matrices(case,base,resources,config)
    hG,hB,rangeG,rangeB=decouple_network_constraints(H_G,H_B,h_net,config['NETWORK_DECOUPLING_MODE'],config,resources,base)
    assert np.all(hG+hB <= h_net+1e-8), 'Decoupling split violates h_G_dec + h_B_dec <= h_net'
    sens_rows=validate_network_sensitivity_matrices(case,base,resources,config,H_G,H_B,row_meta)
    data=dict(config=config,case=case,profiles=profiles,resources=resources,base=base,H_G=H_G,H_B=H_B,h_net=h_net,h_G_dec=hG,h_B_dec=hB,range_G=rangeG,range_B=rangeB,row_meta=row_meta)
    bg=solve_all_omega_g_bounds(data,config); bb=solve_all_omega_b_bounds(data,config)
    Ag,bgv=build_vg_polytope(bg['table'],config['T'],config['DT']); Ab,bbv=build_vb_polytope(bb['table'],config['T'],config['DT'])
    log('[6/7] Running no-shrink random objective diagnostics (no boundary modification)...')
    randG=sample_random_objective_trajectories(Ag,bgv,config['n_random_checks'],config); randB=sample_random_objective_trajectories(Ab,bbv,config['n_random_checks'],config)
    chkG=[check_disaggregation_g(r,data,config) for r in randG]; chkB=[check_disaggregation_b(r,data,config) for r in randB]
    export_results(out,case,profiles,resources,config,data,bg,bb,Ag,bgv,Ab,bbv,randG,randB,chkG,chkB,sens_rows); plot_results(out,data,bg,bb,randG,randB)
    log('Done. Results are initial / no-shrink / not guaranteed safe disaggregation.')

if __name__ == '__main__': main()

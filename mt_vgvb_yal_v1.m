function mt_vgvb_yal_v1()
% MT_VGVB_YAL_V1 First-stage VG/VB aggregate flexibility reproduction (MATLAB/YALMIP/Gurobi).
%
% Sign convention and scope:
% - pG_agg(t) is the aggregated active-power injection deviation of GDERs from the base dispatch.
% - pB_agg(t) is the aggregated active-power injection deviation of BDERs from the base dispatch.
% - Positive values mean increased injection into the distribution network, or equivalently reduced net load.
% - pVPP_flex(t) = pG_agg(t) + pB_agg(t).
% - This version only computes initial boundaries of pG_agg and pB_agg; it does not compute final safe VPP boundaries.
% - The VG/VB polytopes are initial / no-shrink approximations and are not guaranteed safe for disaggregation.
% - Not implemented here: bound shrinking, infeasible point search, paper-style (41)-(43), neural networks, BPINN datasets.
%
% Run in MATLAB: mt_vgvb_yal_v1
% Outputs: results_mt_vgvb_yal_v1/

fprintf('VG/VB first-stage initial/no-shrink reproduction (MATLAB/YALMIP/Gurobi)\n');
if exist('sdpvar','file') ~= 2, error('YALMIP is required but sdpvar was not found.'); end
if exist('gurobi','file') ~= 2, fprintf('Warning: gurobi() not found on MATLAB path; YALMIP will still try solver=gurobi.\n'); end
outdir = 'results_mt_vgvb_yal_v1';
if exist(outdir,'dir'), error('Output directory %s already exists; refusing to overwrite existing files.', outdir); end
mkdir(outdir); mkdir(fullfile(outdir,'figures')); mkdir(fullfile(outdir,'omega_g_solutions')); mkdir(fullfile(outdir,'omega_b_solutions'));
config = default_config();
caseData = build_case_ieee33(); profiles = build_profiles(config.T); resources = build_resources();
ops = sdpsettings('solver','gurobi','verbose',1,'gurobi.OutputFlag',config.OutputFlag,'gurobi.TimeLimit',config.TimeLimit,'gurobi.FeasibilityTol',config.FeasibilityTol);
base = solve_base_dispatch_yalmip(caseData, profiles, resources, config, ops);
[H_G,H_B,h_net,row_meta] = build_network_sensitivity_matrices(caseData, base, resources, config);
[h_G_dec,h_B_dec,range_G,range_B] = decouple_network_constraints(H_G,H_B,h_net,config.MODE_NETWORK_DECOUPLING,config,resources,base);
data = struct('caseData',caseData,'profiles',profiles,'resources',resources,'config',config,'base',base,'H_G',H_G,'H_B',H_B,'h_net',h_net,'h_G_dec',h_G_dec,'h_B_dec',h_B_dec,'range_G',range_G,'range_B',range_B,'row_meta',{row_meta});
bounds_g = solve_all_omega_g_bounds(data, config, ops, outdir);
bounds_b = solve_all_omega_b_bounds(data, config, ops, outdir);
[A_G,b_G] = build_vg_polytope(bounds_g.table, config.T, config.DT);
[A_B,b_B] = build_vb_polytope(bounds_b.table, config.T, config.DT);
fprintf('[6/7] Running no-shrink random objective diagnostics (no boundary modification)...\n');
randG = sample_random_objective_trajectories(A_G,b_G,config.n_random_checks,config,ops);
randB = sample_random_objective_trajectories(A_B,b_B,config.n_random_checks,config,ops);
chkG = check_disaggregation_g(randG,data,config,ops); chkB = check_disaggregation_b(randB,data,config,ops);
export_results(outdir, config, data, bounds_g, bounds_b, A_G,b_G,A_B,b_B, randG,randB,chkG,chkB);
plot_results(outdir, data, bounds_g, bounds_b, randG, randB);
fprintf('Done. Results are initial / no-shrink / not guaranteed safe disaggregation.\n');
end

function config = default_config()
config = struct(); config.CASE_NAME='ieee33'; config.T=24; config.DT=1.0; config.FAST_5MIN_TEST=false;
if config.FAST_5MIN_TEST, config.T=288; config.DT=5/60; end
config.V_MIN=0.95^2; config.V_MAX=1.05^2; config.V0=1.0^2; config.MODE_NETWORK_DECOUPLING='capacity_weighted'; config.alpha_net=0.5;
config.OutputFlag=0; config.TimeLimit=120; config.FeasibilityTol=1e-7; config.n_random_checks=12; config.random_seed=7;
config.result_note='initial / no-shrink / not guaranteed safe disaggregation';
end

function c = build_case_ieee33()
raw = [1 2 .0922 .0470;2 3 .4930 .2511;3 4 .3660 .1864;4 5 .3811 .1941;5 6 .8190 .7070;6 7 .1872 .6188;7 8 .7114 .2351;8 9 1.0300 .7400;9 10 1.0440 .7400;10 11 .1966 .0650;11 12 .3744 .1238;12 13 1.4680 1.1550;13 14 .5416 .7129;14 15 .5910 .5260;15 16 .7463 .5450;16 17 1.2890 1.7210;17 18 .7320 .5740;2 19 .1640 .1565;19 20 1.5042 1.3554;20 21 .4095 .4784;21 22 .7089 .9373;3 23 .4512 .3083;23 24 .8980 .7091;24 25 .8960 .7011;6 26 .2030 .1034;26 27 .2842 .1447;27 28 1.0590 .9337;28 29 .8042 .7006;29 30 .5075 .2585;30 31 .9744 .9630;31 32 .3105 .3619;32 33 .3410 .5302];
c.nbus=33; c.branches=[raw(:,1:2), raw(:,3:4)/100, 3.5*ones(size(raw,1),1)]; c.nbranch=size(c.branches,1);
c.p_load_base=zeros(c.nbus,1); vals=[2 .10;3 .09;4 .12;5 .06;6 .06;7 .20;8 .20;9 .06;10 .06;11 .045;12 .06;13 .06;14 .12;15 .06;16 .06;17 .09;18 .09;19 .09;20 .09;21 .09;22 .09;23 .09;24 .42;25 .42;26 .06;27 .06;28 .06;29 .12;30 .20;31 .15;32 .21;33 .06];
for k=1:size(vals,1), c.p_load_base(vals(k,1))=vals(k,2); end; c.q_load_base=.45*c.p_load_base;
end

function p = build_profiles(T)
h=(0:T-1)'; load=0.72+0.24*sin((h-7)/24*2*pi)+0.10*sin((h-18)/24*4*pi); p.load_mult=min(max(load,.55),1.08); p.price=45+25*((h>=17)&(h<=21))+10*((h>=8)&(h<=15));
end

function r = build_resources()
r.G = struct('bus',{8,25,31},'pmin',{0,.05,0},'pmax',{.65,.85,.55},'qmin',{-.25,-.30,-.20},'qmax',{.25,.30,.20},'rup',{.35,.40,.28},'rdn',{.35,.40,.28},'cost',{18,22,25});
r.B = struct('bus',{15,29},'pch',{.45,.35},'pdis',{.45,.35},'emin',{.15,.10},'emax',{1.60,1.20},'e0',{.80,.60},'eta_ch',{.95,.95},'eta_dis',{.95,.95},'cost',{1.5,1.5});
end

function [D,A] = topo_matrices(c)
D=zeros(c.nbranch,c.nbus); children=cell(c.nbus,1); brTo=zeros(c.nbus,1); parent=zeros(c.nbus,1);
for l=1:c.nbranch, f=c.branches(l,1); t=c.branches(l,2); children{f}=[children{f}; t l]; parent(t)=f; brTo(t)=l; end
for l=1:c.nbranch, fill(c.branches(l,2),l); end
A=zeros(c.nbus,c.nbranch); for b=2:c.nbus, cur=b; while cur~=1, bl=brTo(cur); A(b,bl)=1; cur=parent(cur); end, end
    function fill(node,br), D(br,node)=1; cc=children{node}; for ii=1:size(cc,1), fill(cc(ii,1),br); end, end
end

function base = solve_base_dispatch_yalmip(c,p,r,config,ops)
fprintf('[1/7] Solving full coupled LinDistFlow base dispatch with YALMIP/Gurobi...\n'); T=config.T; ng=numel(r.G); nb=numel(r.B); [D,A]=topo_matrices(c); rr=c.branches(:,3); xx=c.branches(:,4); S=c.branches(:,5);
pG=sdpvar(ng,T,'full'); qG=sdpvar(ng,T,'full'); ch=sdpvar(nb,T,'full'); dis=sdpvar(nb,T,'full'); e=sdpvar(nb,T+1,'full'); p0imp=sdpvar(T,1);
F=[p0imp>=0]; obj=0;
for g=1:ng, F=[F, r.G(g).pmin<=pG(g,:)<=r.G(g).pmax, r.G(g).qmin<=qG(g,:)<=r.G(g).qmax]; for t=2:T, F=[F, pG(g,t)-pG(g,t-1)<=r.G(g).rup*config.DT, pG(g,t-1)-pG(g,t)<=r.G(g).rdn*config.DT]; end, obj=obj+r.G(g).cost*sum(pG(g,:)); end
for b=1:nb, F=[F,e(b,1)==r.B(b).e0,e(b,T+1)==r.B(b).e0,r.B(b).emin<=e(b,:)<=r.B(b).emax,0<=ch(b,:)<=r.B(b).pch,0<=dis(b,:)<=r.B(b).pdis]; for t=1:T, F=[F,e(b,t+1)==e(b,t)+r.B(b).eta_ch*ch(b,t)*config.DT-dis(b,t)/r.B(b).eta_dis*config.DT]; end, obj=obj+r.B(b).cost*sum(ch(b,:)+dis(b,:)); end
for t=1:T
    pinj=sdpvar(c.nbus,1); qinj=sdpvar(c.nbus,1); F=[F,pinj==0,qinj==0];
    for g=1:ng, F=[F,pinj(r.G(g).bus)==pinj(r.G(g).bus)+pG(g,t), qinj(r.G(g).bus)==qinj(r.G(g).bus)+qG(g,t)]; end
    for b=1:nb, F=[F,pinj(r.B(b).bus)==pinj(r.B(b).bus)+dis(b,t)-ch(b,t)]; end
    pl=c.p_load_base*p.load_mult(t); ql=c.q_load_base*p.load_mult(t); P=D*(pl-pinj); Q=D*(ql-qinj); v=1-2*A*(rr.*P+xx.*Q);
    F=[F,-S<=P<=S,-S<=Q<=S,config.V_MIN<=v<=config.V_MAX,p0imp(t)>=sum(pl)-sum(pG(:,t))-sum(dis(:,t)-ch(:,t))]; obj=obj+p.price(t)*p0imp(t);
end
diag=optimize(F,obj,ops); if diag.problem~=0, error('Base dispatch failed: %s', diag.info); end
base.pG=value(pG); base.qG=value(qG); base.pCh=value(ch); base.pDis=value(dis); base.pB=base.pDis-base.pCh; base.eB=value(e); base.obj=value(obj);
base.P=zeros(c.nbranch,T); base.Q=zeros(c.nbranch,T); base.V=zeros(c.nbus,T); base.p0=zeros(T,1); base.q0=zeros(T,1);
for t=1:T, pinj=zeros(c.nbus,1); qinj=zeros(c.nbus,1); for g=1:ng, pinj(r.G(g).bus)=pinj(r.G(g).bus)+base.pG(g,t); qinj(r.G(g).bus)=qinj(r.G(g).bus)+base.qG(g,t); end; for b=1:nb, pinj(r.B(b).bus)=pinj(r.B(b).bus)+base.pB(b,t); end; pl=c.p_load_base*p.load_mult(t); ql=c.q_load_base*p.load_mult(t); base.P(:,t)=D*(pl-pinj); base.Q(:,t)=D*(ql-qinj); base.V(:,t)=1-2*A*(rr.*base.P(:,t)+xx.*base.Q(:,t)); base.p0(t)=sum(pl)-sum(pinj); base.q0(t)=sum(ql)-sum(qinj); end
end

function [HG,HB,h,meta] = build_network_sensitivity_matrices(c,base,r,config)
fprintf('[2/7] Building LinDistFlow network sensitivity matrices...\n'); T=config.T; ng=numel(r.G); nb=numel(r.B); [D,A]=topo_matrices(c); rr=c.branches(:,3); xx=c.branches(:,4); S=c.branches(:,5); nyG=2*ng*T; nyB=nb*T; HG=[]; HB=[]; h=[]; meta={};
for t=1:T
 for i=1:c.nbus, cg=zeros(1,nyG); cb=zeros(1,nyB); for g=1:ng, sp=2*sum(A(i,:)'.*rr.*D(:,r.G(g).bus)); sq=2*sum(A(i,:)'.*xx.*D(:,r.G(g).bus)); cg((g-1)*T+t)=sp; cg((ng+g-1)*T+t)=sq; end; for b=1:nb, cb((b-1)*T+t)=2*sum(A(i,:)'.*rr.*D(:,r.B(b).bus)); end; add('voltage_upper',t,i,base.V(i,t),config.V_MAX-base.V(i,t),cg,cb); add('voltage_lower',t,i,-base.V(i,t),base.V(i,t)-config.V_MIN,-cg,-cb); end
 for l=1:c.nbranch, cg=zeros(1,nyG); cb=zeros(1,nyB); for g=1:ng, cg((g-1)*T+t)=-D(l,r.G(g).bus); end; for b=1:nb, cb((b-1)*T+t)=-D(l,r.B(b).bus); end; add('lineP_upper',t,l,base.P(l,t),S(l)-base.P(l,t),cg,cb); add('lineP_lower',t,l,-base.P(l,t),S(l)+base.P(l,t),-cg,-cb); cq=zeros(1,nyG); for g=1:ng, cq((ng+g-1)*T+t)=-D(l,r.G(g).bus); end; add('lineQ_upper',t,l,base.Q(l,t),S(l)-base.Q(l,t),cq,zeros(1,nyB)); add('lineQ_lower',t,l,-base.Q(l,t),S(l)+base.Q(l,t),-cq,zeros(1,nyB)); end
end
if any(h < -1e-6), error('Base point violates network constraints; negative h_net detected.'); end; h=max(h,0);
    function add(type,t,idx,baseval,margin,cg,cb), HG=[HG;cg]; HB=[HB;cb]; h=[h;margin]; meta{end+1}=struct('type',type,'time',t,'index',idx,'base_value',baseval,'remaining_margin',margin); end
end

function [hG,hB,rangeG,rangeB] = decouple_network_constraints(HG,HB,h,mode,config,r,base)
fprintf('[3/7] Decoupling network margins (%s)...\n',mode); T=config.T; ng=numel(r.G); nb=numel(r.B); lbG=[]; ubG=[]; for g=1:ng, lbG=[lbG, r.G(g).pmin-base.pG(g,:)]; ubG=[ubG, r.G(g).pmax-base.pG(g,:)]; end; for g=1:ng, lbG=[lbG, r.G(g).qmin-base.qG(g,:)]; ubG=[ubG, r.G(g).qmax-base.qG(g,:)]; end; lbB=[]; ubB=[]; for b=1:nb, lbB=[lbB, -r.B(b).pch-base.pB(b,:)]; ubB=[ubB, r.B(b).pdis-base.pB(b,:)]; end
rangeG=max(HG,0)*ubG' + min(HG,0)*lbG'; rangeB=max(HB,0)*ubB' + min(HB,0)*lbB'; rangeG=max(rangeG,0); rangeB=max(rangeB,0);
if strcmp(mode,'equal_margin'), hG=config.alpha_net*h; hB=h-hG; else, den=rangeG+rangeB; frac=.5*ones(size(h)); idx=den>1e-10; frac(idx)=rangeG(idx)./den(idx); hG=h.*frac; hB=h-hG; end
end

function [F,pG,qG,pagg] = g_constraints(data, candidate)
T=data.config.T; r=data.resources; ng=numel(r.G); pG=sdpvar(ng,T,'full'); qG=sdpvar(ng,T,'full'); F=[];
for g=1:ng, F=[F,r.G(g).pmin<=pG(g,:)<=r.G(g).pmax,r.G(g).qmin<=qG(g,:)<=r.G(g).qmax]; for t=2:T, F=[F,pG(g,t)-pG(g,t-1)<=r.G(g).rup*data.config.DT,pG(g,t-1)-pG(g,t)<=r.G(g).rdn*data.config.DT]; end, end
y=[]; for g=1:ng, y=[y, pG(g,:)-data.base.pG(g,:)]; end; for g=1:ng, y=[y, qG(g,:)-data.base.qG(g,:)]; end; F=[F,data.H_G*y'<=data.h_G_dec]; pagg=sum(pG-data.base.pG,1); if nargin>1, F=[F,pagg==candidate(:)']; end
end
function [F,ch,dis,e,pagg] = b_constraints(data, candidate)
T=data.config.T; r=data.resources; nb=numel(r.B); ch=sdpvar(nb,T,'full'); dis=sdpvar(nb,T,'full'); e=sdpvar(nb,T+1,'full'); F=[];
for b=1:nb, F=[F,e(b,1)==r.B(b).e0,e(b,T+1)==r.B(b).e0,r.B(b).emin<=e(b,:)<=r.B(b).emax,0<=ch(b,:)<=r.B(b).pch,0<=dis(b,:)<=r.B(b).pdis]; for t=1:T, F=[F,e(b,t+1)==e(b,t)+r.B(b).eta_ch*ch(b,t)*data.config.DT-dis(b,t)/r.B(b).eta_dis*data.config.DT]; end, end
y=[]; for b=1:nb, y=[y, dis(b,:)-ch(b,:)-data.base.pB(b,:)]; end; F=[F,data.H_B*y'<=data.h_B_dec]; pagg=sum(dis-ch-data.base.pB,1); if nargin>1, F=[F,pagg==candidate(:)']; end
end

function bounds = solve_all_omega_g_bounds(data,config,ops,outdir)
fprintf('[4/7] Solving Ω^G initial VG boundary LPs...\n'); T=config.T; tbl=nan(T,4); logRows={};
for t=1:T, [tbl(t,1),logRows]=solveG('p_dn',t,-1,tbl,logRows); [tbl(t,2),logRows]=solveG('p_up',t,1,tbl,logRows); if t>1, [tbl(t,3),logRows]=solveG('r_dn',t,-1,tbl,logRows); [tbl(t,4),logRows]=solveG('r_up',t,1,tbl,logRows); end, end; tbl(1,3:4)=0; bounds.table=tbl; bounds.log=logRows;
    function [val,logRows]=solveG(name,t,sense,~,logRows), fprintf('  G %s t=%d/%d\n',name,t,T); [F,pG,qG,pagg]=g_constraints(data); obj=pagg(t); if name(1)=='r', obj=pagg(t)-pagg(t-1); end; tic; diag=optimize(F,-sense*obj,ops); rt=toc; if diag.problem~=0, error('G bound failed: %s',diag.info); end; val=value(obj); writematrix(value(pagg)',fullfile(outdir,'omega_g_solutions',[name '_' num2str(t) '.csv'])); logRows(end+1,:)={['G_' name '_' num2str(t)],diag.problem,val,rt}; end
end
function bounds = solve_all_omega_b_bounds(data,config,ops,outdir)
fprintf('[5/7] Solving Ω^B initial VB boundary LPs...\n'); T=config.T; tbl=zeros(T,4); logRows={};
for t=1:T, [tbl(t,1),logRows]=solveB('p_dn',t,-1,logRows); [tbl(t,2),logRows]=solveB('p_up',t,1,logRows); [tbl(t,3),logRows]=solveB('e_dn',t,-1,logRows); [tbl(t,4),logRows]=solveB('e_up',t,1,logRows); end; bounds.table=tbl; bounds.log=logRows;
    function [val,logRows]=solveB(name,t,sense,logRows), fprintf('  B %s t=%d/%d\n',name,t,T); [F,ch,dis,e,pagg]=b_constraints(data); EB=sum(pagg(1:t))*config.DT; obj=pagg(t); if name(1)=='e', obj=EB; end; tic; diag=optimize(F,-sense*obj,ops); rt=toc; if diag.problem~=0, error('B bound failed: %s',diag.info); end; val=value(obj); pa=value(pagg); writematrix([pa(:), cumsum(pa(:))*config.DT],fullfile(outdir,'omega_b_solutions',[name '_' num2str(t) '.csv'])); logRows(end+1,:)={['B_' name '_' num2str(t)],diag.problem,val,rt}; end
end

function [A,b]=build_vg_polytope(tbl,T,DT), A=[]; b=[]; for t=1:T, e=zeros(1,T); e(t)=1; A=[A;e;-e]; b=[b;tbl(t,2);-tbl(t,1)]; end; for t=2:T, e=zeros(1,T); e(t)=1; e(t-1)=-1; A=[A;e;-e]; b=[b;tbl(t,4);-tbl(t,3)]; end, end
function [A,b]=build_vb_polytope(tbl,T,DT), A=[]; b=[]; for t=1:T, e=zeros(1,T); e(t)=1; A=[A;e;-e]; b=[b;tbl(t,2);-tbl(t,1)]; c=zeros(1,T); c(1:t)=DT; A=[A;c;-c]; b=[b;tbl(t,4);-tbl(t,3)]; end, end
function X=sample_random_objective_trajectories(A,b,n,config,ops), rng(config.random_seed); T=size(A,2); X=nan(n,T); for s=1:n, x=sdpvar(T,1); u=randn(T,1); diag=optimize([A*x<=b],-u'*x,ops); if diag.problem==0, X(s,:)=value(x)'; end, end, end
function chk=check_disaggregation_g(X,data,config,ops), chk=zeros(size(X,1),2); for i=1:size(X,1), [F,~,~,~]=g_constraints(data,X(i,:)); d=optimize(F,[],ops); chk(i,:)=[d.problem==0,d.problem]; end, end
function chk=check_disaggregation_b(X,data,config,ops), chk=zeros(size(X,1),2); for i=1:size(X,1), [F,~,~,~,~]=b_constraints(data,X(i,:)); d=optimize(F,[],ops); chk(i,:)=[d.problem==0,d.problem]; end, end

function export_results(outdir,config,data,bg,bb,AG,bG,AB,bB,randG,randB,chkG,chkB)
fprintf('[6/7] Exporting CSV/MAT results...\n'); writematrix([(1:config.T)',data.base.p0(:),data.base.q0(:)],fullfile(outdir,'base_dispatch.csv'));
rows=cell(numel(data.row_meta),6); dec=cell(numel(data.row_meta),9); for i=1:numel(data.row_meta), m=data.row_meta{i}; rows(i,:)={i,m.type,m.time,m.index,m.base_value,data.h_net(i)}; dec(i,:)={i,m.type,m.time,m.index,data.h_net(i),data.h_G_dec(i),data.h_B_dec(i),data.range_G(i),data.range_B(i)}; end
writecell([{'row','type','time','index','base_value','h_net'};rows],fullfile(outdir,'network_margins.csv')); writecell([{'row','type','time','index','h_net','h_G_dec','h_B_dec','range_G','range_B'};dec],fullfile(outdir,'decoupling_summary.csv'));
writematrix([(1:config.T)',bg.table],fullfile(outdir,'omega_g_bounds.csv')); writematrix([(1:config.T)',bb.table],fullfile(outdir,'omega_b_bounds.csv')); writematrix(AG,fullfile(outdir,'poly_g_init_A.csv')); writematrix(bG,fullfile(outdir,'poly_g_init_b.csv')); writematrix(AB,fullfile(outdir,'poly_b_init_A.csv')); writematrix(bB,fullfile(outdir,'poly_b_init_b.csv'));
writecell([{'problem','status','objective','runtime_sec'}; bg.log; bb.log],fullfile(outdir,'solver_log.csv')); writematrix([(1:size(chkG,1))',chkG],fullfile(outdir,'no_shrink_feasibility_check_g.csv')); writematrix([(1:size(chkB,1))',chkB],fullfile(outdir,'no_shrink_feasibility_check_b.csv'));
msg='This version does not perform bound shrinking. Infeasible candidate trajectories may exist inside Ω2_G_init or Ω2_B_init. These diagnostics are only for later shrink/NN-feedback development.'; writecell({'field','value';'result_note',config.result_note;'message',msg;'g_infeasible_ratio',1-mean(chkG(:,1));'b_infeasible_ratio',1-mean(chkB(:,1));'no_bound_shrinking','true';'no_neural_network','true'},fullfile(outdir,'no_shrink_summary.csv'));
save(fullfile(outdir,'config.mat'),'config'); save(fullfile(outdir,'results_data.mat'),'data','bg','bb','AG','bG','AB','bB','randG','randB','chkG','chkB','-v7.3');
end

function plot_results(outdir,data,bg,bb,randG,randB)
fprintf('[7/7] Plotting figures...\n'); figdir=fullfile(outdir,'figures'); T=numel(data.base.p0); x=1:T;
figure('visible','off'); plot(x,data.base.p0); title('Base PCC active power'); xlabel('t'); ylabel('MW'); saveas(gcf,fullfile(figdir,'base_p0_profile.png')); close;
figure('visible','off'); plot(x,bg.table(:,1),x,bg.table(:,2)); legend('pG\_dn','pG\_up'); saveas(gcf,fullfile(figdir,'omega_g_power_bounds.png')); close;
figure('visible','off'); plot(x,bg.table(:,3),x,bg.table(:,4)); legend('rG\_dn','rG\_up'); saveas(gcf,fullfile(figdir,'omega_g_ramp_bounds.png')); close;
figure('visible','off'); plot(x,bb.table(:,1),x,bb.table(:,2)); legend('pB\_dn','pB\_up'); saveas(gcf,fullfile(figdir,'omega_b_power_bounds.png')); close;
figure('visible','off'); plot(x,bb.table(:,3),x,bb.table(:,4)); legend('eB\_dn','eB\_up'); saveas(gcf,fullfile(figdir,'omega_b_energy_bounds.png')); close;
types=unique(cellfun(@(m)m.type,data.row_meta,'UniformOutput',false)); vals=zeros(numel(types),3); for k=1:numel(types), idx=find(strcmp(cellfun(@(m)m.type,data.row_meta,'UniformOutput',false),types{k})); vals(k,:)=[mean(data.h_net(idx)),mean(data.h_G_dec(idx)),mean(data.h_B_dec(idx))]; end; figure('visible','off'); bar(vals); set(gca,'XTickLabel',types,'XTickLabelRotation',30); legend('h\_net','h\_G\_dec','h\_B\_dec'); saveas(gcf,fullfile(figdir,'network_margin_split.png')); close;
figure('visible','off'); plot(x,randG'); hold on; plot(x,bg.table(:,1),'k--',x,bg.table(:,2),'k--'); saveas(gcf,fullfile(figdir,'optional_random_trajs_g.png')); close;
figure('visible','off'); plot(x,randB'); hold on; plot(x,bb.table(:,1),'k--',x,bb.table(:,2),'k--'); saveas(gcf,fullfile(figdir,'optional_random_trajs_b.png')); close;
end

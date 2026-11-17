"""Joint inference with explicit covariance diagnostics and conditional resampling.
No inverse of an indefinite multiway matrix is used. Seed and bandwidths fixed.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import f,t
B=Path('green2026/revision');O=B/'outputs'
q=pd.read_parquet(O/'revised_panel_PRIVATE.parquet').sort_values(['pair','date']).reset_index(drop=True)
terms=['w'+str(i) for i in [-7,-6,-5,-4,-3,-2,0,1,2,3,4]]
def design(terms):
 a=q[['reconstructed_gap']+terms].astype(float)
 a-=a.groupby(q.pair).transform('mean')
 return a[terms].to_numpy(),a.reconstructed_gap.to_numpy()
X,y=design(terms);bread=np.linalg.inv(X.T@X);beta=bread@X.T@y;u=y-X@beta
n,k=X.shape;P=q.pair.nunique();T=q.date.nunique();df=min(P,T)-1
def meat(key,scores):
 s=pd.DataFrame(scores).groupby(np.asarray(key)).sum().to_numpy();G=len(s)
 return s.T@s*G/(G-1)*(n-1)/(n-P-k)
S=X*u[:,None];Vi=bread@meat(q.pair,S)@bread;Vt=bread@meat(q.date,S)@bread
Vint=bread@meat(q.pair+'|'+q.date.astype(str),S)@bread
Vraw=Vi+Vt-Vint;Vadd=Vi+Vt
Vraw=(Vraw+Vraw.T)/2;Vadd=(Vadd+Vadd.T)/2
assert np.linalg.eigvalsh(Vadd).min()>0
assert np.linalg.eigvalsh(Vadd-Vraw).min()>-1e-12
R=np.arange(6);F=float(beta[R]@np.linalg.solve(Vadd[np.ix_(R,R)],beta[R])/6)
# Conservative additive covariance retains overlap rather than subtracting it.
# Its inflation relative to the usual estimator is PSD, not universal coverage.
rng=np.random.default_rng(20261117)
draws=rng.multivariate_normal(np.zeros(k),Vadd,size=49999)
draws/=np.sqrt(rng.chisquare(df,size=len(draws))/df)[:,None]
se=np.sqrt(np.diag(Vadd));critical=float(np.quantile(np.max(abs(draws/se),axis=1),.95))
weekly=pd.DataFrame(dict(term=terms,coef=beta,se=se,lo=beta-critical*se,hi=beta+critical*se,
 p=2*t.sf(abs(beta/se),df),simultaneous_critical=critical))
weekly.to_csv(O/'weekly_simultaneous.csv',index=False)
# Null-imposed dependent Gaussian wild bootstrap: all pre-week coefficients zero,
# post-week coefficients and pair intercepts unrestricted. One multiplier per date
# shared by all issuers; Bartlett lag dependence; observed basket is held fixed.
X0=X[:,6:];b0=np.linalg.lstsq(X0,y,rcond=None)[0];u0=y-X0@b0
date_codes,dates=pd.factorize(q.date,sort=True)
score0=pd.DataFrame(X*u0[:,None]).groupby(date_codes).sum().to_numpy()
bootstrap=[];maxerrors=[]
for L in [3,7,10]:
 K=np.maximum(1-abs(np.arange(T)[:,None]-np.arange(T)[None,:])/(L+1),0)
 assert np.linalg.eigvalsh(K).min()>0
 W=np.random.default_rng(20261117+L).standard_normal((9999,T))@np.linalg.cholesky(K).T
 bstar=W@score0@bread
 # Exact FWL shortcut checked against an explicitly demeaned bootstrap outcome.
 ys=X0@b0+u0*W[0,date_codes]
 direct=np.linalg.lstsq(X,ys,rcond=None)[0]
 expected=np.r_[np.zeros(6),b0]+bstar[0]
 err=float(abs(direct-expected).max());assert err<1e-9;maxerrors.append(err)
 scale=bstar[:,:6].std(axis=0,ddof=1)
 observed=float(np.max(abs(beta[:6])/scale));sim=np.max(abs(bstar[:,:6])/scale,axis=1)
 pv=float((1+np.sum(sim>=observed))/(len(sim)+1))
 bootstrap.append(dict(lags=L,draws=len(sim),statistic=observed,p=pv,mc_se=float(np.sqrt(pv*(1-pv)/(len(sim)+1))),scope='conditional_on_observed_issuer_basket'))
pd.DataFrame(bootstrap).to_csv(O/'preweek_dependent_wild.csv',index=False)
counts=q.groupby(['pair','post']).size().unstack();chg=q.groupby(['pair','post']).reconstructed_gap.mean().unstack();diff=chg[True]-chg[False]
summary=dict(raw_min_eigenvalue=float(np.linalg.eigvalsh(Vraw).min()),raw_negative_eigenvalues=int((np.linalg.eigvalsh(Vraw)<-1e-12).sum()),
 preblock_min_eigenvalue=float(np.linalg.eigvalsh(Vraw[:6,:6]).min()),additive_min_eigenvalue=float(np.linalg.eigvalsh(Vadd).min()),
 joint_pre_F=F,joint_pre_df=[6,df],joint_pre_p=float(f.sf(F,6,df)),simultaneous_critical=critical,
 simultaneous_weeks_excluding_zero=weekly.loc[(weekly.lo>0)|(weekly.hi<0),'term'].tolist(),
 equal_pair_mean=float(diff.mean()),resampling_validation_max_error=max(maxerrors),seed=20261117,
 interpretation='Additive issuer/date covariance is a conservative sensitivity under the original cluster assumptions. Dependent-wild results are conditional time-series diagnostics, not multiway bootstrap or proof of parallel trends.')
(O/'joint_inference.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));print(pd.DataFrame(bootstrap).to_string(index=False))
np.savez_compressed(O/'weekly_covariances.npz',raw=Vraw,additive=Vadd,terms=np.array(terms))
pd.DataFrame([
 dict(ISIN=i,historical_RIC={'EU000A3K4C42':'XS239966560=','EU000A3K4DC0':'XS245745761='}[i],venue='MOT',venue_trading_type='Clean',frequency=1,day_count=dc,source=url,
 historical_LSEG_basis_verified=False,accessed='2026-11-17')
 for i,dc,url in [('EU000A3K4C42','ACT/ACT ANNUALLY','https://www.borsaitaliana.it/borsa/obbligazioni/mot/euro-obbligazioni/scheda/EU000A3K4C42-MOTX.html?lang=en'),('EU000A3K4DC0','ACT/ACT PERIODIC','https://www.borsaitaliana.it/borsa/obbligazioni/mot/euro-obbligazioni/scheda/EU000A3K4DC0-MOTX.html?lang=en')]
]).to_csv(O/'quote_convention_evidence.csv',index=False)

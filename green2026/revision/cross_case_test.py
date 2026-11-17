"""Direct NZ-minus-Australia contrast with date-aggregated HAC scores.
Fixed two-case inference; exploratory, not cross-country population inference.
"""
from pathlib import Path
import numpy as np,pandas as pd,json
from scipy.stats import t
B=Path('green2026/revision');O=B/'outputs'
d=pd.read_parquet(O/'core_local_curve_panel_PRIVATE.parquet')
external=pd.read_csv(O/'core_external_price_diagnostics_PRIVATE.csv')
e=external[external.basis.eq('clean')&external.settlement_weekdays.eq(2)].pivot(index=['issuer','date'],columns='side',values='predicted_price').reset_index()
e['date']=pd.to_datetime(e.date);e['external_gap']=100*np.log(e.g/e.c)
d=d.merge(e[['issuer','date','external_gap']],on=['issuer','date'],validate='one_to_one')
out=[];checks=[]
for sample,q0 in [('original_dates',d),('shared_dates',d[d.date.isin(d.groupby('date').issuer.nunique().loc[lambda x:x.eq(2)].index)])]:
 for outcome in ['reconstructed_gap','external_gap']:
  for trend in [False,True]:
   q=q0.sort_values(['date','issuer']).copy();nz=q.issuer.eq('New Zealand').astype(float);au=1-nz
   z=[np.ones(len(q)),nz,q.post.astype(float),q.post*nz,au*q.r2,au*q.slope,nz*q.r2,nz*q.slope]
   if trend:z += [au*q.linear_time,nz*q.linear_time]
   X=np.column_stack(z);y=q[outcome].to_numpy();n,k=X.shape;assert np.linalg.matrix_rank(X)==k
   b=np.linalg.lstsq(X,y,rcond=None)[0];bread=np.linalg.inv(X.T@X);scores=X*(y-X@b)[:,None]
   S=pd.DataFrame(scores).assign(date=q.date.to_numpy()).groupby('date',sort=True).sum().to_numpy();T=len(S)
   # Verify contrast equals difference of separate issuer regressions.
   bs=[]
   for issuer in ['Australia','New Zealand']:
    a=q[q.issuer.eq(issuer)];cols=['post','r2','slope']+(['linear_time'] if trend else [])
    xx=np.column_stack([np.ones(len(a)),a[cols].astype(float)]);bs.append(np.linalg.lstsq(xx,a[outcome],rcond=None)[0][1])
   err=abs((bs[1]-bs[0])-b[3]);assert err<1e-10
   checks.append(dict(sample=sample,outcome=outcome,trend=trend,contrast_error=float(err)))
   for L in [3,7,10]:
    M=S.T@S
    for j in range(1,L+1):
     C=S[j:].T@S[:-j];M+=(1-j/(L+1))*(C+C.T)
    V=bread@M@bread*n/(n-k);se=np.sqrt(V[3,3]);df=T-1;crit=t.ppf(.975,df)
    out.append(dict(sample=sample,outcome=outcome,trend=trend,lags=L,n=n,dates=T,Australia=b[2],New_Zealand=b[2]+b[3],NZ_minus_AU=b[3],se=se,lo=b[3]-crit*se,hi=b[3]+crit*se,p=2*t.sf(abs(b[3]/se),df)))
pd.DataFrame(out).to_csv(O/'cross_case_contrasts.csv',index=False)
(O/'cross_case_validation.json').write_text(json.dumps(checks,indent=2))
print(pd.DataFrame(out).query('lags == 7').to_string(index=False))

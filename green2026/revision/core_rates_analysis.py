"""Local-currency controls and external-yield price diagnostics for AU/NZ.
Source snapshots in inputs/core_rates. No network needed to reproduce.
"""
from pathlib import Path
import json,hashlib,warnings
import pandas as pd
import numpy as np
from scipy.stats import t
warnings.filterwarnings('ignore',message='Workbook contains no default style')
B=Path('green2026/revision');O=B/'outputs';I=B/'inputs/core_rates'
d=pd.read_parquet(O/'revised_panel_PRIVATE.parquet')
p=pd.read_csv(O/'contract_eligibility_audit5_PRIVATE.csv')
au=pd.read_excel(I/'au.xlsx',header=None)
assert au.iloc[10,1]=='FCMYGBAG2D' and au.iloc[10,4]=='FCMYGBAG10D'
ar=au.iloc[11:,[0,1,4]].copy();ar.columns=['date','r2','r10'];ar['date']=pd.to_datetime(ar.date)
nz=pd.read_excel(I/'nz.xlsx',header=None)
assert nz.iloc[4,9]=='INM.DG102.NZZCF' and nz.iloc[4,11]=='INM.DG110.NZZCF'
nr=nz.iloc[5:,[0,9,11]].copy();nr.columns=['date','r2','r10'];nr['date']=pd.to_datetime(nr.date)
specific=pd.read_csv(I/'au_specific.csv',skiprows=1,header=None,low_memory=False)
assert '4.25% 21-Jun-2034' in specific.iloc[1,38] and '3.75% 21-May-2034' in specific.iloc[1,37]
ay=specific.iloc[8:,[0,38,37]].copy();ay.columns=['date','yield_g','yield_c'];ay.date=pd.to_datetime(ay.date,format='%d-%b-%Y')
assert nz.iloc[1,26]=='May 2034' and nz.iloc[1,27]=='May 2035'
ny=nz.iloc[5:,[0,26,27]].copy();ny.columns=['date','yield_g','yield_c'];ny.date=pd.to_datetime(ny.date)
models=[];diagnostics=[];coverage=[];panels=[];validation=[]
def ols_hac(q,terms,label):
 X=np.column_stack([np.ones(len(q)),q[terms].astype(float)]);y=q.reconstructed_gap.to_numpy();n,k=X.shape
 assert np.linalg.matrix_rank(X)==k
 b=np.linalg.lstsq(X,y,rcond=None)[0];bread=np.linalg.inv(X.T@X);S=X*(y-X@b)[:,None]
 for L in [3,7,10]:
  meat=S.T@S
  for j in range(1,L+1):
   cross=S[j:].T@S[:-j];meat+=(1-j/(L+1))*(cross+cross.T)
  V=bread@meat@bread*n/(n-k);se=np.sqrt(V[1,1]);crit=t.ppf(.975,n-k)
  models.append(dict(issuer=issuer,model=label,HAC_lags=L,n=n,coef=b[1],se=se,lo=b[1]-crit*se,hi=b[1]+crit*se,p=2*t.sf(abs(b[1]/se),n-k),sample_hash=hashlib.sha256('|'.join(q.date.astype(str)).encode()).hexdigest()))
 # Independent FWL partialling-out check of the post coefficient.
 Z=np.column_stack([np.ones(n),q[terms[1:]].astype(float)])
 xp=X[:,1]-Z@np.linalg.lstsq(Z,X[:,1],rcond=None)[0]
 yp=y-Z@np.linalg.lstsq(Z,y,rcond=None)[0]
 check=xp@yp/(xp@xp);assert abs(check-b[1])<1e-10
 validation.append(dict(issuer=issuer,model=label,FWL_error=float(abs(check-b[1]))))
for issuer,rates,yields in [('Australia',ar,ay),('New Zealand',nr,ny)]:
 row=p[p.IssuerName.eq(issuer)].iloc[0];assert row.contract_status=='eligible'
 q=d[d.pair.eq(row.pair)].copy().sort_values('date');original=len(q)
 rates[['r2','r10']]=rates[['r2','r10']].apply(pd.to_numeric,errors='coerce')
 q=q.merge(rates,on='date',how='left',validate='one_to_one').dropna(subset=['r2','r10'])
 q['slope']=q.r10-q.r2;q['issuer']=issuer
 assert q.post.nunique()==2 and not q.date.duplicated().any()
 coverage.append(dict(issuer=issuer,original_quotes=original,common_rate_rows=len(q),dropped_dates=original-len(q),pre=int((~q.post).sum()),post=int(q.post.sum())))
 ols_hac(q,['post'],'same_rows_baseline');ols_hac(q,['post','r2','slope'],'local_curve_controlled')
 q['linear_time']=(q.date-q.date.min()).dt.days.astype(float)
 ols_hac(q,['post','r2','slope','linear_time'],'local_curve_and_trend')
 pre=q[~q.post].copy();pre['post']=pre.date.ge('2026-02-07')
 ols_hac(pre,['post','r2','slope'],'pre_event_placebo_Feb07')
 panels.append(q)
 yields[['yield_g','yield_c']]=yields[['yield_g','yield_c']].apply(pd.to_numeric,errors='coerce')
 v=q.merge(yields,on='date',how='left',validate='one_to_one')
 for side in ['g','c']:
  mat=pd.Timestamp(row['maturity_'+side]);freq=int(row['frequency_'+side]);coupon=float(row['coupon_'+side])/freq
  dates=[];dt=mat
  while dt>=pd.Timestamp('2024-01-01'):dates.append(dt);dt-=pd.DateOffset(months=12//freq)
  dates=sorted(dates)
  assert not any(v.date.min()<=x<=v.date.max() for x in dates)
  for lag in [0,2]:
   for _,obs in v.dropna(subset=['yield_'+side]).iterrows():
    settle=obs.date+pd.offsets.BDay(lag);prev=max(x for x in dates if x<=settle);future=[x for x in dates if x>settle]
    frac=(future[0]-settle).days/(future[0]-prev).days;periods=frac+np.arange(len(future));cash=np.full(len(future),coupon);cash[-1]+=100
    dirty=np.sum(cash/(1+obs['yield_'+side]/100/freq)**periods);ai=coupon*(settle-prev).days/(future[0]-prev).days;clean=dirty-ai
    for basis,pred in [('clean',clean),('dirty',dirty)]:
     diagnostics.append(dict(issuer=issuer,ISIN=row['ISIN_'+side],side=side,date=str(obs.date.date()),settlement_weekdays=lag,basis=basis,observed_price=obs['arithmetic_mid_'+side],predicted_price=pred,residual=obs['arithmetic_mid_'+side]-pred))
v=pd.DataFrame(diagnostics)
for q in panels:
 issuer=q.issuer.iloc[0]
 z=v[v.issuer.eq(issuer)&v.basis.eq('clean')&v.settlement_weekdays.eq(2)].pivot(index='date',columns='side',values='predicted_price')
 z.index=pd.to_datetime(z.index);z['external_gap']=100*np.log(z.g/z.c)
 q=q.merge(z[['external_gap']],left_on='date',right_index=True,validate='one_to_one').copy()
 q['reconstructed_gap']=q.external_gap
 ols_hac(q,['post','r2','slope'],'external_yield_implied_clean_local_curves')
pd.DataFrame(models).to_csv(O/'core_local_curve_models.csv',index=False)
pd.DataFrame(coverage).to_csv(O/'core_local_curve_coverage.csv',index=False)
pd.concat(panels).to_parquet(O/'core_local_curve_panel_PRIVATE.parquet',index=False)
v.to_csv(O/'core_external_price_diagnostics_PRIVATE.csv',index=False)
summary=v.groupby(['issuer','side','settlement_weekdays','basis']).residual.agg(n='size',mean='mean',rmse=lambda x:np.sqrt(np.mean(x*x)))
summary.to_csv(O/'core_external_price_fit.csv')
(O/'core_rates_validation.json').write_text(json.dumps(validation,indent=2))
meta={'sources':{'AU_rates':'https://www.rba.gov.au/statistics/tables/xls/f02d.xlsx','AU_specific':'https://www.rba.gov.au/statistics/tables/csv/f16-data.csv','NZ':'https://rbnz.govt.nz/-/media/project/sites/rbnz/files/statistics/series/b/b2/hb2-daily-close.xlsx'},'sample_rule':'No forward filling; same-date complete r2/r10 and original price rows; same rows for baseline and controls','inference':'Separate issuer time-series OLS; Newey-West Bartlett HAC lags 3,7,10. Conditional on these cases, not cross-country clustered inference. Rates may respond to the same event; coefficients are conditional associations.','pricing':'Nominal fixed bullet, regular semiannual periods, Actual/Actual coupon period. Settlement 0 or 2 weekdays without local holiday adjustment. No coupon/ex-coupon in sample window. Indicative external closing yields may differ in time and methodology from LSEG quotes; diagnostic fit does not certify historical quote convention.','Denmark':'No Danish government yield curve downloaded. DNRENTD metadata currently exposes policy/overnight rates only; no EUR/USD substitution made.','retrieved_date':'2026-11-17'}
(O/'core_rates_metadata.json').write_text(json.dumps(meta,indent=2))
print(pd.DataFrame(coverage).to_string(index=False));print(pd.DataFrame(models).query('HAC_lags == 7').to_string(index=False));print(summary.to_string())

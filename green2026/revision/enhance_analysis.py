"""Additional diagnostics. Cash-flow scenarios are conditional, not verified returns.
Run after analyze_revision.py, from package root. No network required.
"""
from pathlib import Path
import json, numpy as np,pandas as pd
from scipy.stats import t
from scipy.optimize import brentq
B=Path('green2026/revision');O=B/'outputs'
d=pd.read_parquet(O/'revised_panel_PRIVATE.parquet');c=pd.read_parquet(O/'curve_panel_PRIVATE.parquet');p=pd.read_csv(O/'contract_eligibility_PRIVATE.csv')
def within(q,terms=['post']):
 a=q[['reconstructed_gap']+terms].astype(float);a-=a.groupby(q.pair).transform('mean');X=a[terms].to_numpy();y=a.reconstructed_gap.to_numpy();b=np.linalg.lstsq(X,y,rcond=None)[0];return X,y,b

def fit(q,terms,label):
 X,y,b=within(q,terms);bread=np.linalg.inv(X.T@X);scores=X*(y-X@b)[:,None];n=len(q);P=q.pair.nunique();T=q.date.nunique();k=len(terms)
 def meat(key):
  S=pd.DataFrame(scores).assign(key=key.to_numpy()).groupby('key').sum().to_numpy();return S.T@S*len(S)/(len(S)-1)
 V=bread@(meat(q.pair)+meat(q.date)-meat(q.pair+'|'+q.date.astype(str)))@bread*(n-1)/(n-P-k)
 assert np.diag(V).min()>0
 se=np.sqrt(np.diag(V));crit=t.ppf(.975,min(P,T)-1)
 return [dict(test=label,term=x,coef=b[i],se=se[i],lo=b[i]-crit*se[i],hi=b[i]+crit*se[i],p=2*t.sf(abs(b[i]/se[i]),min(P,T)-1),pairs=P,n=n,days=T) for i,x in enumerate(terms)]
loo=[]
for group in ['pair','RCSCurrencyLeaf']:
 for item in sorted(d[group].unique()):
  z=d[d[group].ne(item)];_,_,b=within(z);loo.append(dict(exclusion=group,omitted=item,coef=b[0],pairs=z.pair.nunique(),n=len(z)))
pd.DataFrame(loo).to_csv(O/'leave_one_out.csv',index=False)
# Time-HAC on date-aggregated within scores keeps the FE point estimate.
# It answers a time-series uncertainty question; it is not selected over clustering.
hac=[]
for label,q in [('broad74',d),('balanced40',d[d.pair.isin(d.groupby('pair').size().loc[lambda x:x.eq(58)].index)])]:
 X,y,b=within(q);x=X[:,0];score=pd.Series(x*(y-X@b)).groupby(q.date.reset_index(drop=True)).sum().sort_index().to_numpy();T=len(score)
 for L in [3,7,10]:
  M=score@score+sum(2*(1-k/(L+1))*(score[k:]@score[:-k]) for k in range(1,L+1));se=np.sqrt(M*T/(T-1))/(x@x);crit=t.ppf(.975,T-1)
  hac.append(dict(sample=label,lags=L,coef=b[0],se=se,lo=b[0]-crit*se,hi=b[0]+crit*se,p=2*t.sf(abs(b[0]/se),T-1),days=T))
pd.DataFrame(hac).to_csv(O/'calendar_score_hac.csv',index=False)
# Observable cash-flow timing proxy, not estimated duration: signed maturity gap.
c['signed_maturity_gap']=(pd.to_datetime(c.maturity_g)-pd.to_datetime(c.maturity_c)).dt.days/365.25
terms=['post','eu_r2','eu_slope','us_r2','us_slope']
for curr in ['eu','us']:
 for var in ['r2','slope']:
  name=curr+'_'+var+'_maturity';c[name]=c[curr+'_'+var]*c.signed_maturity_gap;terms.append(name)
models=fit(c,terms,'maturity_interacted_curves')
pd.DataFrame(models).to_csv(O/'additional_models.csv',index=False)
# Three documented nominal sovereigns. Same-day settlement is an explicit scenario.
# Actual/actual coupon-period accrual; regular scheduled coupons; principal=100.
# No coupon or ex-coupon dates occur in this window for these six bonds.
scenarios=[];risk=[];maximum_error=0.
for _,r in p[p.contract_status.eq('eligible')].iterrows():
 q=d[d.pair.eq(r.pair)].copy(); schedules={}; ais={}; conditional_risks={}
 for side in ['g','c']:
  maturity=pd.Timestamp(r['maturity_'+side]);freq=int(r['frequency_'+side]);coupon=float(r['coupon_'+side])/freq
  dates=[];dt=maturity
  while dt>=pd.Timestamp('2024-01-01'):
   dates.append(dt);dt=dt-pd.DateOffset(months=12//freq)
  dates=sorted(dates);schedules[side]=dates
  # Cross-check reconstructed coupon calendar against explicitly recorded dates.
  assert set(x.strftime('%m-%d') for x in dates)==set(r['payment_dates_'+side].split(';'))
  assert not any(q.date.min()<=x<=q.date.max() for x in dates)
  def accrued(now):
   prev=max(x for x in dates if x<=now);nxt=min(x for x in dates if x>now);return coupon*(now-prev).days/(nxt-prev).days
  ai=q.date.apply(accrued).to_numpy();ais[side]=ai
  quoted=q['arithmetic_mid_'+side].to_numpy()
  q['dirty_if_clean_'+side]=quoted+ai
  q['clean_if_dirty_'+side]=quoted-ai
  base=q[q.date.eq('2026-02-27')].iloc[0];now=pd.Timestamp('2026-02-27');a0=accrued(now)
  nxt=min(x for x in dates if x>now);prev=max(x for x in dates if x<=now);fraction=(nxt-now).days/(nxt-prev).days
  future=[x for x in dates if x>now];periods=fraction+np.arange(len(future));cf=np.full(len(future),coupon);cf[-1]+=100
  def price(yield_):return np.sum(cf/(1+yield_/freq)**periods)
  for interpretation,target in [('quoted_clean',float(base['arithmetic_mid_'+side])+a0),('quoted_dirty',float(base['arithmetic_mid_'+side]))]:
   yy=brentq(lambda y:price(y)-target,-.05,.30)
   v=1+yy/freq;pv=cf/v**periods
   duration=float(np.sum(periods/freq*pv/v)/target)
   convexity=float(np.sum(periods*(periods+1)/freq**2*pv/v**2)/target)
   h=1e-5;numdur=-(price(yy+h)-price(yy-h))/(2*h*target);maximum_error=max(maximum_error,abs(duration-numdur));assert abs(duration-numdur)<1e-6
   risk.append(dict(issuer=r.IssuerName,ISIN=r['ISIN_'+side],side=side,quote_assumption=interpretation,base_date=str(now.date()),conditional_yield=yy,modified_duration=duration,convexity=convexity,exact_plus_50bp_pct=100*(price(yy+.005)/target-1)))
 for label,g,cc in [('quoted',q.arithmetic_mid_g,q.arithmetic_mid_c),('dirty_value_if_quotes_clean',q.dirty_if_clean_g,q.dirty_if_clean_c),('clean_value_if_quotes_dirty',q.clean_if_dirty_g,q.clean_if_dirty_c)]:
  gap=100*np.log(np.asarray(g)/np.asarray(cc));q['scenario']=gap;change=q.loc[q.post,'scenario'].mean()-q.loc[~q.post,'scenario'].mean()
  scenarios.append(dict(issuer=r.IssuerName,pair=r.pair,scenario=label,change=change,n=len(q)))
pd.DataFrame(scenarios).to_csv(O/'cashflow_scenarios.csv',index=False);pd.DataFrame(risk).to_csv(O/'conditional_duration.csv',index=False)
candidates=pd.read_csv(B/'inputs/candidate_inventory_PRIVATE.csv');candidates['coupon_gap_abs']=(candidates.coupon_g-candidates.CouponRate).abs()
elig=candidates[candidates.window_eligible].copy();elig['coupon_known']=elig.coupon_gap_abs.notna()
# A diagnostic queue, not approved replacement matches: missing terms remain unknown.
rank=elig.sort_values(['ISIN_g','coupon_known','coupon_gap_abs','gap_years','ISIN'],ascending=[True,False,True,True,True]).groupby('ISIN_g').head(3)
rank.to_csv(O/'candidate_review_queue_PRIVATE.csv',index=False)
summary={'leave_one_issuer_range':[min(x['coef'] for x in loo if x['exclusion']=='pair'),max(x['coef'] for x in loo if x['exclusion']=='pair')],'leave_one_currency_range':[min(x['coef'] for x in loo if x['exclusion']=='RCSCurrencyLeaf'),max(x['coef'] for x in loo if x['exclusion']=='RCSCurrencyLeaf')],'candidate_records':len(candidates),'window_eligible_candidates':len(elig),'eligible_with_saved_history':int(elig.quoted_in_pool.sum()),'cashflow_scenario_limit':'Three documented sovereign cases only; same-day settlement; no coupon payments inside window; conditional price-convention assumptions, not verified total returns. Duration is model-implied under each assumption, not LSEG reported duration.','duration_numeric_derivative_max_error':maximum_error,'maturity_model_row_count':len(c),'maturity_model_same_rows_as_curve_panel':True,'maturity_proxy_is_not_duration':True}
(O/'enhancement_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));print(pd.DataFrame(models).to_string(index=False));print(pd.DataFrame(scenarios).to_string(index=False));print(pd.DataFrame(risk).to_string(index=False))

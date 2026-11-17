"""Targeted coupon-calendar audit; does not certify historical quote conventions.
Run from replication package root after analyze_revision.py.
"""
from pathlib import Path
import ast,json
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.optimize import brentq
B=Path('green2026/revision'); O=B/'outputs'
d=pd.read_parquet(O/'revised_panel_PRIVATE.parquet')
p=pd.read_csv(O/'contract_eligibility_PRIVATE.csv')
# Reuse only the two estimation functions, avoiding script-level side effects.
tree=ast.parse((B/'enhance_analysis.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['within','fit']],type_ignores=[]),'<estimation>','exec'))
calendar={'EU':{'g':(.4,'2037-02-04',1),'c':(.875,'2037-03-11',1)},
          'Poland':{'g':(1.,'2029-03-07',1),'c':(1.,'2028-10-25',1)}}
sources={'EU':['https://www.borsaitaliana.it/borsa/obbligazioni/mot/euro-obbligazioni/scheda/EU000A3K4C42-MOTX.html?lang=en','https://www.borsaitaliana.it/borsa/obbligazioni/mot/euro-obbligazioni/scheda/EU000A3K4DC0-MOTX.html?lang=en'],
'Poland':['https://www.gov.pl/attachment/e4131d22-ed26-46ef-8cc6-0c17a4f72968','https://finanse-arch.mf.gov.pl/documents/766655/fd2d846e-b82b-4e91-bd14-45b7a269d33d']}
models=fit(d,['post'],'baseline74'); scenario=[]; events=[]; yieldchecks=[]
flag=pd.Series(False,index=d.index); selected=[]
for issuer,terms in calendar.items():
 r=p[p.IssuerName.eq(issuer)].iloc[0]; pair=r.pair;selected.append(pair)
 q=d[d.pair.eq(pair)].copy(); assert len(q)>0
 for side,(annual,mat,freq) in terms.items():
  assert abs(float(r['coupon_'+side])-annual)<1e-10
  assert pd.Timestamp(r['maturity_'+side])==pd.Timestamp(mat)
  dates=list(pd.date_range(end=mat,periods=16,freq=pd.DateOffset(years=1)))
  coupon=annual/freq
  def ai(now):
   prev=max(x for x in dates if x<=now); nxt=min(x for x in dates if x>now)
   return coupon*(now-prev).days/(nxt-prev).days
  accrual=q.date.map(ai)
  paid=q.date.map(lambda now:coupon*sum(q.date.min()<x<=now for x in dates))
  quote=q['arithmetic_mid_'+side]
  q['wealth_if_clean_'+side]=quote+accrual+paid
  q['wealth_if_dirty_'+side]=quote+paid
  for when in dates:
   if q.date.min()<=when<=q.date.max():
    events.append(dict(issuer=issuer,ISIN=r['ISIN_'+side],side=side,scheduled_coupon_date=str(when.date()),coupon=coupon))
    flag|=d.pair.eq(pair)&(d.date-when).dt.days.abs().le(5)
  # Diagnostic model-yield fit, not proof of vendor methodology/settlement.
  for _,row in q.iterrows():
   observed=row['YLDTOMAT_'+side]
   if pd.isna(observed):continue
   now=row.date; future=[x for x in dates if x>now];prev=max(x for x in dates if x<=now)
   frac=(future[0]-now).days/(future[0]-prev).days
   periods=frac+np.arange(len(future)); cf=np.full(len(future),coupon);cf[-1]+=100
   def price(y):return np.sum(cf/(1+y)**periods)
   for assumption,target in [('clean',row['arithmetic_mid_'+side]+ai(now)),('dirty',row['arithmetic_mid_'+side])]:
    implied=brentq(lambda y:price(y)-target,-.05,.3)*100
    yieldchecks.append(dict(issuer=issuer,side=side,date=str(now.date()),assumption=assumption,residual_bp=100*(implied-observed)))
 for name,g,c in [('quoted',q.arithmetic_mid_g,q.arithmetic_mid_c),('cash_wealth_if_clean',q.wealth_if_clean_g,q.wealth_if_clean_c),('cash_wealth_if_dirty',q.wealth_if_dirty_g,q.wealth_if_dirty_c)]:
  gap=100*np.log(g/c)
  scenario.append(dict(issuer=issuer,scenario=name,change=gap[q.post].mean()-gap[~q.post].mean(),n=len(q)))
 models+=fit(d[~d.pair.eq(pair)],['post'],'omit_'+issuer)
models+=fit(d[~d.pair.isin(selected)],['post'],'omit_EU_and_Poland')
models+=fit(d[~flag],['post'],'omit_targeted_coupon_windows_5_calendar_days')
pd.DataFrame(models).to_csv(O/'coupon_audit_models.csv',index=False)
pd.DataFrame(scenario).to_csv(O/'coupon_audit_scenarios.csv',index=False)
pd.DataFrame(events).to_csv(O/'coupon_audit_calendar.csv',index=False)
y=pd.DataFrame(yieldchecks)
y.groupby(['issuer','side','assumption']).residual_bp.agg(['count','mean',lambda x:np.sqrt(np.mean(x*x))]).rename(columns={'<lambda_0>':'rmse_bp'}).to_csv(O/'coupon_audit_yield_fit.csv')
meta={'sources':sources,'verified_scope':'Annual coupon rates, maturity dates and regular calendar; EU ACT/ACT exchange convention. Poland ACT/ACT is a scenario assumption.', 'limitations':'Same-day settlement; scheduled unadjusted coupon dates; no ex-coupon adjustment; coupons held as cash without reinvestment. Wealth scenarios are conditional, not certified total returns. No change to contract eligibility or historical LSEG quote-convention verification. Targeted two-issuer audit is not a full-sample coupon audit.', 'removed_rows':int(flag.sum())}
(O/'coupon_audit_metadata.json').write_text(json.dumps(meta,indent=2))
print(pd.DataFrame(models).to_string(index=False));print(pd.DataFrame(scenario).to_string(index=False));print((O/'coupon_audit_yield_fit.csv').read_text())

"""Instrument-specific contract update, independent research checkpoint.
Run after analyze_revision.py. Existing Word documents remain the prior version.
"""
from pathlib import Path
import ast,json
import numpy as np
import pandas as pd
from scipy.stats import t
B=Path('green2026/revision');O=B/'outputs'
p=pd.read_csv(O/'contract_eligibility_PRIVATE.csv')
d=pd.read_parquet(O/'revised_panel_PRIVATE.parquet')
c=pd.read_parquet(O/'curve_panel_PRIVATE.parquet')
for filename,names in [('analyze_revision.py',['classify']),('enhance_analysis.py',['within','fit'])]:
 tree=ast.parse((B/filename).read_text())
 exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),filename,'exec'))
poland='https://dl.bourse.lu/dl?v=Ad1eA7yf95LO7OtKkVeNjWsDzzWm57akCLcv2kSIWznPQmi3TJcyoTWIhvwsR9%2FrnuZfWMhiRzTKYLDcS5h+0pvGjYiCLHqdOEjPG1%2FK4%2FJfYgu74+X9ZJrg4KtLUKb7O3mpINp2T9ILSeQ7GSBPcHLwJQLw6hqAAeqeILWOxs9Wa+vO4Wm4GO%2F9sagzlNjo'
base='https://www.jpx.co.jp/english/equities/products/tpbm/announcement/detail/b5b4pj0000028lze-att/b5b4pj0000028m13.pdf'
cpp='https://ise-prodnr-eu-west-1-data-integration.s3-eu-west-1.amazonaws.com/202305/04c863c1-422c-4c5c-8125-54fd772585ee.pdf'
updates=[]
def update(isin,side,values,source):
 mask=p['ISIN_'+side].eq(isin);assert mask.sum()==1
 for field,value in values.items():
  col=field+'_'+side
  old=p.loc[mask,col].iloc[0] if col in p else None
  p.loc[mask,col]=value
  updates.append(dict(ISIN=isin,field=field,old_value=old,new_value=value,source=source))
update('XS1958534528','g',{'rate_type':'Fixed','seniority':'senior','security':'unsecured','call':'none','reset':'none','extension':'none','frequency':1,'payment_dates':'03-07','indexation':'nominal','day_count':'Actual/Actual ICMA','payment_adjustment':'Following; accrual unadjusted'},poland+' | '+base)
update('XS1508566392','c',{'frequency':1,'payment_dates':'10-25'},'https://finanse-arch.mf.gov.pl/documents/766655/fd2d846e-b82b-4e91-bd14-45b7a269d33d')
update('CA12593CAV33','c',{'rate_type':'Fixed','call':'tax_event_only','reset':'none','extension':'none','frequency':2,'payment_dates':'03-08;09-08','indexation':'nominal','day_count':'Actual/Actual Canadian Compound: regular semiannual 30/360; other calculations Actual/365','payment_adjustment':'Following Toronto business day; accrual unadjusted'},cpp)
p.loc[p.ISIN_g.eq('XS1958534528'),'source_primary']=poland+' | '+base
p.loc[p.ISIN_c.eq('CA12593CAV33'),'source_primary']=cpp
p[['contract_status','reason','missing_fields']]=p.apply(lambda r:pd.Series(classify(r)),axis=1)
p.to_csv(O/'contract_eligibility_audit5_PRIVATE.csv',index=False)
pd.DataFrame(updates).to_csv(O/'contract_field_updates_audit5.csv',index=False)
status=p.set_index('pair').contract_status
for q in [d,c]:q['contract_status']=q.pair.map(status)
results=fit(d,['post'],'broad74_unchanged')
results+=fit(d[d.contract_status.ne('ineligible')],['post'],'audit5_no_known_contract_failure')
cc=c[c.contract_status.ne('ineligible')].copy()
results+=fit(cc,['post'],'audit5_screened_common_baseline')
results+=fit(cc,['post','eu_r2','eu_slope','us_r2','us_slope'],'audit5_screened_common_curves')
results+=fit(d[~d.pair.str.contains('CA12593CAF82',regex=False)],['post'],'broad_omit_CPPIB')
pd.DataFrame(results).to_csv(O/'contract_models_audit5.csv',index=False)
row=pd.DataFrame(results).query("test == 'audit5_no_known_contract_failure'").iloc[0]
q=d[d.contract_status.ne('ineligible')]
X=np.column_stack([q.post.astype(float),pd.get_dummies(q.pair,dtype=float)])
independent=np.linalg.lstsq(X,q.reconstructed_gap,rcond=None)[0][0]
assert abs(independent-row.coef)<1e-10
meta={'counts':p.contract_status.value_counts().to_dict(),'verified_instrument':'XS1958534528 terms completed; comparison bond remains incomplete','tax_call_treatment':'Existing conservative classify() rejects any documented call, including tax-event-only. No claim of economic equivalence between tax and ordinary calls. Broad sample retains CPPIB.','independent_coefficient_error':float(abs(independent-row.coef)),'word_documents':'Prior checkpoint; audit5 results not yet incorporated','historical_price_conventions':'No new historical LSEG convention certified','sources':[poland,base,cpp]}
(O/'contract_audit5_summary.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2));print(pd.DataFrame(results).query("term == 'post'").to_string(index=False))

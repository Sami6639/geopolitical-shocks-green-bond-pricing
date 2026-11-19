"""Revision audit: deterministic contractual rules and identical-sample rate models.
Run from the replication root. No outcomes enter contract classification.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import t

B=Path('green2026/revision'); O=B/'outputs'; O.mkdir(parents=True,exist_ok=True)
R=Path('green2026/review/outputs')
p=pd.read_csv('green2026/final/outputs/contract_register_PRIVATE.csv')
ref=pd.read_csv(R/'conventional_reference_PRIVATE.csv').drop_duplicates('ISIN')
p=p.merge(ref[['ISIN','IssuerName']],left_on='ISIN_c',right_on='ISIN',validate='one_to_one')
d=pd.read_parquet(R/'paired_panel_PRIVATE.parquet')
assert len(p)==76 and d.pair.nunique()==76

# Contract fields distinguish fixed current coupon from options and seniority.
for side in ['g','c']:
    p['rate_type_'+side]=p['couponKind_'+side].fillna('unknown').replace({'Extendible':'fixed_until_extension'})
    for key in ['seniority','security','call','reset','extension','frequency','payment_dates','indexation']:
        p[key+'_'+side]='unknown'
p['source_primary']='';p['structural_mismatch']=False
idx=p.ISIN_g.eq('XS2831094706')
p.loc[idx,['coupon_g','rate_type_g','seniority_g','security_g','call_g','reset_g','extension_g','frequency_g','payment_dates_g']]=[4.266,'fixed_reset','tier2_subordinated','unsecured','2029-06-04','2029-06-04','none',1,'06-04']
p.loc[idx,['rate_type_c','seniority_c','security_c','call_c','reset_c','extension_c','frequency_c','payment_dates_c']]=['fixed_until_extension','senior','covered','none','on_extension','2034-10-27',1,'10-27']
p.loc[idx,'structural_mismatch']=True
p.loc[idx,'source_primary']='https://www.commbank.com.au/content/dam/commbank-assets/about-us/capital-instruments/docs/tier-2-capital-instruments.pdf | https://www.commbank.com.au/content/dam/commbank/about-us/group-funding/articles/CBA-EUR-118-5-CB-due-2033-FTs-EV.pdf'
for isin in ['FR001400Q6Q8','FR0014010BU9','FR0014013WA1','FR001400ICR2']:
    for side in ['g','c']:
        z=p['ISIN_'+side].eq(isin);p.loc[z,'extension_'+side]='yes';p.loc[z,'rate_type_'+side]='fixed_until_extension'
# Public final terms reveal call features even when portfolio records say Fixed.
z=p.ISIN_g.eq('US695114DA39')
for side,date,spread,pay in [('g','2053-11-15',25,'05-15;11-15'),('c','2054-07-15',30,'01-15;07-15')]:
    p.loc[z,['seniority_'+side,'security_'+side,'call_'+side,'reset_'+side,'extension_'+side,'frequency_'+side,'payment_dates_'+side]]=['first_mortgage','secured',f'make_whole_T+{spread};par_from_{date}','none','none',2,pay]
p.loc[z,'source_primary']='https://www.sec.gov/Archives/edgar/data/75594/000110465923060913/tm2315350d3_fwp.htm | https://www.sec.gov/Archives/edgar/data/75594/000110465924001017/tm2329358d3_fwp.htm'
z=p.ISIN_g.eq('XS2381853279')
for side,date,pay,url in [('g','2028-06-01','09-01','602141'),('c','2027-10-20','01-20','602136')]:
    p.loc[z,['call_'+side,'reset_'+side,'extension_'+side,'frequency_'+side,'payment_dates_'+side]]=[f'make_whole_15bp;par_from_{date}','none','none',1,pay]
p.loc[z,'source_primary']='https://www.nationalgrid.com/document/602141/download | https://www.nationalgrid.com/document/602136/download'

sovereigns={
'DK0009924615':(1,'11-15','11-15','https://www.nationalbanken.dk/en/government-debt/funding-strategy/green-bonds'),
'AU3TB0000200':(2,'06-21;12-21','05-21;11-21','https://www.aofm.gov.au/media/1118 | https://www.aofm.gov.au/media/962 | https://www.aofm.gov.au/media/228'),
'NZGOVDT534C4':(2,'05-15;11-15','05-15;11-15','https://debtmanagement.treasury.govt.nz/sites/default/files/media/media_attachment/May-34%20nominal%20green%20bond%20-%20term%20sheet.pdf | https://debtmanagement.treasury.govt.nz/sites/default/files/2024-04/Final%20Term%20Sheet%20-%20NZGB%202035.pdf | https://debtmanagement.treasury.govt.nz/sites/default/files/media/media_attachment/nz-govt-bonds-information-memorandum-19oct22.pdf')}
for isin,(freq,pg,pc,source) in sovereigns.items():
    z=p.ISIN_g.eq(isin)
    for side,pay in [('g',pg),('c',pc)]:
        p.loc[z,['rate_type_'+side,'seniority_'+side,'security_'+side,'call_'+side,'reset_'+side,'extension_'+side,'frequency_'+side,'payment_dates_'+side]]=['Fixed','sovereign_direct','general_sovereign_obligation','none','none','none',freq,pay]
    p.loc[z,'source_primary']=source
# Instrument-specific official records added in the second audit.
for isin in sovereigns:
    p.loc[p.ISIN_g.eq(isin),['indexation_g','indexation_c']]='nominal'
z=p.ISIN_g.eq('XS2226974504')
p.loc[z,['coupon_g','rate_type_g','indexation_g','indexation_c','structural_mismatch']]=[0.125,'Fixed','nominal','inflation_linked',True]
p.loc[z,'source_primary']='https://www.riksgalden.se/contentassets/847d727863d046fe9cdacb1d81e34b61/specific-terms-sgb-il-3114.pdf | https://www.riksgalden.se/globalassets/dokument_eng/debt/green-bonds/200901-kingdom-of-sweden-10-yr-green-bond-press-release.pdf'
z=p.ISIN_g.eq('IE00BFZRQ242')
p.loc[z,['coupon_g','rate_type_g','frequency_g','payment_dates_g','frequency_c','payment_dates_c']]=[1.35,'Fixed',1,'03-18',1,'10-18']
p.loc[z,'source_primary']='https://www.ntma.ie/business-areas/funding-and-debt-management/government-securities/government-bonds'
# Third audit: instrument-specific issuer terms and exchange metadata.
z=p.ISIN_g.eq('US458140CA64')
p.loc[z,['coupon_g','rate_type_g','frequency_g','payment_dates_g','seniority_g','call_g']]=[4.15,'Fixed',2,'02-05;08-05','senior','make_whole_T+25;par_from_2032-05-05']
p.loc[z,'source_primary']='https://www.sec.gov/Archives/edgar/data/50863/000119312522213699/d390659dex44.htm'
z=p.ISIN_g.eq('EU000A3K4C42')
p.loc[z,['frequency_g','frequency_c','payment_dates_g','payment_dates_c']]=[1,1,'02-04','03-11']
p.loc[z,'source_primary']='https://www.borsaitaliana.it/borsa/obbligazioni/mot/euro-obbligazioni/scheda/EU000A3K4C42-MOTX.html?lang=en | https://www.borsaitaliana.it/borsa/obbligazioni/mot/euro-obbligazioni/scheda/EU000A3K4DC0-MOTX.html?lang=en'
# Exchange clean-price metadata does not identify the historical LSEG quote source.
p['coupon_gap']=p.coupon_g-p.coupon_c

def classify(r):
    reasons=[]
    if r.ISIN_g=='XS2831094706': reasons.append('seniority_security_mismatch')
    if r.indexation_g!='unknown' and r.indexation_c!='unknown' and r.indexation_g!=r.indexation_c:reasons.append('nominal_indexed_mismatch')
    if any(r['rate_type_'+s] in ['Variable','fixed_reset','fixed_until_extension'] for s in ['g','c']):reasons.append('variable_reset_or_extension')
    if any(str(r['call_'+s]) not in ['none','unknown'] for s in ['g','c']):reasons.append('embedded_call')
    if any(pd.notna(r['coupon_'+s]) and r['coupon_'+s]<=0 for s in ['g','c']):reasons.append('nonpositive_coupon')
    for feature in ['reset','extension']:
        if any(str(r[feature+'_'+s]) not in ['none','unknown'] for s in ['g','c']):reasons.append('embedded_'+feature)
    if pd.notna(r.coupon_gap) and abs(r.coupon_gap)>.5:reasons.append('coupon_gap_gt_0.50pp')
    required=['rate_type','seniority','security','call','reset','extension','frequency','payment_dates','indexation']
    missing=[k+'_'+s for k in required for s in ['g','c'] if pd.isna(r[k+'_'+s]) or str(r[k+'_'+s])=='unknown']
    if pd.isna(r.coupon_gap):missing.append('coupon')
    if reasons:return 'ineligible', ';'.join(reasons), ';'.join(missing)
    if missing:return 'unresolved','insufficient_contract_evidence',';'.join(missing)
    if r.seniority_g!=r.seniority_c or r.security_g!=r.security_c or r.frequency_g!=r.frequency_c:return 'ineligible','structure_or_frequency_mismatch',''
    if any(str(r['rate_type_'+s]).lower()!='fixed' for s in ['g','c']):return 'ineligible','not_plain_fixed',''
    if r.indexation_g!='nominal' or r.indexation_c!='nominal':return 'ineligible','not_plain_nominal',''
    return 'eligible','documented_plain_fixed_coupon_gap_le_0.50pp',''
p[['contract_status','reason','missing_fields']]=p.apply(lambda r:pd.Series(classify(r)),axis=1)
assert len(p)==76 and p.contract_status.notna().all()
p.to_csv(O/'contract_eligibility_PRIVATE.csv',index=False)

# Restore exact BID/ASK fields, retaining existing instrument/date eligibility.
rawroot=Path('inspection/data/LSEG_Yesil_Tahviller_Ortaklik_Yapisi_ve_Finansman_2000_2026')
measurement=[];rawparts=[]
for side,name in [('g','Yesil'),('c','Konvansiyonel')]:
    rawpath=rawroot/f'LSEG_{name}_Tahvil_Gunluk_Fiyat_Getiri_Spread_2026.parquet'
    if rawpath.exists():
        a=pd.read_parquet(rawpath).drop_duplicates()
        a['date']=pd.to_datetime(a.observation_date).dt.tz_localize(None).dt.normalize()
        missing=int(a.ISIN.isna().sum());a=a[a.ISIN.notna()];amb=a.duplicated(['ISIN','date'],keep=False);conf=int(amb.sum());a=a[~amb]
    else:
        a=pd.read_parquet(B/'inputs/quotes_PRIVATE.parquet');a=a[a.side.eq(side)].copy()
        audit=json.loads((B/'inputs/measurement_provenance.json').read_text())[side];missing=audit['missing_identifier_rows'];conf=audit['conflicting_valid_identifier_rows']
    a=a.merge(d[['ISIN_'+side,'date']].rename(columns={'ISIN_'+side:'ISIN'}).drop_duplicates(),on=['ISIN','date'],validate='one_to_one')
    fields=['BID','ASK','MID_PRICE','CLEAN_PRC','DIRTY_PRC','MID_YLD_1','YLDTOMAT','MOD_DURTN']
    for f in fields:
        if f not in a:a[f]=np.nan
        a[f]=pd.to_numeric(a[f],errors='coerce')
    a['arithmetic_mid']=(a.BID+a.ASK)/2
    a['valid_quote']=a.BID.gt(0)&a.ASK.ge(a.BID)
    dev=(a.MID_PRICE-a.arithmetic_mid).abs()
    measurement.append(dict(side=side,missing_identifier_rows=missing,conflicting_valid_identifier_rows=conf,matched_rows=len(a),valid_bid_ask=int(a.valid_quote.sum()),mid_deviation_gt_0_0001=int(dev.gt(.0001).sum()),mid_deviation_max=float(dev.max()),clean_nonmissing=int(a.CLEAN_PRC.notna().sum()),dirty_nonmissing=int(a.DIRTY_PRC.notna().sum()),duration_nonmissing=int(a.MOD_DURTN.notna().sum())))
    keep=['ISIN','date']+fields+['arithmetic_mid','valid_quote']
    rawparts.append(a[keep].assign(side=side))
    d=d.merge(a[['ISIN','date','BID','ASK','arithmetic_mid','valid_quote']].rename(columns={k:k+'_'+side for k in ['ISIN','BID','ASK','arithmetic_mid','valid_quote']}),on=['ISIN_'+side,'date'],validate='one_to_one')
pd.concat(rawparts).to_parquet(O/'quote_audit_PRIVATE.parquet',index=False)
d['vendor_gap']=d.gap_logprice
d['reconstructed_gap']=100*np.log(d.arithmetic_mid_g/d.arithmetic_mid_c)
d['valid_quotes']=d.valid_quote_g&d.valid_quote_c
d=d.merge(p[['pair','contract_status','structural_mismatch','coupon_gap','reason']],on='pair',validate='many_to_one')

# Strict finite-sample covariance: no negative-variance clipping.
out=[];warnings=[]
def fit(z,label,terms=['post'],outcome='reconstructed_gap'):
    z=z.dropna(subset=[outcome]+terms).copy();g=z.groupby('pair').post.nunique();z=z[z.pair.isin(g[g.eq(2)].index)]
    P=z.pair.nunique();T=z.date.nunique();n=len(z);k=len(terms)
    if P<3 or T<3:return
    A=z[[outcome]+terms].astype(float);A-=A.groupby(z.pair).transform('mean');X=A[terms].to_numpy();y=A[outcome].to_numpy()
    if np.linalg.matrix_rank(X)<k:raise ValueError('Rank deficiency '+label)
    bread=np.linalg.inv(X.T@X);beta=bread@X.T@y;scores=X*(y-X@beta)[:,None]
    def meat(keys):
        x=pd.DataFrame(scores);x['key']=keys.to_numpy();s=x.groupby('key').sum().to_numpy();return s.T@s*len(s)/(len(s)-1)
    V=bread@(meat(z.pair)+meat(z.date)-meat(z.pair+'|'+z.date.astype(str)))@bread*(n-1)/(n-P-k)
    eig=np.linalg.eigvalsh((V+V.T)/2)
    if eig.min() < -1e-12:
        warnings.append(dict(test=label,issue='indefinite_covariance_no_joint_inversion',minimum_eigenvalue=float(eig.min())))
    se=np.where(np.diag(V)>=0,np.sqrt(np.maximum(np.diag(V),0)),np.nan)
    if np.diag(V).min()<0:warnings.append(dict(test=label,issue='negative_variance_no_inference',terms=[terms[j] for j in np.where(np.diag(V)<0)[0]]))
    crit=t.ppf(.975,min(P,T)-1)
    rowhash=hashlib.sha256('\n'.join(sorted(z.pair+'|'+z.date.astype(str))).encode()).hexdigest()
    for j,key in enumerate(terms):out.append(dict(test=label,term=key,outcome=outcome,coef=beta[j],se=se[j],lo=beta[j]-crit*se[j],hi=beta[j]+crit*se[j],p=2*t.sf(abs(beta[j]/se[j]),min(P,T)-1),pairs=P,n=n,days=T,row_hash=rowhash))
    return z,X,y,beta

fit(d,'historical_vendor_76',outcome='vendor_gap')
v=d[d.valid_quotes].copy()
fit(v,'vendor_same_quote_rows_76',outcome='vendor_gap');fit(v,'arithmetic_mid_76')
fit(v[v.ISIN_g.ne('XS2831094706')],'previous_broad_75')
main=v[~v.structural_mismatch].copy();fit(main,'revised_broad_74');fit(main,'vendor_revised_74',outcome='vendor_gap')
screen=main[main.contract_status.ne('ineligible')];fit(screen,'no_known_contract_failure')
fit(main[main.contract_status.eq('eligible')],'documented_plain_contracts')
close=main[main.coupon_gap.abs().le(.5)&main.contract_status.ne('ineligible')];fit(close,'close_coupon_no_known_failure')
for cap in [.5]:fit(main[main.maturity_gap_years.le(cap)],'revised_maturity_0.5')
main['trend']=(main.date-main.date.min()).dt.days
fit(main,'revised_trend',['post','trend'])
pre=main[~main.post].copy();pre['post']=pre.date.ge('2026-02-07');fit(pre,'revised_placebo')
count=main.groupby('pair').date.nunique();fit(main[main.pair.isin(count[count.eq(58)].index)],'revised_balanced')
z=main.sort_values(['pair','date']);unch=z.groupby('pair')[['arithmetic_mid_g','arithmetic_mid_c']].diff().eq(0).any(axis=1);fit(z[~unch],'revised_unchanged_removed')
for end in ['2026-03-06','2026-03-13','2026-04-01']:fit(main[main.date.le(end)],'revised_end_'+end)
main['week']=np.floor((main.date-pd.Timestamp('2026-02-28')).dt.days/7).astype(int);terms=[]
for week in sorted(main.week.unique()):
    if week!=-1:col='w'+str(week);main[col]=main.week.eq(week).astype(float);terms.append(col)
fit(main,'revised_event_weeks',terms)

# Currency-specific curves; exact row hashes prove same-sample comparisons.
c=main.merge(pd.read_csv(R/'us_rates.csv',parse_dates=['date']),on='date',how='left',validate='many_to_one')
for term in ['2y','10y']:
    a=pd.read_csv('green2026/review/raw/ecb_'+term+'.dat');a['date']=pd.to_datetime(a.TIME_PERIOD)
    c=c.merge(a[['date','OBS_VALUE']].rename(columns={'OBS_VALUE':'eu'+term}),on='date',how='left',validate='many_to_one')
c=c[c.RCSCurrencyLeaf.isin(['Euro','US Dollar'])].copy()
c['r2']=np.where(c.RCSCurrencyLeaf.eq('Euro'),c.eu2y,c.us2);c['r10']=np.where(c.RCSCurrencyLeaf.eq('Euro'),c.eu10y,c.us10)
c=c.dropna(subset=['r2','r10']);c['slope']=c.r10-c.r2
for currency,lab in [('Euro','eu'),('US Dollar','us')]:
    for var in ['r2','slope']:c[lab+'_'+var]=c[var]*c.RCSCurrencyLeaf.eq(currency)
curve_terms=['post','eu_r2','eu_slope','us_r2','us_slope']
validations=[]
common_samples = {
    'revised_common': c,
    'screened_common': c[c.contract_status.ne('ineligible')],
    'six_month_common': c[c.maturity_gap_years.le(0.5)],
}
for label, z in common_samples.items():
    fit(z,label+'_baseline');result=fit(z,label+'_curves',curve_terms)
    if result:
        q,X,y,beta=result
        full=pd.concat([pd.Series(1.,index=q.index),q[curve_terms].astype(float),pd.get_dummies(q.pair,drop_first=True,dtype=float)],axis=1).to_numpy()
        b=np.linalg.lstsq(full,q.reconstructed_gap.to_numpy(),rcond=None)[0]
        validations.append(dict(test=label,within_post=float(beta[0]),dummy_post=float(b[1]),absolute_difference=float(abs(beta[0]-b[1]))))

models=pd.DataFrame(out);models.to_csv(O/'revised_models.csv',index=False)
for label in common_samples:
    z=models[models.test.isin([label+'_baseline',label+'_curves'])&models.term.eq('post')]
    assert z.row_hash.nunique()==1 and z.n.nunique()==1
six_month_post = models[models.test.str.startswith('six_month_common_') & models.term.eq('post')]
six_month_post.to_csv(O/'six_month_common.csv', index=False)
assert all(x['absolute_difference']<1e-8 for x in validations)
main.to_parquet(O/'revised_panel_PRIVATE.parquet',index=False);c.to_parquet(O/'curve_panel_PRIVATE.parquet',index=False)
daily=main.groupby('date').agg(pairs=('pair','nunique'),mean_gap=('reconstructed_gap','mean'));daily.to_csv(O/'daily.csv')
changes=main.groupby(['pair','post']).reconstructed_gap.mean().unstack();changes['change']=changes[True]-changes[False];changes.to_csv(O/'pair_changes_PRIVATE.csv')
cases=changes.join(p.set_index('pair')[['IssuerName','contract_status']]);cases[cases.contract_status.eq('eligible')].to_csv(O/'documented_cases.csv')
liquidity={s:float((100*(main.loc[~main.post,'ASK_'+s]-main.loc[~main.post,'BID_'+s])/main.loc[~main.post,'arithmetic_mid_'+s]).mean()) for s in ['g','c']}
info=dict(contract_status_counts=p.contract_status.value_counts().to_dict(),measurement=measurement,main_pairs=int(main.pair.nunique()),main_n=len(main),curve_pairs=int(c.pair.nunique()),curve_n=len(c),main_currency_counts=main.drop_duplicates('pair').RCSCurrencyLeaf.value_counts().to_dict(),negative_pairs=int(changes.change.lt(0).sum()),median_pair_change=float(changes.change.median()),liquidity=liquidity,independent_checks=validations,warnings=warnings,strict_rule='Same issuer/currency; <=1y maturity; nominal fixed cash flows; positive coupons <=0.50pp apart; documented compatible seniority/security/frequency; no call, reset or extension; unknown fails documentary eligibility. Coupon payment dates recorded, not required identical. Retrospective rule, not preregistered.',price_definition='Arithmetic BID/ASK midpoint, quoted-price outcome only; clean/dirty basis not certified. No total-return or yield-greenium claim.',alternative_files_scanned=7909,alternative_matching_isins=0)
(O/'revision_audit.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2));print(models[models.term.eq('post')][['test','coef','lo','hi','p','pairs','n']].to_string(index=False))

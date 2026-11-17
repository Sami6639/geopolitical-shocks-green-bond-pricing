from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path('green2026/revision');O=B/'outputs'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
c=pd.read_csv(O/'core_local_curve_models.csv')
models=['same_rows_baseline','local_curve_controlled','external_yield_implied_clean_local_curves']
labels=['Quoted prices: baseline','Quoted prices: local curve','Yield-implied prices: local curve']
fig,axes=plt.subplots(1,2,figsize=(8,3.2),sharex=True,sharey=True)
for ax,issuer,col in zip(axes,['Australia','New Zealand'],['#24577a','#9d4e32']):
 z=c[(c.issuer==issuer)&(c.HAC_lags==7)].set_index('model').loc[models]
 ax.errorbar(z.coef,np.arange(3),xerr=[z.coef-z.lo,z.hi-z.coef],fmt='o',capsize=3,color=col)
 ax.axvline(0,color='#666666',lw=.8);ax.set_title(issuer);ax.set_yticks(range(3),labels);ax.grid(axis='x',alpha=.15)
axes[0].invert_yaxis();axes[0].set_xlim(-.065,.105)
fig.supxlabel('Post-event coefficient (log percentage points)',fontsize=10)
fig.tight_layout();fig.savefig(B/'figures/case_conditioning.png',dpi=240);plt.close(fig)
x=pd.read_csv(O/'cross_case_contrasts.csv')
specs=[('reconstructed_gap',False),('reconstructed_gap',True),('external_gap',False),('external_gap',True)]
labs=['Quoted / no trend','Quoted / case trends','Yield-implied / no trend','Yield-implied / case trends']
fig,axes=plt.subplots(1,2,figsize=(8,4.3),sharex=True,sharey=True)
for ax,sample,title in zip(axes,['original_dates','shared_dates'],['Original dates (57)','Shared dates (54)']):
 for lag,offset,color,marker in [(3,-.20,'#24577a','o'),(7,0,'#9d4e32','s'),(10,.20,'#58764a','^')]:
  rows=[x[(x['sample']==sample)&(x.outcome==o)&(x.trend==tr)&(x.lags==lag)].iloc[0] for o,tr in specs]
  z=pd.DataFrame(rows);v=z.NZ_minus_AU.to_numpy()
  ax.errorbar(v,np.arange(4)+offset,xerr=[v-z.lo.to_numpy(),z.hi.to_numpy()-v],fmt=marker,capsize=2,color=color,label=f'HAC {lag}',ms=4)
 ax.axvline(0,color='#666666',lw=.8);ax.set_title(title);ax.set_yticks(range(4),labs);ax.grid(axis='x',alpha=.15)
axes[0].invert_yaxis();axes[0].set_xlim(-.095,.015)
axes[0].set_ylim(3.65,-.5)
axes[0].legend(loc='lower left',fontsize=8,frameon=False,ncol=3,handletextpad=.3,columnspacing=.7)
fig.supxlabel('New Zealand − Australia (log percentage points)',fontsize=10)
fig.tight_layout();fig.savefig(B/'figures/contrast_sensitivity.png',dpi=240);plt.close(fig)
print('Generated two comparative figures from saved estimates')

"""Editable Office Math equations with right-aligned equation numbers."""
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.enum.text import WD_TAB_ALIGNMENT

def el(tag,*children):
 e=OxmlElement('m:'+tag)
 for c in children:e.append(c)
 return e
def r(s):
 e=el('r');t=el('t');t.text=s;e.append(t);return e
def seq(*items):return [r(x) if isinstance(x,str) else x for x in items]
def sub(base,index):return el('sSub',el('e',r(base)),el('sub',r(index)))
def sup(base,power):return el('sSup',el('e',r(base)),el('sup',r(power)))
def both(base,index,power):return el('sSubSup',el('e',r(base)),el('sub',r(index)),el('sup',r(power)))
def fraction(*parts):return el('f',el('num',*parts[0]),el('den',*parts[1]))
def equation(doc,num):
 y=lambda:sub('y','at')
 if num==1:items=seq(sub('y','it'),' = 100 × [ln(',both('P','it','g'),') − ln(',both('P','it','c'),')]')
 elif num==2:items=seq(sub('y','it'),' = ',sub('α','i'),' + β',sub('Post','t'),' + ',sub('ε','it'))
 elif num==3:items=seq(y(),' = ',sub('α','a'),' + ',sub('β','a'),sub('Post','t'),' + ',sub('λ','a'),sub('L','at'),' + ',sub('κ','a'),sub('S','at'),' + ',sub('ε','at'))
 elif num==4:items=seq(y(),' = α + η',sub('N','a'),' + β',sub('Post','t'),' + δ(',sub('N','a'),' × ',sub('Post','t'),') + ',both('θ','a','′'),sub('X','at'),' + ',sub('τ','a'),'t + ',sub('ε','at'))
 else:
  base=el('e',*seq('(1 + ',fraction(seq(sub('z','jt')),seq(sub('m','j'))),')'))
  power=el('sup',*seq('−(k − 1 + ',sub('f','jt'),')'))
  summand=el('e',sub('CF','jk'),el('sSup',base,power))
  prop=el('naryPr');ch=el('chr');ch.set(qn('m:val'),'∑');prop.append(ch)
  lim=el('limLoc');lim.set(qn('m:val'),'undOvr');prop.append(lim)
  summation=el('nary',prop,el('sub',r('k = 1')),el('sup',sub('K','jt')),summand)
  items=seq(sub('P','jt'),' = ',summation,' − ',sub('AI','jt'))
 p=doc.add_paragraph()
 p.paragraph_format.tab_stops.add_tab_stop(Inches(3.4),WD_TAB_ALIGNMENT.CENTER)
 p.paragraph_format.tab_stops.add_tab_stop(Inches(6.8),WD_TAB_ALIGNMENT.RIGHT)
 p.paragraph_format.space_before=Pt(6);p.paragraph_format.space_after=Pt(8)
 p.add_run('\t');p._p.append(el('oMath',*items));p.add_run('\t('+str(num)+')')

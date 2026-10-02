"""Fit HST->Roman colour terms on synthetic photometry of the training SEDs (LANL KN + OU CC),
one fit for both classes (class-blind), and report the residual scatter per class."""
import numpy as np, pyarrow.parquet as pq, pandas as pd, json
S="/tmp/claude-1000/-home-nicolas-nico-git-spectro-transformer/507fa92d-aefa-4fe8-ba0e-ff9ee04b18c6/scratchpad/hst/"
R="/home/nicolas/nico/git/Kilonova/"
names={"R062":"Roman_WFI.F062","Z087":"Roman_WFI.F087","Y106":"Roman_WFI.F106","J129":"Roman_WFI.F129","H158":"Roman_WFI.F158",
 "F606W":"HST_WFC3_UVIS2.F606W","F625W":"HST_ACS_WFC.F625W","F814W":"HST_ACS_WFC.F814W","F105W":"HST_WFC3_IR.F105W",
 "F110W":"HST_WFC3_IR.F110W","F125W":"HST_WFC3_IR.F125W","F160W":"HST_WFC3_IR.F160W","F184":"Roman_WFI.F184","N090":"JWST_NIRCam.F090W","N115":"JWST_NIRCam.F115W","N150":"JWST_NIRCam.F150W","N200":"JWST_NIRCam.F200W"}
F={k:np.loadtxt(S+v+".dat") for k,v in names.items()}
def mags(lam,fl):
    out={}
    for k,(w,t) in ((k,v.T) for k,v in F.items()):
        m=t>1e-3*t.max()
        if lam.min()>w[m].min() or lam.max()<w[m].max(): out[k]=np.nan; continue
        num=np.trapezoid(np.interp(w,lam,fl)*t*w,w); den=np.trapezoid(t/w,w)*2.998e18
        out[k]=-2.5*np.log10(num/den) if num>0 else np.nan
    return out
rng=np.random.default_rng(0)
md=pq.read_schema(R+"data/dust_generation/lanl_spectra.parquet").metadata
wl=np.frombuffer(md[b"wavelength_rest_aa"],dtype=np.float32).astype(float)
pf=pq.ParquetFile(R+"data/dust_generation/lanl_spectra.parquet")
kn=[]
for rg in rng.choice(pf.num_row_groups,40,replace=False):
    t=pf.read_row_group(int(rg),columns=["time_days","flux_rest"]); td=t["time_days"].to_numpy()
    sel=np.where((td>=1)&(td<=15))[0]
    for i in rng.choice(sel,min(60,len(sel)),replace=False): kn.append(np.asarray(t["flux_rest"][int(i)].as_py(),float))
c=np.load(R+"data/openuniverse/cc_templates.npz",allow_pickle=True); cw=c["wavelength"].astype(float); cc=[]
for j in range(len(c["template_names"])):
    ph=c[f"phase_{j}"]
    for i in np.where((ph>=-5)&(ph<=20))[0][::3]: cc.append(c[f"flux_{j}"][i])
rows=[]
for z in [0.0098,0.02,0.05,0.1,0.162,0.2,0.3,0.5]:
    for f in kn: rows.append(dict(cls="KN",z=z,**mags(wl*(1+z),f)))
for z in [0.02,0.1,0.3,0.5,0.8,1.0,1.31,1.6,2.0]:
    for f in cc: rows.append(dict(cls="CC",z=z,**mags(cw*(1+z),f)))
d=pd.DataFrame(rows); d.to_parquet(S+"synth_mags_all.parquet")
# target <- (hst band, predictor colours)
SPECS={
       "R062<F606W|rzy":("R062","F606W",[("F606W","F814W"),("F814W","F105W")]),
       "Z087<F814W|rzy":("Z087","F814W",[("F606W","F814W"),("F814W","F105W")]),
       "Y106<F105W|rzy":("Y106","F105W",[("F606W","F814W"),("F814W","F105W")]),
       "Z087<N090|zyj":("Z087","N090",[("N090","N115"),("N115","N150")]),
       "Y106<N115":("Y106","N115",[("N090","N115"),("N115","N150")]),
       "J129<N115":("J129","N115",[("N090","N115"),("N115","N150")]),
       "J129<N150":("J129","N150",[("N090","N115"),("N115","N150")]),
       "Z087<N090|zhf":("Z087","N090",[("N090","N150"),("N150","N200")]),
       "H158<N150":("H158","N150",[("N090","N150"),("N150","N200")]),
       "F184<N200":("F184","N200",[("N090","N150"),("N150","N200")]),"R062<F606W":("R062","F606W",[("F606W","F110W"),("F110W","F160W")]),
       "J129<F110W":("J129","F110W",[("F606W","F110W"),("F110W","F160W")]),
       "H158<F160W":("H158","F160W",[("F606W","F110W"),("F110W","F160W")]),
       "R062<F625W":("R062","F625W",[("F625W","F110W"),("F110W","F160W")]),
       "J129<F110W|625":("J129","F110W",[("F625W","F110W"),("F110W","F160W")]),
       "H158<F160W|625":("H158","F160W",[("F625W","F110W"),("F110W","F160W")]),
       "Z087<F814W":("Z087","F814W",[("F814W","F105W"),("F105W","F125W")]),
       "Y106<F105W":("Y106","F105W",[("F814W","F105W"),("F105W","F125W")]),
       "J129<F125W":("J129","F125W",[("F814W","F105W"),("F105W","F125W")])}
def design(df,cols):
    X=[np.ones(len(df))]
    cs=[df[a]-df[b] for a,b in cols]
    X+=cs+[x**2 for x in cs]+[cs[0]*cs[1]]
    return np.vstack(X).T
fits={}
for key,(tgt,src,cols) in SPECS.items():
    need=[tgt,src]+[x for p in cols for x in p]
    dd=d.dropna(subset=need)
    w=np.where(dd.cls=="KN",(dd.cls=="CC").sum()/(dd.cls=="KN").sum(),1.0)
    X=design(dd,cols); y=(dd[tgt]-dd[src]).values
    coef,*_=np.linalg.lstsq(X*np.sqrt(w)[:,None],y*np.sqrt(w),rcond=None); res=y-X@coef
    st={c:float(res[(dd.cls==c).values].std()) for c in ["KN","CC"]}
    raw={c:float(np.median(y[(dd.cls==c).values])) for c in ["KN","CC"]}
    fits[key]=dict(target=tgt,src=src,cols=cols,coef=coef.tolist(),resid_std=st,raw_median=raw)
    print(f"{key:16s} sin corr. mediana KN {raw['KN']:+.3f} CC {raw['CC']:+.3f} | residuo std KN {st['KN']:.3f} CC {st['CC']:.3f}")
json.dump(fits,open(S+"colorterms_all.json","w"),indent=1)

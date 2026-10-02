import sys, json, numpy as np, torch
sys.path.insert(0,"/home/nicolas/nico/git/Kilonova/training")
from openuniverse_data import collate_token_windows, BAND_TO_INDEX, TOKEN_TYPE_TO_INDEX
from train_lightning import LitKilonova, MODEL_INPUT_KEYS
S="/tmp/claude-1000/-home-nicolas-nico-git-spectro-transformer/507fa92d-aefa-4fe8-ba0e-ff9ee04b18c6/scratchpad/hst/"
NORM={'MAG_MEAN': 22.966102600097656, 'MAG_STD': 2.986361503601074, 'SIGMA_MAG_MEAN': 0.05732026323676109, 'SIGMA_MAG_STD': 0.06428132206201553}
CT=json.load(open(S+"colorterms_all.json")); NZ={"deep":json.load(open(S+"roman_deep_noise.json")),"wide":json.load(open(S+"roman_wide_noise.json"))}
TIER={"wide":"RZYJH","deep":"ZYJHF"}
V2AB={"F606W":0.086,"F625W":0.17,"F814W":0.424,"F105W":0.645,"F110W":0.760,"F160W":1.251}   # AB - Vega
def to_roman(tier,ep,keys):
    out={}
    for k in keys:
        f=CT[k]; cs=[ep[a]-ep[b] for a,b in f["cols"]]; X=[1.0]+cs+[x**2 for x in cs]+[cs[0]*cs[1]]
        L="F" if f["target"]=="F184" else f["target"][0]
        m=ep[f["src"]]+float(np.dot(f["coef"],X)); rs=0.5*(f["resid_std"]["KN"]+f["resid_std"]["CC"]); n=NZ[tier][L]
        out[L]=(m,float(np.hypot(np.interp(m,n["mag"],n["sigma"]),rs)),True) if m<n["lim5_median"] else (n["lim5_median"],np.nan,False)
    return out
def item(tier,visits,z):
    rows=[]
    for t,meas in visits:
        for b in TIER[tier]:
            if b in meas: m,e,d=meas[b]; rows.append((t,BAND_TO_INDEX[b],TOKEN_TYPE_TO_INDEX["d" if d else "u"],m,e if d else np.nan))
            else: rows.append((t,BAND_TO_INDEX[b],TOKEN_TYPE_TO_INDEX["n"],np.nan,np.nan))
    rows.sort(key=lambda r:(r[0],r[1])); a=np.array(rows,float); mag,sig=a[:,3],a[:,4]; mm,sm=~np.isnan(mag),~np.isnan(sig); f=lambda x:torch.tensor(x,dtype=torch.float32)
    return {"delta_time":f(a[:,0]-a[0,0]),"band_index":torch.tensor(a[:,1],dtype=torch.long),"token_type_index":torch.tensor(a[:,2],dtype=torch.long),
      "magnitude":f(np.where(mm,(mag-NORM["MAG_MEAN"])/NORM["MAG_STD"],0)),"sigma_magnitude":f(np.where(sm,(sig-NORM["SIGMA_MAG_MEAN"])/NORM["SIGMA_MAG_STD"],0)),
      "magnitude_mask":f(mm*1.0),"sigma_mask":f(sm*1.0),"redshift":f(z if z is not None else 0.0),"has_redshift":f(1.0 if z is not None else 0.0),
      "label":torch.tensor(0),"cid":torch.tensor(0),"true_redshift":f(0.0)}
model=LitKilonova.load_from_checkpoint("/home/nicolas/nico/git/Kilonova/training/checkpoints/izc-trainonly-2026-09-30/kilonova_transformer-soup.ckpt",class_weights=torch.ones(2),map_location="cpu").eval()
def p(it):
    b=collate_token_windows([it])
    with torch.no_grad(): return float(torch.softmax(model({k:v for k,v in b.items() if k in MODEL_INPUT_KEYS}),1)[0,1])
ab=lambda d:{k:v+V2AB[k] for k,v in d.items()}
import pandas as pd
A_EBV={"F606W":2.47,"F625W":2.26,"F814W":1.53,"F105W":0.97,"F110W":0.88,"F125W":0.73,"F160W":0.51}
red=lambda d,ebv:{k:v+A_EBV[k]*ebv for k,v in d.items()}
ab=lambda d:{k:v+V2AB[k] for k,v in d.items()}
RJH=["R062<F606W","J129<F110W","H158<F160W"]; RJH625=["R062<F625W","J129<F110W|625","H158<F160W|625"]; RZY=["R062<F606W|rzy","Z087<F814W|rzy","Y106<F105W|rzy"]
ZYJh=["Z087<F814W","Y106<F105W","J129<F125W"]
ZYJ=["Z087<N090|zyj","Y106<N115","J129<N115"]; ZHF=["Z087<N090|zhf","H158<N150","F184<N200"]
RISE=18.0  # SN Ia rest-frame rise time used to estimate explosion (approx.)
def ia(tpk,t,z): ph=(t-tpk)/(1+z); return f"{(t-tpk):+.1f} obs / {ph:+.1f} rest del pico", f"~{ph+RISE:.0f} (estimado: pico − {RISE:.0f} d)"
# object, class, z, z_type, ref, tier, [ (combo, t, obs_mags_shown, mags_for_conversion, keys, phase_peak, phase_expl) ], multi
O=[]
t0=57621.94; z=0.162
O.append(("GRB 160821B (Troja+19)","KN (sGRB)",z,"spec","Troja+2019, MNRAS 489, 2104 (arXiv:1905.01290), Tabla 1","wide",[
  ("RJH",57625.63,{"F606W":"26.02±0.06","F110W":"24.82±0.05","F160W":"24.53±0.08"},red({"F606W":26.02,"F110W":24.82,"F160W":24.53},0.04),RJH,"—",f"+{57625.63-t0:.1f} obs / +{(57625.63-t0)/(1+z):.1f} rest (trigger)"),
  ("RJH",57632.39,{"F606W":"27.9±0.3","F110W":"26.9±0.4","F160W":"26.6±0.3"},red({"F606W":27.9,"F110W":26.9,"F160W":26.6},0.04),RJH,"—",f"+{57632.39-t0:.1f} obs / +{(57632.39-t0)/(1+z):.1f} rest (trigger)")],True))
O.append(("GRB 160821B (Lamb+19)","KN (sGRB)",z,"spec","Lamb+2019, ApJ 883, 48 (arXiv:1905.02159), Tabla 1","wide",[
  ("RJH",57625.64,{"F606W":"25.90±0.06","F110W":"24.69±0.02","F160W":"24.43±0.03"},red({"F606W":25.90,"F110W":24.69,"F160W":24.43},0.038),RJH,"—",f"+{3.7:.1f} obs / +{3.7/(1+z):.1f} rest (trigger)"),
  ("RJH",57632.40,{"F606W":"27.55±0.11","F110W":"26.69±0.15","F160W":"26.55±0.23"},red({"F606W":27.55,"F110W":26.69,"F160W":26.55},0.038),RJH,"—",f"+{10.5:.1f} obs / +{10.5/(1+z):.1f} rest (trigger)")],True))
O.append(("AT2017gfo","KN (GW170817)",0.0098,"spec","Cowperthwaite+2017, ApJL 848, L17 (vía Villar+2017)","wide",[
  ("RJH",57992.39,{"F625W":"22.88±0.07","F110W":"20.57±0.04","F160W":"19.89±0.04"},{"F625W":22.88,"F110W":20.57,"F160W":19.89},RJH625,"—","+9.8 obs / +9.8 rest (trigger)")],False))
O.append(("AT2025ulz","SN IIb (spec)",0.08489,"spec","arXiv:2510.18854 (ApJL), Tabla GTC/HST","wide",[
  ("RJH",58000,{"F606W":"21.78±0.02","F110W":"21.97±0.03","F160W":"22.26±0.04"},{"F606W":21.78,"F110W":21.97,"F160W":22.26},RJH,"—","+8.8–10.4 obs desde el trigger GW (explosión no acotada)")],False))
pa=ia(56816.3,56833.0,1.3457)
O.append(("HFF14Tom","SN Ia (spec)",1.3457,"spec","Rodney+2015, ApJ 811, 70; fotometría github.com/srodney/snTomas","deep",[
  ("ZYJ",56833.0,{"F814W":"25.03±0.05","F105W":"24.08±0.03","F125W":"24.01±0.04"},{"F814W":25.03,"F105W":24.08,"F125W":24.01},ZYJh,pa[0],pa[1])],False))
O.append(("CLN12Did","SN Ia (fot.)",0.851,"spec (host)","Patel+2014, ApJ 786, 9 (arXiv:1312.0943), Tabla did-phot (Vega→AB)","wide",[
  ("RJH",55990.3,{"F625W":"24.182±0.023 V","F110W":"22.915±0.013 V","F160W":"23.041±0.041 V"},ab({"F625W":24.182,"F110W":22.915,"F160W":23.041}),RJH625,"pico no publicado",f"≥ +{55990.3-55960.64:.1f} obs / ≥ +{(55990.3-55960.64)/1.851:.1f} rest (desde 1ª detección)"),
  ("RZY",56002.64,{"F606W":"25.491±0.038 V","F814W":"23.410±0.018 V","F105W":"23.138±0.019 V"},ab({"F606W":25.491,"F814W":23.410,"F105W":23.138}),RZY,"pico no publicado",f"≥ +{56002.64-55960.64:.1f} obs / ≥ +{(56002.64-55960.64)/1.851:.1f} rest (desde 1ª detección)")],True))
for img,tpk in [("2b",60037.77),("2c",60037.77-48.6)]:
    e1=ia(tpk,60033,1.783); e2=ia(tpk,60057,1.783)
    M={"2b":[(25.53,24.69,23.93,24.21),(26.44,None,24.03,24.35)],"2c":[(28.71,26.16,24.95,25.25),(30.23,None,25.45,25.06)]}[img]
    a,b=M
    O.append((f"SN H0pe {img}","SN Ia (spec), lente",1.783,"spec","Pierel+2024, ApJ 967, 50 (arXiv:2403.18954), Tabla im_mags","deep",[
      ("ZYJ",60033,{"F090W":a[0],"F115W":a[1],"F150W":a[2]},{"N090":a[0],"N115":a[1],"N150":a[2]},ZYJ,e1[0],e1[1]),
      ("ZHF",60033,{"F090W":a[0],"F150W":a[2],"F200W":a[3]},{"N090":a[0],"N150":a[2],"N200":a[3]},ZHF,e1[0],e1[1]),
      ("ZHF",60057,{"F090W":b[0],"F150W":b[2],"F200W":b[3]},{"N090":b[0],"N150":b[2],"N200":b[3]},ZHF,e2[0],e2[1])],False))
JADES=[("JD22-16","II",1.77,26.40,25.73,25.27,25.15),("JD22-17","II",1.00,27.32,26.51,26.13,26.17),("JD22-1","Ia",1.69,27.31,26.46,26.48,26.74),
       ("JD22-3","II",0.665,27.47,26.83,26.53,26.49),("JD22-2","Ia",1.79,27.94,26.93,26.67,26.60),("JD23-14","II",0.657,30.15,28.12,26.53,25.91),("JD23-24","II",1.01,27.21,26.80,26.45,26.57)]
for n,ty,zz,a,b,c,d in JADES:
    O.append((f"JADES {n}",f"SN {ty} (fot.)",zz,"spec (host)","DeCoursey+2025, ApJ 979, 250 (arXiv:2406.05060), tablas PSF JD22/JD23","deep",[
      ("ZYJ",0,{"F090W":a,"F115W":b,"F150W":c},{"N090":a,"N115":b,"N150":c},ZYJ,"desconocida","desconocida (1 época de diferencia)"),
      ("ZHF",0,{"F090W":a,"F150W":c,"F200W":d},{"N090":a,"N150":c,"N200":d},ZHF,"desconocida","desconocida (1 época de diferencia)")],False))
rows=[]
for name,cls,z,zt,ref,tier,eps,multi in O:
    vis=[]
    for combo,t,shown,mags,keys,php,phe in eps:
        r=to_roman(tier,mags,keys)
        if not any(d for _,_,d in r.values()):
            rows.append(dict(objeto=name,clase=cls,z=z,z_tipo=zt,tier=tier,combo=combo,fase_desde_explosion_dias=phe,fase_pico=php,
              obs=" ".join(f"{k}={v}" for k,v in shown.items()),roman="sin detección en Roman",P_KN_con_z=None,P_KN_sin_z=None,ref=ref)); continue
        vis.append((t,r))
        if t!=eps[0][1]: continue   # una epoca posterior no se clasifica sola: si hubo una anterior, va con ella
        rows.append(dict(objeto=name,clase=cls,z=z,z_tipo=zt,tier=tier,combo=combo,fase_desde_explosion_dias=phe,fase_pico=php,
          obs=" ".join(f"{k}={v}" for k,v in shown.items()),roman=" ".join(f"{k}={m:.2f}{'' if d else ' (lím)'}" for k,(m,e,d) in r.items()),
          P_KN_con_z=round(p(item(tier,[(t,r)],z)),3),P_KN_sin_z=round(p(item(tier,[(t,r)],None)),3),ref=ref))
    if multi and len(vis)>1:
        rows.append(dict(objeto=name,clase=cls,z=z,z_tipo=zt,tier=tier,combo="+".join(e[0] for e in eps)+" (2 épocas)",fase_desde_explosion_dias=f"Δt = {vis[1][0]-vis[0][0]:.1f} d obs",fase_pico="",
          obs="",roman="",P_KN_con_z=round(p(item(tier,vis,z)),3),P_KN_sin_z=round(p(item(tier,vis,None)),3),ref=ref))
df=pd.DataFrame(rows); df.to_csv(S+"resumen_transientes_reales.csv",index=False)
pd.set_option("display.width",250,"display.max_colwidth",60)
print(df[["objeto","combo","fase_desde_explosion_dias","roman","P_KN_con_z","P_KN_sin_z"]].to_string())

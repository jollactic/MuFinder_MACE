#!/usr/bin/env python3
from pathlib import Path
import argparse
import json, shutil
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
from ase import Atom
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from mufinder.core.sitegen.gen_sites import gen_random

VERSION = "1.0.1"

D = dict(level="standard", supercell=[1,1,1], n_sites=20, zpe=True,
         dispersion=False, calculator="mace-polar", model="polar-1-m", device="cpu",
         dtype="float64", minr=.5, vdws=1., seed=123, host_fmax=.02,
         host_steps=500, muon_fmax=.02, muon_steps=500, cluster_tol=.5,
         zpe_delta=.005, muon_mass=.11342892566, output_dir="muon_candidates")

def boolean(x):
    x=x.lower()
    if x in ("true","yes","1","on"): return True
    if x in ("false","no","0","off"): return False
    raise ValueError(x)

def config(fn):
    c=D.copy()
    ints={"n_sites","seed","host_steps","muon_steps"}
    floats={"minr","vdws","host_fmax","muon_fmax","cluster_tol","zpe_delta","muon_mass"}
    strings={"structure","level","calculator","model","device","dtype","output_dir"}
    for raw in open(fn):
        s=raw.split("#",1)[0].strip()
        if not s: continue
        if "=" not in s: raise ValueError(f"Bad input line: {raw.rstrip()}")
        k,v=map(str.strip,s.split("=",1)); k=k.lower()
        if k in strings: c[k]=v
        elif k in ints: c[k]=int(v)
        elif k in floats: c[k]=float(v)
        elif k in ("zpe", "dispersion"): c[k]=boolean(v)
        elif k=="supercell":
            c[k]=[int(q) for q in v.replace(","," ").split()]
            if len(c[k])!=3 or min(c[k])<1: raise ValueError("supercell needs 3 positive integers")
        else: raise ValueError(f"Unknown option: {k}")
    if "structure" not in c: raise ValueError("structure = FILE is required")
    c["level"]=c["level"].lower()
    if c["level"] not in ("quick","standard","full"): raise ValueError("level: quick|standard|full")
    return c

def calculator(c):
    fam=c["calculator"].lower().replace("_","-")
    if fam in ("mace-polar","polar"):
        from mace.calculators import mace_polar
        return mace_polar(model=c["model"],device=c["device"],default_dtype=c["dtype"])
    if fam in ("mace-mp","mace-mp0","mp","mp0"):
        from mace.calculators import mace_mp
        return mace_mp(model=c["model"],device=c["device"],default_dtype=c["dtype"],dispersion=c["dispersion"])
    raise ValueError("calculator must be mace-polar or mace-mp")

def state(a):
    a.info["charge"]=1; a.info["spin"]=1; a.info["external_field"]=[0.,0.,0.]

def mu_index(d,a):
    f=d/"metadata.json"
    i=json.load(open(f))["muon_index"] if f.exists() else len(a)-1
    i=int(i)
    if a[i].symbol!="H": raise RuntimeError(f"{d}: muon index {i} is {a[i].symbol}")
    return i

def clean_write(fn,a,vasp=False):
    b=a.copy(); b.calc=None
    if vasp: write(fn,b,format="vasp",direct=True,vasp5=True,sort=False)
    else: write(fn,b)

def host(c,calc):
    hd=Path("host"); hd.mkdir(exist_ok=True)
    sf=hd/"host_supercell.extxyz"; of=hd/"host_optimized.extxyz"
    ef=hd/"host_energy.dat"
    if sf.exists():
        a=read(sf)
        if not ef.exists():
            a.set_constraint()
            state(a)
            a.info["charge"]=0
            a.info["spin"]=1
            a.calc=calc
            host_E=float(a.get_potential_energy())
            ef.write_text(f"{host_E:.16f}\n")
            a.calc=None
        return a
    a=read(c["structure"])
    if c["level"]!="quick":
        if of.exists(): a=read(of)
        else:
            a.calc=calc
            opt=BFGS(FrechetCellFilter(a),trajectory=str(hd/"host_opt.traj"),
                     logfile=str(hd/"host_opt.log"))
            print("Host converged:",opt.run(fmax=c["host_fmax"],steps=c["host_steps"]))
            clean_write(of,a)
    a=a.repeat(tuple(c["supercell"])); a.set_constraint()
    # Store the pristine-host energy used as the common insertion-energy reference.
    state(a)
    a.info["charge"] = 0
    a.info["spin"] = 1
    a.calc=calc
    host_E=float(a.get_potential_energy())
    ef.write_text(f"{host_E:.16f}\n")
    clean_write(sf,a)
    return a

def sites(c,h):
    root=Path(c["output_dir"]); root.mkdir(exist_ok=True)
    sf=root/"generated_sites.dat"
    if sf.exists(): ps=np.loadtxt(sf,ndmin=2)[:,:3]
    else:
        ps=np.asarray(gen_random(h,sites=[],size=c["n_sites"],minr=c["minr"],
                                 vdws=c["vdws"],seed=c["seed"]))
        np.savetxt(sf,ps,header="fx fy fz")
    for n,p in enumerate(ps):
        d=root/f"site_{n:03d}"; d.mkdir(exist_ok=True)
        ini=d/"initial.extxyz"
        if ini.exists() and (d/"metadata.json").exists(): continue
        a=h.copy(); a.append(Atom("H",position=h.cell.cartesian_positions(p))); im=len(a)-1
        json.dump(dict(site=n,muon_index=im,initial_fractional=list(map(float,p)),
                       n_host_atoms=len(h)),open(d/"metadata.json","w"),indent=2)
        clean_write(ini,a); clean_write(d/"POSCAR",a,True)
    return root

def relax(c,calc,root):
    out=[]
    for d in sorted(root.glob("site_*")):
        ini=d/"initial.extxyz"; rf=d/"relaxed.extxyz"
        if not ini.exists(): continue
        if rf.exists(): a=read(rf); im=mu_index(d,a); conv=True
        else:
            a=read(ini); im=mu_index(d,a); a.set_constraint()
            if c["level"] in ("quick","standard"):
                a.set_constraint(FixAtoms(indices=[i for i in range(len(a)) if i!=im]))
            state(a); a.calc=calc
            opt=BFGS(a,trajectory=str(d/"relax.traj"),logfile=str(d/"relax.log"))
            conv=opt.run(fmax=c["muon_fmax"],steps=c["muon_steps"])
            write(rf,a); clean_write(d/"CONTCAR_MACE",a,True)
        state(a); a.calc=calc; E=float(a.get_potential_energy())
        neigh=sorted((float(a.get_distance(im,j,mic=True)),j,x.symbol)
                     for j,x in enumerate(a) if j!=im)
        out.append(dict(site=d.name,dir=d,E=E,im=im,conv=conv,near=neigh,
                        frac=a.get_scaled_positions(wrap=True)[im]))
        print(f"{d.name}: E={E:.8f} eV nearest={neigh[0][2]}{neigh[0][1]} {neigh[0][0]:.3f} A")
    E0=min(x["E"] for x in out)
    with open(root/"relaxation_summary.dat","w") as f:
        f.write("# site converged E_eV dE_eV fx fy fz nearest distance_A\n")
        for x in sorted(out,key=lambda q:q["E"]):
            r,j,s=x["near"][0]; p=x["frac"]
            f.write(f"{x['site']:12s} {int(x['conv'])} {x['E']:18.10f} {x['E']-E0:12.8f} "
                    f"{p[0]:12.8f} {p[1]:12.8f} {p[2]:12.8f} {s}{j} {r:10.6f}\n")
    return out

def mudist(a,ia,b,ib):
    fa=a.get_scaled_positions(wrap=True)[ia]; fb=b.get_scaled_positions(wrap=True)[ib]
    df=fa-fb; df-=np.round(df)
    return float(np.linalg.norm(df@a.cell.array))

def cluster(c,res,root):
    ordered=sorted(res,key=lambda x:x["E"]); structures={x["site"]:read(x["dir"]/"relaxed.extxyz") for x in ordered}
    cs=[]
    for x in ordered:
        hit=None
        for k,C in enumerate(cs):
            y=C["rep"]
            if mudist(structures[x["site"]],x["im"],structures[y["site"]],y["im"])<=c["cluster_tol"]:
                hit=k; break
        if hit is None: cs.append(dict(rep=x,members=[x]))
        else: cs[hit]["members"].append(x)
    u=root/"unique"; u.mkdir(exist_ok=True); E0=cs[0]["rep"]["E"]
    with open(root/"clusters.dat","w") as f:
        f.write("# cluster representative E_eV dE_eV n_members members\n")
        for k,C in enumerate(cs):
            x=C["rep"]; ms=[m["site"] for m in C["members"]]
            f.write(f"{k:4d} {x['site']:12s} {x['E']:18.10f} {x['E']-E0:12.8f} {len(ms):4d} {' '.join(ms)}\n")
            shutil.copyfile(x["dir"]/"relaxed.extxyz",u/f"cluster_{k:03d}_{x['site']}.extxyz")
    # Machine-readable cluster membership for --status and later tools.
    with open(root/"clusters.json","w") as jf:
        json.dump([
            {
                "cluster": k,
                "representative": C["rep"]["site"],
                "members": [m["site"] for m in C["members"]],
            }
            for k, C in enumerate(cs)
        ], jf, indent=2)
    print(f"Unique clusters: {len(cs)} / {len(res)}")
    return cs

def zpe_one(c,calc,k,x):
    a=read(x["dir"]/"relaxed.extxyz"); im=mu_index(x["dir"],a)
    a.set_constraint(FixAtoms(indices=[i for i in range(len(a)) if i!=im]))
    state(a); m=a.get_masses(); m[im]=c["muon_mass"]; a.set_masses(m); a.calc=calc
    zd=x["dir"]/"zpe"; zd.mkdir(exist_ok=True)
    v=Vibrations(a,indices=[im],name=str(zd/"vib"),delta=c["zpe_delta"]); v.run()
    es=np.asarray(v.get_energies())
    if len(es)!=3: raise RuntimeError(f"Expected 3 modes, got {len(es)}")
    if np.any(np.abs(np.imag(es))>1e-8): z=np.nan; ec=np.nan
    else: z=float(.5*np.sum(np.real(es))); ec=x["E"]+z
    return dict(cluster=k,site=x["site"],E=x["E"],zpe=z,Ec=ec,modes=es)

def zpes(c,calc,cs,root):
    rr=[]
    for k,C in enumerate(cs):
        try:
            r=zpe_one(c,calc,k,C["rep"]); rr.append(r)
            print(f"ZPE cluster {k}: {r['zpe']:.6f} eV")
        except Exception as e: print(f"WARNING ZPE cluster {k} failed: {e}")
    good=[r for r in rr if np.isfinite(r["Ec"])]
    e0=min((r["Ec"] for r in good),default=np.nan)
    with open(root/"zpe_summary.dat","w") as f:
        f.write("# cluster site E_static_eV ZPE_eV E_corrected_eV dE_corrected_eV mode1 mode2 mode3\n")
        for r in rr:
            q=np.real(r["modes"])
            f.write(f"{r['cluster']:4d} {r['site']:12s} {r['E']:18.10f} {r['zpe']:12.8f} "
                    f"{r['Ec']:18.10f} {r['Ec']-e0:12.8f} {q[0]:12.8f} {q[1]:12.8f} {q[2]:12.8f}\n")
    return rr


def _load_result_database(root):
    """Load results.json if present."""
    fn = Path(root) / "results.json"
    if not fn.exists():
        raise FileNotFoundError(f"No results file found: {fn}")
    return json.loads(fn.read_text())


def _result_database(c, root, results, clusters, zpe_results=None):
    """Write a compact machine-readable summary."""
    host_energy = None
    hef = Path("host") / "host_energy.dat"
    if hef.exists():
        try:
            host_energy = float(hef.read_text().strip())
        except Exception:
            pass

    db = {
        "version": VERSION,
        "structure": c["structure"],
        "calculator": c["calculator"],
        "model": c["model"],
        "dispersion": bool(c.get("dispersion", False)),
        "dtype": c["dtype"],
        "host_energy_eV": host_energy,
        "sites": [],
        "clusters": [],
    }

    emin = min(x["E"] for x in results) if results else None
    zmap = {z["site"]: z for z in (zpe_results or [])}
    corrected_vals = []
    for z in (zpe_results or []):
        if np.isfinite(z["zpe"]):
            corrected_vals.append(z["E"] + z["zpe"])
    corrected_min = min(corrected_vals) if corrected_vals else None

    for x in sorted(results, key=lambda q: q["E"]):
        d, j, s = x["near"][0]
        item = {
            "site": x["site"],
            "converged": bool(x["conv"]),
            "energy_eV": float(x["E"]),
            "delta_energy_eV": float(x["E"] - emin) if emin is not None else None,
            "nearest": {
                "element": s,
                "index": int(j),
                "distance_A": float(d),
            },
        }
        if host_energy is not None:
            item["insertion_energy_eV"] = float(x["E"] - host_energy)
        if x["site"] in zmap:
            z = zmap[x["site"]]
            if np.isfinite(z["zpe"]):
                item["zpe_eV"] = float(z["zpe"])
                item["corrected_energy_eV"] = float(x["E"] + z["zpe"])
                item["delta_corrected_eV"] = float(x["E"] + z["zpe"] - corrected_min)
                if host_energy is not None:
                    item["corrected_insertion_energy_eV"] = float(
                        x["E"] - host_energy + z["zpe"]
                    )
        db["sites"].append(item)

    for C in clusters:
        db["clusters"].append({
            "representative": C["rep"]["site"],
            "members": [m["site"] for m in C["members"]],
        })

    (Path(root) / "results.json").write_text(json.dumps(db, indent=2))
    return db


def _print_results(db):
    print()
    print("=" * 80)
    print(" MUON SITE SEARCH — KEY RESULTS")
    print("=" * 80)

    label = f"{db['calculator']} / {db['model']}"
    if db.get("dispersion", False):
        label += " + D3"
    print(f"Calculator           : {label}")
    if db.get("host_energy_eV") is not None:
        print(f"Host energy          : {db['host_energy_eV']:.10f} eV")
    print(f"Candidate sites      : {len(db.get('sites', []))}")
    print(f"Unique clusters      : {len(db.get('clusters', []))}")
    print()

    has_zpe = any("zpe_eV" in x for x in db["sites"])
    if has_zpe:
        print(
            f"{'rank':>4} {'site':>10} {'dE/eV':>9} {'Einsert/eV':>12} "
            f"{'ZPE/eV':>9} {'Ecorr/eV':>11} {'dE_ZPE/eV':>11} {'nearest':>9} {'r/A':>7}"
        )
        print("-" * 96)
    else:
        print(
            f"{'rank':>4} {'site':>12} {'dE/eV':>10} "
            f"{'Einsert/eV':>13} {'nearest':>10} {'r/A':>8}"
        )
        print("-" * 68)

    for rank, x in enumerate(sorted(db["sites"], key=lambda q: q["energy_eV"])):
        n = x["nearest"]
        eins = x.get("insertion_energy_eV")
        eins_s = f"{eins:12.5f}" if eins is not None else f"{'n/a':>12}"
        nearest = f"{n['element']}{n['index']}"
        if has_zpe:
            z = x.get("zpe_eV")
            ec = x.get("corrected_insertion_energy_eV")
            dz = x.get("delta_corrected_eV")
            zs = f"{z:9.5f}" if z is not None else f"{'n/a':>9}"
            ecs = f"{ec:11.5f}" if ec is not None else f"{'n/a':>11}"
            dzs = f"{dz:11.5f}" if dz is not None else f"{'n/a':>11}"
            print(
                f"{rank:4d} {x['site']:>10} {x['delta_energy_eV']:9.4f} "
                f"{eins_s} {zs} {ecs} {dzs} {nearest:>9} {n['distance_A']:7.3f}"
            )
        else:
            print(
                f"{rank:4d} {x['site']:>12} {x['delta_energy_eV']:10.4f} "
                f"{eins_s} {nearest:>10} {n['distance_A']:8.3f}"
            )

    print()
    if db.get("host_energy_eV") is not None:
        print("Einsert = E(host + mu) - E(host).")
        print("The isolated-muon reference is omitted and cancels between")
        print("materials calculated with identical model settings.")
    else:
        print("Host reference energy unavailable: insertion energies not shown.")
    print("=" * 80)


def _status(c):
    """Cheap filesystem-only progress report; no MACE calculator is loaded."""
    root = Path(c["output_dir"])
    dirs = sorted(root.glob("site_*")) if root.exists() else []
    n_total = len(dirs)
    n_relaxed = sum((d / "relaxed.extxyz").exists() for d in dirs)
    n_results = n_relaxed
    clusters = root / "clusters.json"
    final = root / "results.json"

    print()
    print("=" * 72)
    print(" MUON WORKFLOW STATUS")
    print("=" * 72)
    print(f"Host prepared         : {'yes' if (Path('host')/'host_supercell.extxyz').exists() else 'no'}")
    print(f"Generated sites       : {n_total}")
    print(f"Relaxed structures    : {n_relaxed} / {n_total}")
    print(f"Completed sites       : {n_results} / {n_total}")
    print(f"Clustering completed  : {'yes' if clusters.exists() else 'no'}")
    zsum = root / "zpe_summary.dat"
    print(f"Final results.json    : {'yes' if final.exists() else 'no'}")
    print(f"ZPE summary           : {'yes' if zsum.exists() else 'no'}")
    print("=" * 72)


def _find_results(run):
    run = Path(run)
    candidates = [run / "muon_candidates" / "results.json", run / "results.json"]
    for fn in candidates:
        if fn.exists():
            return json.loads(fn.read_text())
    for fn in run.glob("*/results.json"):
        try:
            d = json.loads(fn.read_text())
            if "sites" in d and "calculator" in d:
                return d
        except Exception:
            pass
    raise FileNotFoundError(f"No results.json found below {run}")

def main():
    ap = argparse.ArgumentParser(description="Standalone MuFinder + MACE muon-site workflow")
    ap.add_argument("input", nargs="?", help="Input file, e.g. muon.in")
    ap.add_argument("--status", action="store_true", help="Show progress without loading MACE")
    ap.add_argument("--results", action="store_true", help="Print saved results without rerunning")
    args = ap.parse_args()

    if args.input is None:
        ap.error("An input file is required.")

    c = config(args.input)
    root = Path(c["output_dir"])

    if args.status:
        _status(c)
        return

    if args.results:
        _print_results(_load_result_database(root))
        return

    print("MuFinder + MACE workflow")
    if c["calculator"].lower().replace("_","-") in ("mace-mp","mace-mp0","mp","mp0") and c.get("dispersion", False):
        print("Dispersion   : MACE native D3 defaults")
    print(c)

    calc = calculator(c)
    h = host(c, calc)
    root = sites(c, h)
    r = relax(c, calc, root)
    cs = cluster(c, r, root)

    zr = None
    if c["zpe"]:
        zr = zpes(c, calc, cs, root)

    db = _result_database(c, root, r, cs, zr)
    _print_results(db)


if __name__ == "__main__":
    main()

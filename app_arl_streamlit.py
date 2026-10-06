# -*- coding: utf-8 -*-
"""
Dashboard Streamlit - Simulasi ARL Peta Kendali Multivariat: Benchmark vs Metode Adaptif
Jalankan :  streamlit run app_arl_streamlit.py
Paket    :  streamlit plotly numpy scipy pandas joblib openpyxl
"""
import hashlib, io, itertools, json, time, warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from joblib import Parallel, delayed
from plotly.subplots import make_subplots
from scipy import stats
from scipy.linalg import solve_triangular

warnings.filterwarnings("ignore")

# =====================================================================
# 1. KONSTANTA & KATALOG METODE
# =====================================================================
BENCH = ["T2", "EWMA", "MEWMA"]
ADAPTIVE = ["MAEWMA", "AMEWMA-Hybrid", "AMEEWMA", "T2-A-MEWMA", "DR-A-MEWMA", "AMEWMA-Exp", "Adaptive-DHWMA"]
ALL = BENCH + ADAPTIVE
INVARIANT = set(ALL) - {"EWMA"}
PAL = {"T2": "#64748b", "EWMA": "#a16207", "MEWMA": "#111827", "MAEWMA": "#2563eb", "AMEWMA-Hybrid": "#7c3aed",
       "AMEEWMA": "#db2777", "T2-A-MEWMA": "#059669", "DR-A-MEWMA": "#ea580c", "AMEWMA-Exp": "#0891b2",
       "Adaptive-DHWMA": "#dc2626"}
INFO = pd.DataFrame([
    ("T2", "Benchmark", "Hotelling T², UCL = χ²(1-α, p). Peka terhadap shift besar, tanpa memori.", "Skrip R"),
    ("EWMA", "Benchmark", "EWMA univariat per variabel, L dikalibrasi bersama. Bergantung pada ρ.", "Skrip R"),
    ("MEWMA", "Benchmark", "Lowry dkk. (1992), varians eksak. Peka terhadap shift kecil.", "Skrip R"),
    ("MAEWMA", "Adaptif", "Skor Huber pada galat prediksi.", "Mahmoud & Zahran (2010)"),
    ("AMEWMA-Hybrid", "Adaptif", "Skor hibrida Huber + Tukey bisquare.", "Usulan 1"),
    ("AMEEWMA", "Adaptif", "Extended MEWMA, λ adaptif dari estimasi shift.", "Usulan 3"),
    ("T2-A-MEWMA", "Adaptif", "λₜ fungsi pangkat dari Rₜ = T²ₜ/UCL.", "Konsep 1"),
    ("DR-A-MEWMA", "Adaptif", "λₜ eksponensial dari sinyal kumulatif Cₜ (dual response).", "Konsep 2"),
    ("AMEWMA-Exp", "Adaptif", "λₜ = 1 − (1−λ₀)·exp(−γ·D²ₜ).", "Notebook AMEWMA-VSI"),
    ("Adaptive-DHWMA", "Adaptif", "Double HWMA, λₜ adaptif + standardisasi referensi in-control.", "Notebook Adaptive-DHWMA"),
], columns=["Peta", "Kelompok", "Ide singkat", "Sumber"])

DEFAULT_CFG = dict(
    alpha=0.00273, arl0=370, lam=0.1, max_rl=20000, shift_mode="semua",
    maewma_rk={2: (0.06, 3.90), 3: (0.07, 4.00), 4: (0.06, 4.28), 5: (0.06, 4.45), 6: (0.06, 4.45)},
    hyb=dict(lam=0.05, q=0.001, kmult=3.0, alpha=0.75),
    ame=dict(lmin=0.03, c=8.0, theta=0.25, psi=0.20),
    t2a=dict(lam_min=0.05, lam_max=0.50, gamma=2.0),
    dra=dict(lam_min=0.05, lam_max=0.50, rho_c=0.90, alpha=0.30, beta=0.50),
    aexp=dict(lam0=0.10, gamma=0.05),
    dhw=dict(lam_min=0.03, lam_max=0.20, k=2.0, ref_reps=20000, ref_tmax=1500),
)
H0 = {"EWMA": 3.0, "MEWMA": 10.0, "MAEWMA": 0.4, "AMEWMA-Hybrid": 0.5, "AMEEWMA": 0.5,
      "T2-A-MEWMA": 12.0, "DR-A-MEWMA": 12.0, "AMEWMA-Exp": 12.0, "Adaptive-DHWMA": 8.0}

PRESETS = {
    "⚡ Cepat (demo, ± 1–3 menit)": dict(k_p=[2, 3], k_rho=[0.0, 0.5], k_smax=3.0, k_sstep=0.5, k_reps=200, k_calib=1500,
                                         k_valid=2000, k_ssreps=300, k_ssshifts=[0.5, 1.0, 2.0], k_dhw_reps=3000, k_dhw_tmax=300),
    "⚖️ Standar": dict(k_p=[2, 3, 4], k_rho=[0.0, 0.5], k_smax=3.0, k_sstep=0.25, k_reps=500, k_calib=5000,
                       k_valid=5000, k_ssreps=1000, k_ssshifts=[0.2, 0.5, 1.0, 2.0, 3.0], k_dhw_reps=10000, k_dhw_tmax=800),
    "🔬 Penuh (sama seperti notebook)": dict(k_p=[2, 3, 4, 5], k_rho=[0.0, 0.3, 0.5, 0.7], k_smax=3.0, k_sstep=0.1, k_reps=1000,
                                              k_calib=10000, k_valid=10000, k_ssreps=2000, k_ssshifts=[0.2, 0.5, 1.0, 2.0, 3.0],
                                              k_dhw_reps=20000, k_dhw_tmax=1500),
}


# =====================================================================
# 2. INTI SIMULASI (satu fungsi langkah dipakai oleh simulasi RL & animasi)
# =====================================================================
def make_sigma(p, rho):
    return np.full((p, p), rho) + np.eye(p) * (1 - rho)


def shift_vector(p, s, mode="semua"):
    v = np.zeros(p)
    if mode == "semua":
        v[:] = s
    else:
        v[0] = s
    return v


def noncentrality(mu1, Sigma):
    return float(np.sqrt(mu1 @ np.linalg.solve(Sigma, mu1)))


def qf(A, Sinv=None):
    if Sinv is None:
        return np.einsum("ij,ij->i", A, A)
    return np.einsum("ij,jk,ik->i", A, Sinv, A)


def dhwma_lambda(score, prm):
    return prm["lam_min"] + (prm["lam_max"] - prm["lam_min"]) * np.minimum(score / prm["k"], 1.0)


def build_dhwma_reference(p, prm, seed=1001):
    reps, tmax = int(prm["ref_reps"]), int(prm["ref_tmax"])
    rng = np.random.default_rng(seed)
    D_prev = np.zeros((reps, p)); sum_X = np.zeros((reps, p)); sum_H = np.zeros((reps, p))
    centers = np.zeros((tmax, p)); inv_covs = np.zeros((tmax, p, p))
    for t in range(1, tmax + 1):
        X = rng.standard_normal((reps, p))
        mX = sum_X / (t - 1) if t > 1 else np.zeros((reps, p))
        mH = sum_H / (t - 1) if t > 1 else np.zeros((reps, p))
        lam = dhwma_lambda(np.sqrt(qf(X - D_prev)), prm)[:, None]
        H = lam * X + (1 - lam) * mX
        D = lam * H + (1 - lam) * mH
        center = np.median(D, axis=0); C = D - center
        r = np.sqrt(qf(C)); cap = np.quantile(r, 0.995)
        Cw = C * np.minimum(1.0, cap / np.maximum(r, 1e-12))[:, None]
        V = Cw.T @ Cw / (reps - 1)
        centers[t - 1] = center
        inv_covs[t - 1] = np.linalg.pinv(V + 1e-6 * np.trace(V) / p * np.eye(p))
        sum_X += X; sum_H += H; D_prev = D
    return {"center": centers, "inv_cov": inv_covs, "tmax": tmax}


def make_ctx(p, Sigma, cfg, ref=None):
    rk = cfg["maewma_rk"].get(p, cfg["maewma_rk"][max(cfg["maewma_rk"])])
    return dict(sd=np.sqrt(np.diag(Sigma)), rk=rk, ucl_t2=stats.chi2.ppf(1 - cfg["alpha"], p), ref=ref)


def new_state(reps, p):
    z = lambda: np.zeros((reps, p))
    return dict(Z=z(), D=z(), Xp=z(), sX=z(), sH=z(), c=np.zeros(reps), Cc=np.zeros(reps))


def chart_step(chart, t, X, S, p, cfg, ctx):
    """Satu langkah waktu. Mengembalikan (statistik, bobot efektif λₜ). Aturan sinyal: statistik > UCL."""
    lam = cfg["lam"]; one = np.ones(len(X))
    if chart == "T2":
        return qf(X), one
    if chart == "EWMA":
        S["Z"] = lam * X + (1 - lam) * S["Z"]
        cv = lam / (2 - lam) * (1 - (1 - lam) ** (2 * t))
        return (np.abs(S["Z"]) / (ctx["sd"] * np.sqrt(cv))).max(1), lam * one
    if chart == "MEWMA":
        S["Z"] = lam * X + (1 - lam) * S["Z"]
        return qf(S["Z"]) / (lam / (2 - lam) * (1 - (1 - lam) ** (2 * t))), lam * one
    if chart in ("MAEWMA", "AMEWMA-Hybrid"):
        R = X - S["Z"]; e = np.sqrt(qf(R))
        if chart == "MAEWMA":
            r_, k_ = ctx["rk"]
            w = np.where(e <= k_, r_, 1 - (1 - r_) * k_ / np.maximum(e, 1e-12))
        else:
            H = cfg["hyb"]; la = H["lam"]; g = np.sqrt(stats.chi2.ppf(1 - H["q"], p)); kk = H["kmult"] * g
            wH = np.where(e <= g, la, 1 - (1 - la) * g / np.maximum(e, 1e-12))
            wT = np.where(e <= kk, 1 - (1 - la) * (1 - (e / kk) ** 2) ** 2, 1.0)
            w = H["alpha"] * wH + (1 - H["alpha"]) * wT
        S["Z"] = S["Z"] + w[:, None] * R
        return qf(S["Z"]), w
    if chart == "AMEEWMA":
        a = cfg["ame"]; psi = a["psi"]
        S["D"] = psi * X + (1 - psi) * S["D"]
        dhat = np.sqrt(np.abs(qf(S["D"]) - psi * p / (2 - psi)))
        l1 = a["lmin"] + (1 - a["lmin"]) * (1 - np.exp(-dhat ** 2 / a["c"])); l2 = a["theta"] * l1
        S["Z"] = l1[:, None] * X - l2[:, None] * S["Xp"] + (1 - l1 + l2)[:, None] * S["Z"]
        S["Xp"] = X
        return qf(S["Z"]), l1
    if chart in ("T2-A-MEWMA", "DR-A-MEWMA", "AMEWMA-Exp"):
        t2 = qf(X); u = ctx["ucl_t2"]
        if chart == "T2-A-MEWMA":
            a = cfg["t2a"]; q = (t2 / u) ** a["gamma"]
            lt = a["lam_min"] + (a["lam_max"] - a["lam_min"]) * q / (1 + q)
        elif chart == "DR-A-MEWMA":
            a = cfg["dra"]; Rt = t2 / u
            S["Cc"] = a["rho_c"] * S["Cc"] + (1 - a["rho_c"]) * Rt
            St = a["alpha"] * Rt + (1 - a["alpha"]) * S["Cc"]
            lt = a["lam_min"] + (a["lam_max"] - a["lam_min"]) * (1 - np.exp(-a["beta"] * St))
        else:
            a = cfg["aexp"]; lt = 1 - (1 - a["lam0"]) * np.exp(-a["gamma"] * t2)
        S["Z"] = lt[:, None] * X + (1 - lt[:, None]) * S["Z"]
        S["c"] = lt ** 2 + (1 - lt) ** 2 * S["c"]
        return qf(S["Z"]) / S["c"], lt
    if chart == "Adaptive-DHWMA":
        a = cfg["dhw"]
        mX = S["sX"] / (t - 1) if t > 1 else 0.0
        mH = S["sH"] / (t - 1) if t > 1 else 0.0
        lt = dhwma_lambda(np.sqrt(qf(X - S["D"])), a)[:, None]
        H = lt * X + (1 - lt) * mX
        S["D"] = lt * H + (1 - lt) * mH
        ref = ctx["ref"]; i = min(t, ref["tmax"]) - 1
        stat = qf(S["D"] - ref["center"][i], ref["inv_cov"][i])
        S["sX"] += X; S["sH"] += H
        return stat, lt[:, 0]
    raise ValueError(chart)


def simulate_rl(chart, Sigma, mu1, h, cfg, ref, reps, seed, max_t=None, tau=0):
    max_t = max_t or cfg["max_rl"]
    p = Sigma.shape[0]; rng = np.random.default_rng(seed); Lc = np.linalg.cholesky(Sigma)
    if chart in INVARIANT:
        mu = np.linalg.solve(Lc, mu1); A = None
    else:
        mu = mu1; A = Lc.T
    ctx = make_ctx(p, Sigma, cfg, ref)
    rl = np.full(reps, max_t, dtype=np.int64); alive = np.ones(reps, bool); S = new_state(reps, p)
    for t in range(1, max_t + 1):
        E = rng.standard_normal((reps, p))
        X = (E @ A if A is not None else E) + (mu if t > tau else 0.0)
        stat, _ = chart_step(chart, t, X, S, p, cfg, ctx)
        sig = stat > h
        rl[sig & alive] = t
        alive &= ~sig
        if not alive.any():
            break
    return rl


def calibrate(chart, Sigma, cfg, ref, reps, seed=12345, tol=1.0, max_iter=40):
    p = Sigma.shape[0]; z = np.zeros(p); target = cfg["arl0"]
    f = lambda h: simulate_rl(chart, Sigma, z, h, cfg, ref, reps, seed, max_t=10000).mean()
    lo = hi = H0[chart]
    while f(lo) > target: lo /= 1.5
    while f(hi) < target: hi *= 1.5
    mid = hi
    for _ in range(max_iter):
        mid = (lo + hi) / 2; a = f(mid)
        if abs(a - target) < tol: break
        if a < target: lo = mid
        else: hi = mid
    return mid


def rl_summary(rl):
    return dict(ARL=rl.mean(), SDRL=rl.std(ddof=1), MRL=float(np.median(rl)), P_RL_le10=(rl <= 10).mean())


# ---- tugas paralel (fungsi top-level agar bisa di-pickle) ----
def t_ref(p, cfg, seed):
    return build_dhwma_reference(p, cfg["dhw"], seed)


def t_cal(chart, p, rho, cfg, ref, reps):
    S = np.eye(p) if chart in INVARIANT else make_sigma(p, rho)
    return calibrate(chart, S, cfg, ref, reps)


def t_valid(idx, p, rho, c, h, cfg, ref, reps):
    rl = simulate_rl(c, make_sigma(p, rho), np.zeros(p), h, cfg, ref, reps, seed=770000 + 100 * idx + ALL.index(c))
    return dict(p=p, rho=rho, chart=c, UCL=h, **rl_summary(rl))


def t_zero(idx, p, rho, c, h, cfg, ref, reps, shifts):
    S = make_sigma(p, rho); out = []
    for i, s in enumerate(shifts):
        mu1 = shift_vector(p, s, cfg["shift_mode"])
        rl = simulate_rl(c, S, mu1, h, cfg, ref, reps, seed=2026 + 100000 * idx + 100 * i + ALL.index(c))
        out.append(dict(p=p, rho=rho, shift=s, delta=noncentrality(mu1, S), chart=c, **rl_summary(rl)))
    return out


def t_steady(idx, p, rho, c, h, cfg, ref, reps, ss_shifts, tau):
    S = make_sigma(p, rho); out = []
    for i, s in enumerate(ss_shifts):
        rl = simulate_rl(c, S, shift_vector(p, s, cfg["shift_mode"]), h, cfg, ref, reps,
                         seed=900000 + 1000 * idx + 10 * i + ALL.index(c), tau=tau)
        post = rl[rl > tau] - tau
        out.append(dict(p=p, rho=rho, shift=s, chart=c, ARL_ss=post.mean() if len(post) else np.nan,
                        SDRL_ss=post.std(ddof=1) if len(post) > 1 else np.nan))
    return out


# ---- VSI ----
def _vsi_stat(chart, X, Z, c, t, cfg):
    lam = cfg["lam"]
    if chart == "MEWMA":
        Z = lam * X + (1 - lam) * Z
        return Z, c, qf(Z) / (lam / (2 - lam) * (1 - (1 - lam) ** (2 * t)))
    a = cfg["aexp"]; lt = 1 - (1 - a["lam0"]) * np.exp(-a["gamma"] * qf(X))
    Z = lt[:, None] * X + (1 - lt[:, None]) * Z
    c = lt ** 2 + (1 - lt) ** 2 * c
    return Z, c, qf(Z) / c


def vsi_warning_limit(chart, p, cfg, reps=3000, tmax=200, seed=99):
    rng = np.random.default_rng(seed); Z = np.zeros((reps, p)); c = np.zeros(reps); out = []
    for t in range(1, tmax + 1):
        Z, c, s = _vsi_stat(chart, rng.standard_normal((reps, p)), Z, c, t, cfg); out.append(s)
    return float(np.median(np.concatenate(out)))


def simulate_ts(chart, Sigma, mu1, h, cfg, vsi, wl, reps, seed, max_t):
    p = Sigma.shape[0]; rng = np.random.default_rng(seed)
    mu = np.linalg.solve(np.linalg.cholesky(Sigma), mu1)
    ts = np.full(reps, float(max_t)); alive = np.ones(reps, bool); el = np.zeros(reps)
    Z = np.zeros((reps, p)); c = np.zeros(reps)
    hn = np.full(reps, vsi["hL"] if wl is not None else 1.0)
    for t in range(1, max_t + 1):
        el[alive] += hn[alive]
        Z, c, s = _vsi_stat(chart, rng.standard_normal((reps, p)) + mu, Z, c, t, cfg)
        sig = s > h; new = sig & alive; ts[new] = el[new]; alive &= ~sig
        if not alive.any(): break
        if wl is not None: hn = np.where(s <= wl, vsi["hL"], vsi["hS"])
    return ts


def t_vsi_cal(p, chart, use_vsi, cfg, vsi, reps):
    wl = vsi_warning_limit(chart, p, cfg) if use_vsi else None
    I = np.eye(p); z = np.zeros(p); target = cfg["arl0"]
    f = lambda h: simulate_ts(chart, I, z, h, cfg, vsi, wl, reps, 12345, 10000).mean()
    lo = hi = H0[chart]
    while f(lo) > target: lo /= 1.5
    while f(hi) < target: hi *= 1.5
    mid = hi
    for _ in range(40):
        mid = (lo + hi) / 2; a = f(mid)
        if abs(a - target) < 1.0: break
        if a < target: lo = mid
        else: hi = mid
    return dict(p=p, chart=chart, VSI=use_vsi, WL=wl, UCL=mid)


def t_ats(p, rho, row, cfg, vsi, reps, shifts):
    S = make_sigma(p, rho); out = []; wl = None if pd.isna(row["WL"]) else row["WL"]
    for i, s in enumerate(shifts):
        ts = simulate_ts(row["chart"], S, shift_vector(p, s, cfg["shift_mode"]), row["UCL"], cfg, vsi, wl, reps, 55000 + 10 * i, cfg["max_rl"])
        out.append(dict(p=p, rho=rho, shift=s, chart=row["label"], ATS=ts.mean(), SDTS=ts.std(ddof=1)))
    return out


def run_parallel(func, tasks, n_jobs, bar=None, label=""):
    n = len(tasks); out = []
    if n == 0:
        return out
    def tick(i):
        if bar is not None: bar.progress((i + 1) / n, text=f"{label} ({i + 1}/{n})")
    if n_jobs == 1:
        for i, t in enumerate(tasks): out.append(func(*t)); tick(i)
        return out
    try:
        for i, r in enumerate(Parallel(n_jobs=n_jobs, return_as="generator")(delayed(func)(*t) for t in tasks)):
            out.append(r); tick(i)
        return out
    except Exception as e:  # fallback serial
        st.warning(f"Paralel gagal ({type(e).__name__}); beralih ke mode serial.")
        out = []
        for i, t in enumerate(tasks): out.append(func(*t)); tick(i)
        return out


# =====================================================================
# 3. PIPELINE UTAMA
# =====================================================================
def cfg_key(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def cache():
    return st.session_state.setdefault("cache", {})


def get_ref(p, cfg):
    k = ("ref", p, cfg_key(cfg))
    if k not in cache():
        cache()[k] = build_dhwma_reference(p, cfg["dhw"], 1001 + p)
    return cache()[k]


def get_limit(chart, p, rho, cfg, reps):
    """Batas kendali: ambil dari hasil simulasi bila konfigurasi sama, kalau tidak kalibrasi on-demand (di-cache)."""
    if chart == "T2":
        return float(stats.chi2.ppf(1 - cfg["alpha"], p))
    R = st.session_state.get("R")
    if R and cfg_key(R["cfg"]) == cfg_key(cfg) and chart in R["limits"].columns:
        m = R["limits"][(R["limits"]["p"] == p) & (np.isclose(R["limits"]["rho"], rho))]
        if len(m): return float(m.iloc[0][chart])
    k = ("lim", chart, p, rho if chart == "EWMA" else None, cfg_key(cfg), reps)
    if k not in cache():
        ref = get_ref(p, cfg) if chart == "Adaptive-DHWMA" else None
        cache()[k] = t_cal(chart, p, rho, cfg, ref, reps)
    return cache()[k]


def run_pipeline(C):
    cfg, charts, n_jobs = C["cfg"], C["charts"], C["n_jobs"]
    P, RHO = C["p_list"], C["rho_list"]
    t_start = time.time(); R = dict(cfg=cfg, C=C, charts=charts)
    with st.status("Menjalankan simulasi…", expanded=True) as status:
        refs = {p: None for p in P}
        if "Adaptive-DHWMA" in charts:
            bar = st.progress(0.0, text="Referensi DHWMA")
            out = run_parallel(t_ref, [(p, cfg, 1001 + p) for p in P], n_jobs, bar, "① Referensi in-control Adaptive-DHWMA")
            refs = dict(zip(P, out))
        R["refs"] = refs
        inv = [c for c in charts if c in INVARIANT and c != "T2"]
        tasks = [(c, p, 0.0, cfg, refs[p] if c == "Adaptive-DHWMA" else None, C["calib"]) for p in P for c in inv]
        tasks += [("EWMA", p, rho, cfg, None, C["calib"]) for p in P for rho in RHO] if "EWMA" in charts else []
        bar = st.progress(0.0, text="Kalibrasi")
        hs = run_parallel(t_cal, tasks, n_jobs, bar, f"② Kalibrasi batas kendali (ARL0 = {cfg['arl0']})")
        hmap = {(t[0], t[1], t[2] if t[0] == "EWMA" else None): h for t, h in zip(tasks, hs)}
        rows = []
        for p, rho in itertools.product(P, RHO):
            row = dict(p=p, rho=rho)
            for c in charts:
                row[c] = stats.chi2.ppf(1 - cfg["alpha"], p) if c == "T2" else hmap[(c, p, rho if c == "EWMA" else None)]
            rows.append(row)
        R["limits"] = pd.DataFrame(rows)[["p", "rho"] + charts]
        jobs = [(i, int(L["p"]), L["rho"], c, L[c]) for i, L in R["limits"].iterrows() for c in charts]
        rf = lambda p, c: refs[p] if c == "Adaptive-DHWMA" else None

        if C["run_valid"]:
            bar = st.progress(0.0, text="Validasi")
            v = pd.DataFrame(run_parallel(t_valid, [(i, p, r, c, h, cfg, rf(p, c), C["valid"]) for i, p, r, c, h in jobs], n_jobs, bar, "③ Validasi ARL0 independen"))
            v["Status"] = np.where((v["ARL"] - cfg["arl0"]).abs() <= C["tol"], "LULUS", "CEK")
            R["validation"] = v
        bar = st.progress(0.0, text="Zero-state")
        R["res"] = pd.DataFrame([r for ch in run_parallel(t_zero, [(i, p, r, c, h, cfg, rf(p, c), C["reps"], C["shifts"]) for i, p, r, c, h in jobs],
                                                          n_jobs, bar, "④ ARL zero-state") for r in ch])
        if C["run_ss"]:
            bar = st.progress(0.0, text="Steady-state")
            R["ss"] = pd.DataFrame([r for ch in run_parallel(t_steady, [(i, p, r, c, h, cfg, rf(p, c), C["ss_reps"], C["ss_shifts"], C["tau"]) for i, p, r, c, h in jobs],
                                                             n_jobs, bar, "⑤ ARL steady-state") for r in ch])
        if C["run_vsi"]:
            vs = [("MEWMA", False), ("MEWMA", True), ("AMEWMA-Exp", True)]
            bar = st.progress(0.0, text="VSI")
            lim = pd.DataFrame(run_parallel(t_vsi_cal, [(p, c, u, cfg, C["vsi"], C["calib"]) for p in P for c, u in vs], n_jobs, bar, "⑥ Kalibrasi VSI"))
            lim["label"] = lim["chart"] + np.where(lim["VSI"], "-VSI", "-FSI")
            bar = st.progress(0.0, text="ATS")
            ats = pd.DataFrame([x for ch in run_parallel(t_ats, [(p, rho, r, cfg, C["vsi"], C["ss_reps"], [0.0] + list(C["ss_shifts"]))
                                                               for p in P for rho in RHO for r in lim[lim["p"] == p].to_dict("records")],
                                                         n_jobs, bar, "⑦ ATS (VSI)") for x in ch])
            R["vsi_lim"], R["ats"] = lim, ats
        R["elapsed"] = time.time() - t_start
        status.update(label=f"Simulasi selesai dalam {R['elapsed']:.0f} detik ✅", state="complete", expanded=False)
    st.session_state["R"] = R
    st.session_state.pop("xlsx", None)


# =====================================================================
# 4. UTILITAS ANALISIS & TAMPILAN
# =====================================================================
def interp(md):
    with st.container(border=True):
        st.markdown("**💡 Interpretasi**")
        st.markdown(md)


def howto(md):
    st.caption("📖 **Cara membaca:** " + md)


def style_fig(fig, h=480, legend=True):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=50, b=10), showlegend=legend,
                      font=dict(family="Inter, Segoe UI, sans-serif", size=13), legend=dict(orientation="h", y=-0.18),
                      hoverlabel=dict(font_size=12))
    return fig


def line_style(c):
    return dict(color=PAL[c], dash="solid" if c in BENCH else "dash", width=3 if c in BENCH else 2.2)


def ratio_mewma(d, charts):
    t = d.set_index(["p", "rho", "shift", "chart"])["ARL"].unstack("chart")
    r = t.div(t["MEWMA"], axis=0).groupby(level="shift").mean()
    return r[[c for c in charts if c in r.columns]]


def rmi_table(df, val, charts):
    o = df.copy()
    o["best"] = o.groupby(["p", "rho", "shift"])[val].transform("min")
    o["rel"] = (o[val] - o["best"]) / o["best"]
    return o.pivot_table(index=["p", "rho"], columns="chart", values="rel", aggfunc="mean")[charts]


def best_chart_map(df, val):
    d = df[df["shift"] > 0]
    best = d.loc[d.groupby(["p", "rho", "shift"])[val].idxmin()]
    return best.pivot_table(index="shift", columns=["p", "rho"], values="chart", aggfunc="first"), best


def nearest(vals, target):
    vals = np.asarray(sorted(vals)); return float(vals[np.argmin(np.abs(vals - target))])


def need_results():
    if "R" not in st.session_state:
        st.info("👈 Atur konfigurasi di sidebar lalu klik **🚀 Jalankan Simulasi**. Bagian *Animasi* bisa dipakai tanpa menjalankan simulasi lengkap.")
        return True
    return False


# =====================================================================
# 5. TAB: ARL (generik zero-state & steady-state)
# =====================================================================
def arl_tab(df, val, sdc, tag, R, zero):
    d0 = df.rename(columns={val: "ARL", sdc: "SD"})
    ps, rhos, charts = sorted(d0["p"].unique()), sorted(d0["rho"].unique()), R["charts"]
    c1, c2, c3, c4 = st.columns([1, 1, 3, 1.3])
    p = c1.selectbox("Dimensi p", ps, key=f"{tag}_p"); rho = c2.selectbox("Korelasi ρ", rhos, key=f"{tag}_rho")
    sel = c3.multiselect("Peta yang ditampilkan", charts, default=charts, key=f"{tag}_ch") or charts
    ylog = c4.toggle("Sumbu Y log", True, key=f"{tag}_log")
    d = d0[(d0["p"] == p) & (d0["rho"] == rho) & d0["chart"].isin(sel)]
    shifts = sorted(d["shift"].unique())

    st.markdown("##### 1 · Kurva ARL terhadap besar shift")
    fig = go.Figure()
    for c in sel:
        dd = d[d["chart"] == c].sort_values("shift")
        fig.add_trace(go.Scatter(x=dd["shift"], y=dd["ARL"], mode="lines+markers", name=c, line=line_style(c), marker=dict(size=6),
                                 hovertemplate=f"<b>{c}</b><br>shift %{{x}}σ<br>ARL %{{y:.2f}}<extra></extra>"))
    if zero: fig.add_hline(y=R["cfg"]["arl0"], line=dict(color="#94a3b8", dash="dot"), annotation_text="ARL0 target")
    fig.update_yaxes(type="log" if ylog else "linear", title="ARL" + (" (log)" if ylog else ""))
    fig.update_xaxes(title="Besar shift per variabel (σ)")
    st.plotly_chart(style_fig(fig, 480).update_layout(title=f"ARL — p = {p}, ρ = {rho}"), use_container_width=True)
    howto("semakin ke bawah = semakin cepat mendeteksi. Garis tegas = benchmark, putus-putus = adaptif. Pada shift 0 semua peta harus mendekati ARL0.")

    pos = [s for s in shifts if s > 0]
    lines = []
    for tgt, nm in [(0.5, "kecil"), (1.0, "sedang"), (2.0, "besar")]:
        if not pos: break
        s = nearest(pos, tgt); sub = d[d["shift"] == s].dropna(subset=["ARL"])
        if sub.empty: continue
        b = sub.loc[sub["ARL"].idxmin()]; txt = f"- Shift {nm} (**{s}σ**): tercepat **{b['chart']}** dengan ARL ≈ **{b['ARL']:.1f}**"
        mw = sub[sub["chart"] == "MEWMA"]
        if len(mw) and b["chart"] != "MEWMA": txt += f", atau **{(1 - b['ARL'] / mw.iloc[0]['ARL']) * 100:.0f}% lebih cepat** daripada MEWMA ({mw.iloc[0]['ARL']:.1f})"
        lines.append(txt + ".")
    interp(f"Untuk **p = {p}, ρ = {rho}**:\n" + "\n".join(lines) + "\n\n" +
           ("Kurva zero-state mengukur kecepatan deteksi bila shift sudah ada sejak awal. Peta T² tanpa memori biasanya unggul pada shift besar tetapi tertinggal pada shift kecil; "
            "peta berbasis EWMA/adaptif menutup celah itu dengan mengakumulasi informasi." if zero else
            "Steady-state mengukur kecepatan deteksi bila shift terjadi setelah proses berjalan lama, sehingga peta berbasis memori tidak diuntungkan oleh nilai awal nol (tidak ada *head-start*)."))

    st.markdown("##### 2 · Heatmap ARL (peta × shift)")
    pv = d.pivot_table(index="chart", columns="shift", values="ARL").reindex(sel)
    fig = go.Figure(go.Heatmap(z=np.log10(pv.values), x=[str(x) for x in pv.columns], y=pv.index, colorscale="RdYlGn_r",
                               text=np.round(pv.values, 1), customdata=pv.values, colorbar=dict(title="log10 ARL"),
                               texttemplate="%{text}" if len(pv.columns) <= 14 else None, textfont=dict(size=10),
                               hovertemplate="%{y}<br>shift %{x}σ<br>ARL %{customdata:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed"); fig.update_xaxes(title="Shift (σ)")
    st.plotly_chart(style_fig(fig, 80 + 42 * len(sel), False), use_container_width=True)
    howto("hijau = deteksi cepat (ARL kecil), merah = lambat. Bandingkan warna antarbaris pada kolom shift yang sama.")

    cA, cB = st.columns(2)
    with cA:
        st.markdown("##### 3 · Peringkat pada satu shift")
        s_sel = st.select_slider("Pilih shift", options=pos or shifts, value=nearest(pos or shifts, 1.0), key=f"{tag}_s")
        sub = d[d["shift"] == s_sel].sort_values("ARL", ascending=False)
        fig = go.Figure(go.Bar(x=sub["ARL"], y=sub["chart"], orientation="h", marker_color=[PAL[c] for c in sub["chart"]],
                               text=sub["ARL"].round(1), textposition="outside"))
        fig.update_xaxes(title="ARL"); st.plotly_chart(style_fig(fig, 420, False), use_container_width=True)
        howto("batang terpendek = peta tercepat pada shift terpilih.")
    with cB:
        st.markdown("##### 4 · Rasio ARL terhadap MEWMA")
        if "MEWMA" in charts:
            dd = d0[d0["chart"].isin(sel + ["MEWMA"]) & (d0["shift"] > 0)]
            if st.checkbox("Rata-ratakan seluruh p & ρ (bukan hanya yang dipilih)", True, key=f"{tag}_allr") is False:
                dd = dd[(dd["p"] == p) & (dd["rho"] == rho)]
            r = ratio_mewma(dd, sel); fig = go.Figure()
            for c in r.columns:
                fig.add_trace(go.Scatter(x=r.index, y=r[c], mode="lines+markers", name=c, line=line_style(c)))
            fig.add_hline(y=1, line=dict(color="black", width=1)); fig.update_yaxes(type="log", title="ARL / ARL MEWMA (log)")
            fig.update_xaxes(title="Shift (σ)"); st.plotly_chart(style_fig(fig, 420), use_container_width=True)
            howto("di bawah garis 1 = lebih cepat daripada MEWMA; di atas 1 = lebih lambat.")
        else:
            st.info("Pilih peta MEWMA pada konfigurasi untuk melihat rasio.")

    with st.expander("🧩 Small multiples: semua kombinasi p × ρ"):
        fig = make_subplots(rows=len(ps), cols=len(rhos), shared_xaxes=True, shared_yaxes=True,
                            subplot_titles=[f"p={a}, ρ={b}" for a in ps for b in rhos], vertical_spacing=0.07)
        for i, a in enumerate(ps):
            for j, b in enumerate(rhos):
                for c in sel:
                    dd = d0[(d0["p"] == a) & (d0["rho"] == b) & (d0["chart"] == c)].sort_values("shift")
                    fig.add_trace(go.Scatter(x=dd["shift"], y=dd["ARL"], name=c, legendgroup=c, showlegend=(i == 0 and j == 0),
                                             line=line_style(c)), row=i + 1, col=j + 1)
        fig.update_yaxes(type="log"); st.plotly_chart(style_fig(fig, 260 * len(ps) + 60), use_container_width=True)
    with st.expander("🔀 Pengaruh korelasi ρ dan dimensi p"):
        s2 = st.select_slider("Shift yang dianalisis", options=pos or shifts, value=nearest(pos or shifts, 1.0), key=f"{tag}_s2")
        cA, cB = st.columns(2)
        for col, (xv, fixed, ttl) in zip((cA, cB), [("rho", ("p", p), f"ARL vs ρ (p = {p})"), ("p", ("rho", rho), f"ARL vs p (ρ = {rho})")]):
            dd = d0[(d0["shift"] == s2) & (d0[fixed[0]] == fixed[1]) & d0["chart"].isin(sel)]; fig = go.Figure()
            for c in sel:
                x = dd[dd["chart"] == c].sort_values(xv); fig.add_trace(go.Scatter(x=x[xv], y=x["ARL"], mode="lines+markers", name=c, line=line_style(c)))
            fig.update_yaxes(type="log", title="ARL"); fig.update_xaxes(title=xv, type="category"); col.plotly_chart(style_fig(fig, 380).update_layout(title=ttl), use_container_width=True)
        interp("Semua peta **kecuali EWMA univariat** bersifat *directionally invariant*: ARL-nya hanya bergantung pada δ = √(μ₁'Σ⁻¹μ₁). "
               "Jika shift diberikan pada semua variabel, δ membesar saat ρ naik sehingga kurva bergeser; pada ρ = 0 EWMA univariat sangat kompetitif, "
               "tetapi ia kehilangan keunggulan saat variabel berkorelasi. Semakin besar p, deteksi pada shift tetap umumnya makin cepat karena δ ikut membesar.")


# =====================================================================
# 6. TAB: KALIBRASI & VALIDASI
# =====================================================================
def tab_calib(R):
    lim, charts = R["limits"], R["charts"]
    st.markdown("##### Batas kendali hasil kalibrasi")
    st.dataframe(lim.round(4), use_container_width=True, hide_index=True)
    howto("setiap peta punya batas kendali (UCL) sendiri yang dicari agar ARL0 = target. Nilai tiap peta tidak bisa dibandingkan langsung karena statistiknya berbeda skala.")
    ps = sorted(lim["p"].unique()); cs = [c for c in charts if c != "EWMA"]
    fig = make_subplots(rows=1, cols=len(cs), subplot_titles=cs, shared_yaxes=False)
    for j, c in enumerate(cs):
        d = lim[lim["rho"] == lim["rho"].iloc[0]]
        fig.add_trace(go.Scatter(x=d["p"], y=d[c], mode="lines+markers", line=dict(color=PAL[c], width=3), showlegend=False), row=1, col=j + 1)
    fig.update_xaxes(type="category", title="p"); st.plotly_chart(style_fig(fig, 300, False), use_container_width=True)
    howto("UCL non-EWMA dihitung sekali per p (berlaku untuk semua ρ). Kurva naik menandakan statistik makin besar ketika dimensi bertambah.")
    if "EWMA" in charts:
        fig = go.Figure()
        for p in ps:
            d = lim[lim["p"] == p]; fig.add_trace(go.Scatter(x=d["rho"], y=d["EWMA"], mode="lines+markers", name=f"p = {p}"))
        fig.update_xaxes(title="ρ"); fig.update_yaxes(title="L (EWMA)")
        st.plotly_chart(style_fig(fig, 320).update_layout(title="Faktor L EWMA univariat per (p, ρ)"), use_container_width=True)
        howto("EWMA univariat dikalibrasi per (p, ρ) karena tidak invarian terhadap korelasi.")
    if "validation" not in R:
        st.info("Validasi ARL0 tidak dijalankan (aktifkan di sidebar → Tahapan)."); return
    v = R["validation"]; arl0, tol = R["cfg"]["arl0"], R["C"]["tol"]
    st.markdown("##### Validasi ARL0 independen")
    fig = go.Figure(); fig.add_hrect(y0=arl0 - tol, y1=arl0 + tol, fillcolor="#22c55e", opacity=0.15, line_width=0, annotation_text=f"Toleransi {arl0 - tol}–{arl0 + tol}")
    for c in charts:
        d = v[v["chart"] == c]
        fig.add_trace(go.Box(y=d["ARL"], name=c, marker_color=PAL[c], boxpoints="all", jitter=0.4, pointpos=0, showlegend=False,
                             text=[f"p={a}, ρ={b}" for a, b in zip(d["p"], d["rho"])]))
    fig.add_hline(y=arl0, line=dict(color="black", dash="dot")); fig.update_yaxes(title="ARL0 hasil validasi")
    st.plotly_chart(style_fig(fig, 460, False), use_container_width=True)
    howto("setiap titik = satu kombinasi (p, ρ). Titik di dalam pita hijau berarti ARL0 sesuai target.")
    cA, cB = st.columns(2)
    with cA:
        pv = v.pivot_table(index=["p", "rho"], columns="chart", values="P_RL_le10")[charts]
        fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=[f"p={a}, ρ={b}" for a, b in pv.index], colorscale="Blues", texttemplate="%{z:.3f}", textfont=dict(size=9)))
        fig.update_yaxes(autorange="reversed"); st.plotly_chart(style_fig(fig, 440, False).update_layout(title="P(RL ≤ 10) — alarm palsu dini"), use_container_width=True)
    with cB:
        v2 = v.assign(Status=v["Status"].map({"LULUS": "✅ LULUS", "CEK": "⚠️ CEK"}))
        st.dataframe(v2[["p", "rho", "chart", "UCL", "ARL", "SDRL", "MRL", "P_RL_le10", "Status"]].round(4), height=440, use_container_width=True, hide_index=True)
    n_ok = int((v["Status"] == "LULUS").sum()); bad = v[v["Status"] == "CEK"]
    worst = v.loc[(v["ARL"] - arl0).abs().idxmax()]
    interp(f"**{n_ok} dari {len(v)}** kombinasi (peta × p × ρ) lulus toleransi ±{tol} terhadap ARL0 = {arl0}. "
           + (f"Yang perlu dicek: **{', '.join(sorted(bad['chart'].unique()))}**. " if len(bad) else "Semua peta terkalibrasi dengan baik. ")
           + f"Deviasi terbesar: **{worst['chart']}** (p={int(worst['p'])}, ρ={worst['rho']}) dengan ARL0 = {worst['ARL']:.1f}. "
           f"Untuk run length geometrik, P(RL ≤ 10) idealnya ≈ 2,7%; nilai jauh lebih besar menandakan alarm palsu dini (bentuk distribusi RL berbeda antar peta walau ARL0 sama).")


# =====================================================================
# 7. TAB: PERBANDINGAN & PERINGKAT
# =====================================================================
def tab_rank(R):
    charts, res, ss = R["charts"], R["res"], R.get("ss")
    rz = rmi_table(res[res["shift"] > 0], "ARL", charts)
    out = pd.DataFrame({"RMI_ZeroState": rz.mean()})
    if ss is not None: rs = rmi_table(ss, "ARL_ss", charts); out["RMI_SteadyState"] = rs.mean()
    out["Rata_rata"] = out.mean(axis=1)
    for col in list(out.columns): out[f"Rank_{col}"] = out[col].rank(method="min").astype(int)
    out = out.sort_values("Rata_rata"); R["ringkasan"] = out
    st.markdown("##### RMI — Relative Mean Index")
    fig = go.Figure()
    for col, color in [("RMI_ZeroState", "#2563eb"), ("RMI_SteadyState", "#f97316")]:
        if col in out: fig.add_trace(go.Bar(y=out.index, x=out[col].clip(upper=3), orientation="h", name=col.replace("RMI_", ""), marker_color=color,
                                            text=out[col].round(2), textposition="outside"))
    fig.update_layout(barmode="group"); fig.update_yaxes(autorange="reversed"); fig.update_xaxes(title="RMI (dipotong di 3; kecil = baik)")
    st.plotly_chart(style_fig(fig, 150 + 48 * len(charts)), use_container_width=True)
    howto("RMI = rata-rata selisih relatif ARL terhadap peta terbaik pada tiap shift > 0. Nilai 0 = selalu terbaik.")
    b = out.index[0]; w = out.index[-1]
    interp(f"Berdasarkan rata-rata RMI, peta paling konsisten adalah **{b}** (RMI = {out.loc[b, 'Rata_rata']:.3f}) dan yang paling tertinggal **{w}** ({out.loc[w, 'Rata_rata']:.3f}). "
           f"Peta terbaik di kelompok adaptif: **{out.loc[[c for c in out.index if c in ADAPTIVE]].index[0] if any(c in ADAPTIVE for c in out.index) else '-'}**; "
           f"di kelompok benchmark: **{out.loc[[c for c in out.index if c in BENCH]].index[0] if any(c in BENCH for c in out.index) else '-'}**. "
           "RMI kecil berarti peta jarang kalah jauh pada rentang shift, sehingga cocok bila besar shift tidak diketahui.")
    cA, cB = st.columns(2)
    with cA:
        st.markdown("##### Jumlah kemenangan (ARL terkecil)")
        _, bz = best_chart_map(res, "ARL"); win = pd.DataFrame({"ZeroState": bz["chart"].value_counts()})
        if ss is not None: win["SteadyState"] = best_chart_map(ss, "ARL_ss")[1]["chart"].value_counts()
        win = win.reindex(charts).fillna(0).astype(int); fig = go.Figure()
        for col in win.columns: fig.add_trace(go.Bar(x=win.index, y=win[col], name=col, text=win[col], textposition="outside"))
        fig.update_layout(barmode="group"); st.plotly_chart(style_fig(fig, 400), use_container_width=True)
        howto("tinggi batang = berapa kali peta menjadi tercepat pada kombinasi (p, ρ, shift).")
    with cB:
        st.markdown("##### RMI per kondisi (p, ρ)")
        which = st.radio("Skenario", ["Zero-state"] + (["Steady-state"] if ss is not None else []), horizontal=True)
        tb = rz if which == "Zero-state" else rs
        fig = go.Figure(go.Heatmap(z=tb.values.clip(max=3), x=tb.columns, y=[f"p={a}, ρ={b}" for a, b in tb.index], colorscale="YlGnBu_r",
                                   texttemplate="%{z:.2f}", textfont=dict(size=9))); fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig, 400, False), use_container_width=True)
        howto("kolom paling gelap/rendah = peta konsisten di berbagai kondisi.")
    st.markdown("##### 🗺️ Peta metode terbaik per shift")
    src = st.radio("Sumber", ["Zero-state"] + (["Steady-state"] if ss is not None else []), horizontal=True, key="bm_src")
    pm, _ = best_chart_map(res, "ARL") if src == "Zero-state" else best_chart_map(ss, "ARL_ss")
    code = {c: i for i, c in enumerate(charts)}; z = pm.map(lambda x: code.get(x, np.nan)).values.astype(float)
    n = len(charts); cs = []
    for i, c in enumerate(charts): cs += [[i / n, PAL[c]], [(i + 1) / n, PAL[c]]]
    fig = go.Figure(go.Heatmap(z=z, x=[f"p={a},ρ={b}" for a, b in pm.columns], y=[str(s) for s in pm.index], text=pm.values, texttemplate="%{text}",
                               textfont=dict(size=8, color="white"), colorscale=cs, zmin=-0.5, zmax=n - 0.5, showscale=False,
                               hovertemplate="%{x}<br>shift %{y}σ<br>%{text}<extra></extra>"))
    fig.update_yaxes(title="Shift (σ)", autorange="reversed"); st.plotly_chart(style_fig(fig, 120 + 22 * len(pm), False), use_container_width=True)
    howto("setiap sel diwarnai menurut peta tercepat. Warna yang berubah dari atas ke bawah menunjukkan peta mana unggul pada shift kecil vs besar.")
    st.dataframe(out.round(3), use_container_width=True)


# =====================================================================
# 8. TAB: DISTRIBUSI RUN LENGTH
# =====================================================================
def tab_rl(R):
    cfg, charts = R["cfg"], R["charts"]
    c1, c2, c3, c4 = st.columns(4)
    p = c1.selectbox("p", R["C"]["p_list"], key="rl_p"); rho = c2.selectbox("ρ", R["C"]["rho_list"], key="rl_rho")
    s = c3.slider("Besar shift (σ)", 0.0, 3.0, 1.0, 0.1, key="rl_s"); reps = c4.select_slider("Replikasi", [1000, 2000, 5000, 10000], 2000, key="rl_reps")
    sel = st.multiselect("Peta dibandingkan", charts, default=[c for c in ["T2", "MEWMA", "MAEWMA", "AMEWMA-Exp"] if c in charts] or charts[:3], key="rl_sel")
    if st.button("▶️ Simulasikan distribusi run length", key="rl_go") and sel:
        S = make_sigma(p, rho); mu1 = shift_vector(p, s, cfg["shift_mode"]); data = {}
        with st.spinner("Menyimulasikan…"):
            for c in sel:
                h = get_limit(c, p, rho, cfg, R["C"]["calib"]); ref = R["refs"].get(p) if c == "Adaptive-DHWMA" else None
                data[c] = simulate_rl(c, S, mu1, h, cfg, ref, reps, seed=4242 + ALL.index(c), max_t=min(cfg["max_rl"], 10000))
        st.session_state["rl_data"] = (data, p, rho, s)
    if "rl_data" not in st.session_state: return
    data, p, rho, s = st.session_state["rl_data"]
    cA, cB = st.columns(2)
    fig = go.Figure()
    for c, rl in data.items(): fig.add_trace(go.Violin(y=rl, name=c, line_color=PAL[c], box_visible=True, meanline_visible=True, points=False, spanmode="hard"))
    fig.update_yaxes(type="log", title="Run length (log)"); cA.plotly_chart(style_fig(fig, 430, False).update_layout(title="Sebaran run length"), use_container_width=True)
    fig = go.Figure()
    for c, rl in data.items():
        x = np.sort(rl); fig.add_trace(go.Scatter(x=x, y=np.arange(1, len(x) + 1) / len(x), name=c, line=dict(color=PAL[c], width=2.5)))
    fig.update_xaxes(type="log", title="Run length (log)"); fig.update_yaxes(title="P(RL ≤ x)")
    cB.plotly_chart(style_fig(fig, 430).update_layout(title="ECDF run length"), use_container_width=True)
    howto("violin: lebar = kepadatan, kotak = kuartil, garis = rata-rata (ARL). ECDF: kurva yang lebih cepat naik ke 1 berarti peta lebih cepat dan konsisten mendeteksi.")
    tb = pd.DataFrame({c: dict(ARL=rl.mean(), SDRL=rl.std(ddof=1), P10=np.percentile(rl, 10), Median=np.median(rl), P90=np.percentile(rl, 90), P99=np.percentile(rl, 99)) for c, rl in data.items()}).T
    st.dataframe(tb.round(2), use_container_width=True)
    b = tb["ARL"].idxmin(); bm = tb["Median"].idxmin()
    interp(f"Pada shift **{s}σ** (p={p}, ρ={rho}) peta dengan ARL terkecil adalah **{b}** ({tb.loc[b, 'ARL']:.1f}) dan median terkecil **{bm}** ({tb.loc[bm, 'Median']:.1f}). "
           "Run length bersifat miring ke kanan: rata-rata (ARL) > median. Perhatikan juga **P90/P99** — peta dengan ARL mirip bisa berbeda dalam risiko deteksi sangat terlambat. "
           + ("Pada shift 0 ini adalah distribusi in-control: ekor kanan panjang normal, dan P10 menunjukkan seberapa sering terjadi alarm palsu dini." if s == 0 else ""))


# =====================================================================
# 9. TAB: ANIMASI
# =====================================================================
def run_animation(P, cfg, calib):
    p, rho, s, tau, T, charts = P["p"], P["rho"], P["s"], P["tau"], P["T"], P["charts"]
    S = make_sigma(p, rho); Lc = np.linalg.cholesky(S); mu1 = shift_vector(p, s, cfg["shift_mode"])
    rng = np.random.default_rng(P["seed"])
    mu_t = np.where((np.arange(1, T + 1) > tau)[:, None], mu1[None, :], 0.0)
    X = rng.standard_normal((T, p)) @ Lc.T + mu_t
    Xs = solve_triangular(Lc, X.T, lower=True).T
    A = dict(P, X=X, mu_t=mu_t, stat={}, lam={}, h={}, alarm={}, Sigma=S, Lc=Lc)
    for c in charts:
        h = get_limit(c, p, rho, cfg, calib); ref = get_ref(p, cfg) if c == "Adaptive-DHWMA" else None
        ctx = make_ctx(p, S, cfg, ref); st_ = new_state(1, p); data = Xs if c in INVARIANT else X; sv, lv = np.zeros(T), np.zeros(T)
        for t in range(1, T + 1):
            a, b = chart_step(c, t, data[t - 1:t], st_, p, cfg, ctx); sv[t - 1], lv[t - 1] = a[0], b[0]
        A["stat"][c], A["lam"][c], A["h"][c] = sv, lv, h
        ex = np.where(sv > h)[0]; A["alarm"][c] = int(ex[0]) + 1 if len(ex) else None
        post = np.where((sv > h) & (np.arange(1, T + 1) > tau))[0]; A["detect"] = A.get("detect", {}); A["detect"][c] = int(post[0]) + 1 if len(post) else None
    return A


def build_anim_fig(A, cfg):
    T, tau, charts, p = A["T"], A["tau"], A["charts"], A["p"]; X, mu_t = A["X"], A["mu_t"]; norm, show_lam = A["norm"], A["show_lam"]
    nC = len(charts); rows = 1 + nC + (1 if show_lam else 0); trail = A["trail"]; tt = np.arange(1, T + 1)
    uline = {c: 1.0 if norm else A["h"][c] for c in charts}; y = {c: A["stat"][c] / (A["h"][c] if norm else 1.0) for c in charts}
    ymax = {c: max(1.35 * uline[c], min(1.08 * y[c].max(), 3.2 * uline[c])) for c in charts}
    titles = ["① Data proses vs rata-rata sebenarnya (garis putus-putus)", "② Ruang observasi X₁–X₂"] + \
             [f"{c}: statistik{' / UCL' if norm else ''}" for c in charts] + (["λₜ efektif (bobot adaptif)"] if show_lam else [])
    fig = make_subplots(rows=rows, cols=2, column_widths=[0.64, 0.36], horizontal_spacing=0.07, vertical_spacing=0.06 if rows > 4 else 0.08,
                        specs=[[{}, {"rowspan": rows}]] + [[{}, None] for _ in range(rows - 1)], subplot_titles=titles)
    gray = ["#94a3b8", "#64748b", "#475569", "#334155", "#1e293b", "#0f172a"]

    def traces_at(t):
        L = []; i0 = max(0, t - trail)
        for j in range(p): L.append((go.Scatter(x=tt[:t], y=X[:t, j], mode="lines", line=dict(color=gray[j % 6], width=1.3), name=f"X{j + 1}", showlegend=False), 1, 1))
        L.append((go.Scatter(x=tt[:t], y=mu_t[:t, 0], mode="lines", line=dict(color="#f97316", width=3, dash="dash"), name="μ sebenarnya (X₁)", showlegend=False), 1, 1))
        for k, c in enumerate(charts):
            r = 2 + k; col = PAL[c]; yy = y[c]; ta = A["alarm"][c]; over = np.where(yy[:t] > uline[c])[0]
            L.append((go.Scatter(x=tt[:t], y=yy[:t], mode="lines", line=dict(color=col, width=2.4), showlegend=False, name=c), r, 1))
            L.append((go.Scatter(x=tt[over], y=yy[over], mode="markers", marker=dict(color="#e11d48", size=7), showlegend=False, name="> UCL"), r, 1))
            L.append((go.Scatter(x=[t], y=[min(yy[t - 1], ymax[c])], mode="markers", marker=dict(size=13, color="#e11d48" if yy[t - 1] > uline[c] else col, line=dict(color="white", width=2)),
                                 showlegend=False, name="sekarang"), r, 1))
            on = ta is not None and t >= ta
            L.append((go.Scatter(x=[ta] if on else [], y=[min(yy[ta - 1], ymax[c])] if on else [], mode="markers+text", marker=dict(symbol="x", size=18, color="#e11d48", line=dict(width=3)),
                                 text=["🚨 ALARM PALSU" if ta <= tau else "🚨 ALARM"] if on else [], textposition="top right", textfont=dict(color="#e11d48", size=13), showlegend=False), r, 1))
            L.append((go.Scatter(x=[ta, t, t, ta, ta] if on else [], y=[0, 0, ymax[c], ymax[c], 0] if on else [], fill="toself", mode="none", fillcolor="rgba(225,29,72,0.10)", showlegend=False, hoverinfo="skip"), r, 1))
        if show_lam:
            for c in charts: L.append((go.Scatter(x=tt[:t], y=A["lam"][c][:t], mode="lines", line=dict(color=PAL[c], width=2), name=c, showlegend=False), rows, 1))
        age = np.linspace(0.12, 0.85, t - i0)
        L.append((go.Scatter(x=X[i0:t, 0], y=X[i0:t, 1], mode="lines+markers", line=dict(color="rgba(37,99,235,0.25)", width=1),
                             marker=dict(size=8, color=[f"rgba(37,99,235,{a:.2f})" for a in age]), showlegend=False, name="jejak"), 1, 2))
        any_alarm = any(A["alarm"][c] is not None and A["alarm"][c] <= t for c in charts)
        L.append((go.Scatter(x=[X[t - 1, 0]], y=[X[t - 1, 1]], mode="markers", marker=dict(symbol="diamond", size=17, color="#e11d48" if any_alarm else ("#f97316" if t > tau else "#2563eb"),
                                                                                               line=dict(color="white", width=2)), showlegend=False, name="observasi"), 1, 2))
        L.append((go.Scatter(x=[mu_t[t - 1, 0]], y=[mu_t[t - 1, 1]], mode="markers", marker=dict(symbol="star", size=20, color="black"), showlegend=False, name="pusat proses"), 1, 2))
        L.append((go.Scatter(x=[0, mu_t[t - 1, 0]], y=[0, mu_t[t - 1, 1]], mode="lines", line=dict(color="black", width=2, dash="dot"), showlegend=False, hoverinfo="skip"), 1, 2))
        return L

    def title_at(t):
        n = sum(1 for c in charts if A["alarm"][c] is not None and A["alarm"][c] <= t)
        stt = "<span style='color:#16a34a'>IN-CONTROL</span>" if t <= tau else "<span style='color:#f97316'>OUT-OF-CONTROL (shift aktif)</span>"
        al = f"<span style='color:#e11d48'><b>{n}/{len(charts)} peta sudah alarm</b></span>" if n else "belum ada alarm"
        return dict(text=f"<b>t = {t}</b> &nbsp;│&nbsp; Proses: {stt} &nbsp;│&nbsp; {al}", x=0.0, font=dict(size=16))

    ts = list(range(1, T + 1, A["stride"])); ts += [T] if ts[-1] != T else []
    init = traces_at(ts[0]); n_dyn = len(init)
    for tr, r, c in init: fig.add_trace(tr, row=r, col=c)
    th = np.linspace(0, 2 * np.pi, 200); L2 = np.linalg.cholesky(A["Sigma"][:2, :2]); cr = stats.chi2.ppf(1 - cfg["alpha"], 2)
    ell = np.sqrt(cr) * (L2 @ np.vstack([np.cos(th), np.sin(th)]))
    fig.add_trace(go.Scatter(x=ell[0], y=ell[1], mode="lines", fill="toself", fillcolor="rgba(34,197,94,0.08)", line=dict(color="#16a34a", dash="dot"), name="Batas in-control (3σ)", hoverinfo="skip"), row=1, col=2)
    lim = 1.15 * max(np.abs(ell).max(), np.abs(X[:, :2]).max(), np.abs(mu_t[:, :2]).max())
    fig.update_xaxes(range=[-lim, lim], row=1, col=2, title="X₁", zeroline=True); fig.update_yaxes(range=[-lim, lim], row=1, col=2, title="X₂", scaleanchor="x2", scaleratio=1, zeroline=True)
    for r in range(1, rows + 1):
        fig.update_xaxes(range=[0, T + 1], row=r, col=1)
        if tau < T: fig.add_vrect(x0=tau + 0.5, x1=T + 1, fillcolor="#f97316", opacity=0.07, line_width=0, row=r, col=1)
        fig.add_vline(x=tau + 0.5, line=dict(color="#f97316", dash="dash", width=1.5), row=r, col=1)
    for k, c in enumerate(charts):
        fig.add_hline(y=uline[c], line=dict(color="#e11d48", dash="dash", width=1.8), row=2 + k, col=1, annotation_text="UCL", annotation_position="top left", annotation_font_color="#e11d48")
        fig.update_yaxes(range=[0, ymax[c]], row=2 + k, col=1)
    fig.update_xaxes(title="Waktu (nomor sampel)", row=rows, col=1)
    sp = A["speed"]
    frames = [go.Frame(data=[tr for tr, _, _ in traces_at(t)], traces=list(range(n_dyn)), name=str(t), layout=go.Layout(title=title_at(t))) for t in ts]
    fig.frames = frames
    steps = [dict(method="animate", label=str(t) if (t % 10 == 0 or t == T) else "", args=[[str(t)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))]) for t in ts]
    fig.update_layout(
        template="plotly_white", height=max(680, 215 * rows + 120), title=title_at(ts[0]), margin=dict(l=10, r=10, t=80, b=110), showlegend=False,
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.03, xanchor="left", yanchor="top", pad=dict(t=6), buttons=[
            dict(label="▶ Putar", method="animate", args=[None, dict(frame=dict(duration=sp, redraw=True), fromcurrent=True, transition=dict(duration=0), mode="immediate")]),
            dict(label="⏸ Jeda", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))]),
            dict(label="⏮ Ulang", method="animate", args=[[str(ts[0])], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))])])],
        sliders=[dict(active=0, x=0.17, len=0.83, y=-0.03, pad=dict(t=6), currentvalue=dict(prefix="t = ", font=dict(size=13)), steps=steps)])
    return fig


def tab_anim(cfg_now):
    R = st.session_state.get("R"); cfg = R["cfg"] if R else cfg_now
    calib = R["C"]["calib"] if R else min(st.session_state.get("k_calib", 1500), 3000)
    st.markdown("Simulasikan **satu proses nyata** dengan data yang sama untuk semua peta, lalu putar animasinya untuk melihat *kapan pergeseran terjadi* dan *peta mana yang membunyikan alarm lebih dulu*.")
    if R is None: st.caption("ℹ️ Belum ada hasil simulasi: batas kendali akan dikalibrasi cepat saat animasi dibuat (hasilnya disimpan di cache sesi).")
    else: st.caption("ℹ️ Memakai konfigurasi & batas kendali dari simulasi terakhir.")
    pl = R["C"]["p_list"] if R else st.session_state["k_p"]; rl_ = R["C"]["rho_list"] if R else st.session_state["k_rho"]
    avail = R["charts"] if R else ALL
    with st.form("anim_form"):
        a1, a2, a3, a4 = st.columns(4)
        p = a1.selectbox("Dimensi p", pl); rho = a2.selectbox("Korelasi ρ", rl_)
        s = a3.slider("Besar shift (σ per variabel)", 0.0, 3.0, 1.0, 0.1); seed = a4.number_input("Seed data", 0, 10**6, 7)
        b1, b2, b3, b4 = st.columns(4)
        tau = b1.number_input("Shift terjadi setelah sampel ke-", 0, 400, 30); T = b2.number_input("Panjang pengamatan T", 40, 400, 120, 10)
        speed = b3.slider("Kecepatan (ms/frame)", 30, 600, 120, 10); trail = b4.slider("Panjang jejak (ruang observasi)", 5, 60, 20)
        charts = st.multiselect("Peta yang dibandingkan (maks. 5)", avail, default=[c for c in ["T2", "MEWMA", "MAEWMA", "AMEWMA-Exp"] if c in avail], max_selections=5)
        d1, d2 = st.columns(2)
        norm = d1.checkbox("Normalisasi statistik terhadap UCL (UCL = 1)", True); show_lam = d2.checkbox("Tampilkan bobot adaptif λₜ", True)
        go_btn = st.form_submit_button("🎬 Buat animasi", type="primary", use_container_width=True)
    if go_btn:
        if not charts: st.warning("Pilih minimal satu peta.")
        else:
            with st.spinner("Mengkalibrasi batas kendali & mensimulasikan proses…"):
                P = dict(p=p, rho=rho, s=s, tau=int(tau), T=int(T), seed=int(seed), charts=charts, norm=norm, show_lam=show_lam, speed=speed,
                         trail=trail, stride=max(1, int(np.ceil(T / 130))))
                A = run_animation(P, cfg, calib)
            with st.spinner("Menyusun frame animasi…"):
                st.session_state["anim"] = (A, build_anim_fig(A, cfg), cfg)
    if "anim" not in st.session_state: return
    A, fig, _ = st.session_state["anim"]
    st.plotly_chart(fig, use_container_width=True, key="anim_chart")
    howto("tekan **▶ Putar**. Panel ① menunjukkan data dan rata-rata sebenarnya yang melompat saat shift; panel peta menunjukkan statistik naik menuju UCL (garis merah putus-putus), titik merah = di atas UCL, "
          "❌ + area merah = alarm pertama. Panel kanan menunjukkan titik observasi dan pusat proses (★) berpindah keluar dari elips in-control.")
    rows = []
    for c in A["charts"]:
        ta, td = A["alarm"][c], A["detect"][c]
        rows.append(dict(Peta=c, UCL=round(A["h"][c], 3), **{"Alarm pertama (t)": ta, "Jenis": "—" if ta is None else ("Alarm palsu" if ta <= A["tau"] else "Deteksi benar"),
                                                              "Deteksi setelah shift (t)": td, "Delay (sampel)": None if td is None else td - A["tau"],
                                                              "Titik di atas UCL": int((A["stat"][c] > A["h"][c]).sum())}))
    tb = pd.DataFrame(rows); st.dataframe(tb, use_container_width=True, hide_index=True)
    ok = tb.dropna(subset=["Delay (sampel)"])
    if A["s"] == 0: msg = "Shift = 0σ: proses tetap in-control sepanjang pengamatan. Setiap alarm yang muncul adalah **alarm palsu**; dengan ARL0 = %d, rata-rata satu alarm palsu per %d sampel." % (cfg["arl0"], cfg["arl0"])
    elif ok.empty: msg = "Tidak ada peta yang mendeteksi shift dalam rentang pengamatan; perpanjang T atau naikkan besar shift."
    else:
        f = ok.loc[ok["Delay (sampel)"].idxmin()]; sl = ok.loc[ok["Delay (sampel)"].idxmax()]
        miss = [c for c in A["charts"] if A["detect"][c] is None]; fa = tb[tb["Jenis"] == "Alarm palsu"]["Peta"].tolist()
        msg = (f"Shift **{A['s']}σ** mulai pada t = {A['tau'] + 1}. Peta tercepat: **{f['Peta']}** (alarm di t = {int(f['Deteksi setelah shift (t)'])}, delay **{int(f['Delay (sampel)'])}** sampel)"
               + (f"; terlambat: **{sl['Peta']}** (delay {int(sl['Delay (sampel)'])})" if len(ok) > 1 else "") + ". "
               + (f"Belum mendeteksi: **{', '.join(miss)}**. " if miss else "") + (f"⚠️ Alarm palsu sebelum shift terjadi pada: **{', '.join(fa)}**. " if fa else "")
               + "Satu realisasi hanyalah ilustrasi; ganti *seed* untuk melihat variasi. Perbandingan yang andal ada pada ARL (tab zero-state/steady-state).")
    interp(msg + "\n\nPeta adaptif cenderung menaikkan λₜ (panel λₜ) ketika data menjauh dari target sehingga lebih responsif terhadap shift besar, dan menurunkannya saat proses stabil sehingga tetap sensitif terhadap shift kecil.")


# =====================================================================
# 10. TAB: VSI & UNDUH
# =====================================================================
def tab_vsi(R):
    if "ats" not in R:
        st.info("Aktifkan **Bagian VSI (ATS)** di sidebar → Tahapan untuk menjalankan analisis ini."); return
    vsi, ats, lim = R["C"]["vsi"], R["ats"], R["vsi_lim"]
    st.caption(f"Interval sampling: h_L = {vsi['hL']}, h_S = {vsi['hS']}. WL = median statistik in-control sehingga E(h) ≈ 1. FSI = interval tetap 1.")
    c1, c2 = st.columns(2); p = c1.selectbox("p", R["C"]["p_list"], key="v_p"); rho = c2.selectbox("ρ", R["C"]["rho_list"], key="v_rho")
    d = ats[(ats["p"] == p) & (ats["rho"] == rho)]; fig = go.Figure()
    pal = {"MEWMA-FSI": "#111827", "MEWMA-VSI": "#2563eb", "AMEWMA-Exp-VSI": "#dc2626"}
    for lb, dd in d.groupby("chart"): dd = dd.sort_values("shift"); fig.add_trace(go.Scatter(x=dd["shift"], y=dd["ATS"], mode="lines+markers", name=lb, line=dict(color=pal.get(lb), width=3)))
    fig.update_yaxes(type="log", title="ATS (log)"); fig.update_xaxes(title="Shift (σ)")
    st.plotly_chart(style_fig(fig, 440).update_layout(title=f"ATS — p = {p}, ρ = {rho}"), use_container_width=True)
    howto("ATS = rata-rata waktu hingga sinyal (satuan waktu, bukan jumlah sampel). Garis di bawah = deteksi lebih cepat.")
    pv = d.pivot_table(index="shift", columns="chart", values="ATS"); sh = [s for s in pv.index if s > 0]
    if sh and {"MEWMA-FSI", "MEWMA-VSI", "AMEWMA-Exp-VSI"} <= set(pv.columns):
        s = nearest(sh, 1.0); r = pv.loc[s]
        interp(f"Pada shift **{s}σ**: ATS MEWMA-FSI = {r['MEWMA-FSI']:.2f}, MEWMA-VSI = {r['MEWMA-VSI']:.2f} (**{(1 - r['MEWMA-VSI'] / r['MEWMA-FSI']) * 100:.0f}% lebih cepat** berkat VSI), "
               f"dan AMEWMA-Exp-VSI = {r['AMEWMA-Exp-VSI']:.2f} ({(1 - r['AMEWMA-Exp-VSI'] / r['MEWMA-VSI']) * 100:.0f}% dibanding MEWMA-VSI, efek adaptasi λ). "
               "Membandingkan tiga garis ini memisahkan efek *VSI* (FSI→VSI) dari efek *adaptasi λ* (MEWMA-VSI→AMEWMA-Exp-VSI).")
    st.dataframe(lim.round(4), use_container_width=True, hide_index=True)
    st.dataframe(pv.round(2), use_container_width=True)


def build_excel(R):
    from openpyxl.styles import Alignment, PatternFill
    cfg, charts, bio = R["cfg"], R["charts"], io.BytesIO()
    param = pd.DataFrame([(k, json.dumps(v, default=str)) for k, v in cfg.items()], columns=["Parameter", "Nilai"])
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        param.to_excel(w, sheet_name="Parameter", index=False); R["limits"].round(5).to_excel(w, sheet_name="Batas_Kendali", index=False)
        if "validation" in R: R["validation"].round(4).to_excel(w, sheet_name="Validasi_ARL0", index=False)
        R["res"].round(4).to_excel(w, sheet_name="ZeroState", index=False)
        if "ss" in R: R["ss"].round(4).to_excel(w, sheet_name="SteadyState", index=False)
        for p in sorted(R["res"]["p"].unique()):
            R["res"][R["res"]["p"] == p].pivot_table(index="shift", columns=["rho", "chart"], values="ARL").round(2).to_excel(w, sheet_name=f"ARL_p{p}")
        if "ringkasan" in R: R["ringkasan"].round(4).to_excel(w, sheet_name="Ringkasan_RMI")
        pz, _ = best_chart_map(R["res"], "ARL"); pz.to_excel(w, sheet_name="Peta_Terbaik_ZS")
        if "ss" in R: best_chart_map(R["ss"], "ARL_ss")[0].to_excel(w, sheet_name="Peta_Terbaik_SS")
        if "ats" in R: R["vsi_lim"].round(5).to_excel(w, sheet_name="VSI_Batas", index=False); R["ats"].round(4).to_excel(w, sheet_name="VSI_ATS", index=False)
        hexes = {c: PAL[c].lstrip("#").upper() for c in charts}
        for sh in [n for n in ("Peta_Terbaik_ZS", "Peta_Terbaik_SS") if n in w.book.sheetnames]:
            for row in w.book[sh].iter_rows():
                for cell in row:
                    cell.alignment = Alignment(horizontal="center")
                    if cell.value in hexes: cell.fill = PatternFill("solid", fgColor=hexes[cell.value])
    return bio.getvalue()


def tab_download(R):
    st.markdown("Unduh seluruh hasil (parameter, batas kendali, validasi, ARL, RMI, peta terbaik, VSI) dalam satu berkas Excel.")
    if st.button("📦 Siapkan file Excel"):
        with st.spinner("Menyusun Excel…"): st.session_state["xlsx"] = build_excel(R)
    if "xlsx" in st.session_state:
        st.download_button("⬇️ Unduh hasil_ARL_gabungan_adaptif.xlsx", st.session_state["xlsx"], "hasil_ARL_gabungan_adaptif.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ CSV zero-state", R["res"].to_csv(index=False).encode(), "zero_state.csv", "text/csv")
    if "ss" in R: c2.download_button("⬇️ CSV steady-state", R["ss"].to_csv(index=False).encode(), "steady_state.csv", "text/csv")


# =====================================================================
# 11. SIDEBAR
# =====================================================================
def apply_preset():
    for k, v in PRESETS[st.session_state["preset"]].items(): st.session_state[k] = v


def sidebar():
    if "k_p" not in st.session_state:
        st.session_state["preset"] = list(PRESETS)[0]; apply_preset()
    D = DEFAULT_CFG; sb = st.sidebar
    sb.markdown("## ⚙️ Konfigurasi")
    sb.selectbox("Preset beban komputasi", list(PRESETS), key="preset", on_change=apply_preset, help="Mengatur p, ρ, jumlah replikasi, dsb. sekaligus. Nilai tetap bisa diubah manual.")
    with sb.expander("🎯 Umum", expanded=True):
        arl0 = st.number_input("ARL0 target", 50, 5000, D["arl0"], 10)
        alpha = st.number_input("α (untuk UCL T² & ambang adaptif)", 0.0001, 0.05, D["alpha"], 0.0001, format="%.5f")
        st.multiselect("Dimensi p", [2, 3, 4, 5, 6], key="k_p"); st.multiselect("Korelasi ρ (equicorrelation)", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9], key="k_rho")
        mode = st.radio("Pola shift", ["semua", "satu"], horizontal=True, format_func=lambda x: "Semua variabel" if x == "semua" else "Satu variabel (X₁)")
        c1, c2 = st.columns(2); c1.number_input("Shift maksimum (σ)", 0.5, 5.0, step=0.5, key="k_smax"); c2.number_input("Langkah shift", 0.05, 1.0, step=0.05, key="k_sstep")
        lam = st.slider("λ EWMA/MEWMA", 0.02, 0.5, D["lam"], 0.01)
        charts = st.multiselect("Peta kendali", ALL, default=ALL)
    with sb.expander("🧪 Parameter metode adaptif"):
        st.caption("MAEWMA (r, k) per p")
        rk_df = st.data_editor(pd.DataFrame({"p": list(D["maewma_rk"]), "r": [v[0] for v in D["maewma_rk"].values()], "k": [v[1] for v in D["maewma_rk"].values()]}),
                               hide_index=True, disabled=["p"], use_container_width=True, key="rk_editor")
        st.caption("AMEWMA-Hybrid"); a, b = st.columns(2)
        hyb = dict(lam=a.number_input("λ", 0.01, 0.5, D["hyb"]["lam"], 0.01, key="h1"), q=b.number_input("q", 0.0001, 0.05, D["hyb"]["q"], 0.0005, format="%.4f", key="h2"),
                   kmult=a.number_input("k-mult", 1.0, 6.0, D["hyb"]["kmult"], 0.5, key="h3"), alpha=b.number_input("α Huber", 0.0, 1.0, D["hyb"]["alpha"], 0.05, key="h4"))
        st.caption("AMEEWMA"); a, b = st.columns(2)
        ame = dict(lmin=a.number_input("λ min", 0.01, 0.5, D["ame"]["lmin"], 0.01, key="m1"), c=b.number_input("c", 1.0, 30.0, D["ame"]["c"], 1.0, key="m2"),
                   theta=a.number_input("θ", 0.0, 1.0, D["ame"]["theta"], 0.05, key="m3"), psi=b.number_input("ψ", 0.05, 0.9, D["ame"]["psi"], 0.05, key="m4"))
        st.caption("T2-A-MEWMA (Konsep 1)"); a, b, c = st.columns(3)
        t2a = dict(lam_min=a.number_input("λ min", 0.01, 0.5, D["t2a"]["lam_min"], 0.01, key="t1"), lam_max=b.number_input("λ max", 0.1, 1.0, D["t2a"]["lam_max"], 0.05, key="t2"),
                   gamma=c.number_input("γ", 0.5, 6.0, D["t2a"]["gamma"], 0.5, key="t3"))
        st.caption("DR-A-MEWMA (Konsep 2)"); a, b = st.columns(2)
        dra = dict(lam_min=a.number_input("λ min", 0.01, 0.5, D["dra"]["lam_min"], 0.01, key="d1"), lam_max=b.number_input("λ max", 0.1, 1.0, D["dra"]["lam_max"], 0.05, key="d2"),
                   rho_c=a.number_input("ρc", 0.0, 0.99, D["dra"]["rho_c"], 0.05, key="d3"), alpha=b.number_input("α", 0.0, 1.0, D["dra"]["alpha"], 0.05, key="d4"),
                   beta=a.number_input("β", 0.05, 5.0, D["dra"]["beta"], 0.05, key="d5"))
        st.caption("AMEWMA-Exp"); a, b = st.columns(2)
        aexp = dict(lam0=a.number_input("λ₀", 0.01, 0.5, D["aexp"]["lam0"], 0.01, key="e1"), gamma=b.number_input("γ", 0.005, 1.0, D["aexp"]["gamma"], 0.005, format="%.3f", key="e2"))
        st.caption("Adaptive-DHWMA"); a, b = st.columns(2)
        dhw = dict(lam_min=a.number_input("λ min", 0.01, 0.5, D["dhw"]["lam_min"], 0.01, key="w1"), lam_max=b.number_input("λ max", 0.05, 1.0, D["dhw"]["lam_max"], 0.05, key="w2"),
                   k=a.number_input("k", 0.5, 6.0, D["dhw"]["k"], 0.5, key="w3"), ref_reps=int(b.number_input("Replikasi referensi", 500, 50000, step=500, key="k_dhw_reps")),
                   ref_tmax=int(a.number_input("t maks referensi", 100, 3000, step=100, key="k_dhw_tmax")))
    with sb.expander("🔁 Replikasi & tahapan"):
        reps = st.number_input("Replikasi per titik shift (zero-state)", 50, 20000, step=50, key="k_reps")
        calib = st.number_input("Replikasi kalibrasi UCL", 500, 50000, step=500, key="k_calib")
        run_valid = st.checkbox("Validasi ARL0 independen", True); valid = st.number_input("Replikasi validasi", 500, 50000, step=500, key="k_valid"); tol = st.number_input("Toleransi ARL0 (±)", 1, 50, 10)
        run_ss = st.checkbox("ARL steady-state", True); tau = st.number_input("τ (shift setelah sampel ke-)", 1, 500, 50)
        ss_shifts = st.multiselect("Shift untuk steady-state/VSI", [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0], key="k_ssshifts"); ss_reps = st.number_input("Replikasi steady-state / ATS", 100, 20000, step=100, key="k_ssreps")
        run_vsi = st.checkbox("Bagian VSI (ATS)", False); a, b = st.columns(2); hL = a.number_input("h_L", 0.5, 5.0, 1.9, 0.1); hS = b.number_input("h_S", 0.01, 1.0, 0.1, 0.05)
        max_rl = st.number_input("Batas maksimum RL", 2000, 100000, D["max_rl"], 1000); n_jobs = st.number_input("N_JOBS (−1 = semua core)", -1, 64, -1)
    cfg = dict(alpha=alpha, arl0=int(arl0), lam=lam, max_rl=int(max_rl), shift_mode=mode, maewma_rk={int(r.p): (float(r.r), float(r.k)) for r in rk_df.itertuples()},
               hyb=hyb, ame=ame, t2a=t2a, dra=dra, aexp=aexp, dhw=dhw)
    shifts = np.round(np.arange(0, st.session_state["k_smax"] + 1e-9, st.session_state["k_sstep"]), 2)
    charts = [c for c in ALL if c in charts]
    C = dict(cfg=cfg, charts=charts, p_list=sorted(st.session_state["k_p"]), rho_list=sorted(st.session_state["k_rho"]), shifts=list(shifts), reps=int(reps), calib=int(calib),
             valid=int(valid), tol=tol, tau=int(tau), ss_shifts=sorted(ss_shifts), ss_reps=int(ss_reps), run_valid=run_valid, run_ss=run_ss and bool(ss_shifts),
             run_vsi=run_vsi and bool(ss_shifts), vsi=dict(hL=hL, hS=hS), n_jobs=int(n_jobs))
    load = len(charts) * len(C["p_list"]) * len(C["rho_list"]) * len(shifts) * reps
    sb.caption(f"Perkiraan beban zero-state: **{load / 1e6:.1f} juta** replikasi-peta." + (" ⚠️ Berat — pertimbangkan preset lebih ringan." if load > 5e7 else ""))
    go_run = sb.button("🚀 Jalankan Simulasi", type="primary", use_container_width=True)
    if "R" in st.session_state and sb.button("🗑️ Hapus hasil & cache", use_container_width=True):
        for k in ("R", "cache", "anim", "rl_data", "xlsx"): st.session_state.pop(k, None)
        st.rerun()
    return C, go_run


# =====================================================================
# 12. MAIN
# =====================================================================
CSS = """
<style>
.block-container{padding-top:1.2rem;max-width:1500px}
.hero{background:linear-gradient(120deg,#0f172a 0%,#1d4ed8 55%,#7c3aed 100%);color:#fff;padding:26px 32px;border-radius:18px;margin-bottom:14px}
.hero h1{margin:0;font-size:1.9rem;color:#fff} .hero p{margin:6px 0 0;opacity:.85;font-size:1rem}
div[data-testid="stMetric"]{background:rgba(148,163,184,.12);border:1px solid rgba(148,163,184,.25);padding:12px 16px;border-radius:14px}
button[data-baseweb="tab"]{font-weight:600}
</style>"""


def main():
    st.set_page_config(page_title="Simulasi ARL Peta Kendali Multivariat", page_icon="📊", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>📊 Simulasi ARL Peta Kendali Multivariat</h1>'
                '<p>Benchmark (T², EWMA, MEWMA) vs metode adaptif — kalibrasi, validasi ARL0, zero-state, steady-state, RMI, distribusi RL, VSI, dan animasi peta kendali.</p></div>', unsafe_allow_html=True)
    C, go_run = sidebar()
    if go_run:
        if not C["charts"] or not C["p_list"] or not C["rho_list"]: st.error("Pilih minimal satu peta, satu p, dan satu ρ.")
        else:
            st.session_state.pop("anim", None); run_pipeline(C)
    R = st.session_state.get("R")
    tabs = st.tabs(["🏠 Ringkasan", "🎯 Kalibrasi & Validasi", "📈 ARL Zero-State", "⏱️ ARL Steady-State", "🏆 Peringkat", "📊 Distribusi RL", "🎬 Animasi", "🕒 VSI (ATS)", "💾 Unduh"])
    with tabs[0]:
        m = st.columns(5); m[0].metric("Jumlah peta", len(C["charts"])); m[1].metric("Dimensi p", ", ".join(map(str, C["p_list"])) or "-"); m[2].metric("Korelasi ρ", ", ".join(map(str, C["rho_list"])) or "-")
        m[3].metric("ARL0 target", C["cfg"]["arl0"]); m[4].metric("Titik shift", len(C["shifts"]))
        st.markdown("#### Katalog metode"); st.dataframe(INFO, use_container_width=True, hide_index=True)
        interp("Peta **benchmark** dipakai sebagai pembanding, sedangkan peta **adaptif** menyesuaikan bobot pemulusan (λₜ) menurut besarnya penyimpangan. Idealnya peta adaptif cepat pada shift besar *dan* tetap sensitif pada shift kecil. "
               "Semua peta dikalibrasi ke ARL0 yang sama sehingga perbandingan ARL setelah shift adil.")
        with st.expander("📐 Rumus ringkas"):
            st.markdown("Notasi: $X_t$ observasi, $\\mu_0=0$, $\\lVert v\\rVert^2_{\\Sigma}=v'\\Sigma^{-1}v$.")
            st.latex(r"\textbf{MAEWMA: } e_t=\lVert X_t-Z_{t-1}\rVert_\Sigma,\ \omega(e)=\begin{cases}r & e\le k\\ 1-(1-r)k/e & e>k\end{cases},\ Z_t=Z_{t-1}+\omega(e_t)(X_t-Z_{t-1})")
            st.latex(r"\textbf{Hybrid: } \omega=\alpha\,\omega_{Huber}+(1-\alpha)\,\omega_{Tukey}")
            st.latex(r"\textbf{AMEEWMA: } \hat\delta_t=\sqrt{|\lVert d_t\rVert^2_\Sigma-\psi p/(2-\psi)|},\ \lambda_{1t}=\lambda_{min}+(1-\lambda_{min})(1-e^{-\hat\delta_t^2/c})")
            st.latex(r"\textbf{T2-A-MEWMA: } R_t=T^2_t/\chi^2_{1-\alpha,p},\ \lambda_t=\lambda_{min}+(\lambda_{max}-\lambda_{min})\frac{R_t^\gamma}{1+R_t^\gamma}")
            st.latex(r"\textbf{DR-A-MEWMA: } C_t=\rho_cC_{t-1}+(1-\rho_c)R_t,\ S_t=\alpha R_t+(1-\alpha)C_t,\ \lambda_t=\lambda_{min}+(\lambda_{max}-\lambda_{min})(1-e^{-\beta S_t})")
            st.latex(r"\textbf{AMEWMA-Exp: } \lambda_t=1-(1-\lambda_0)e^{-\gamma D_t^2},\quad Z_t=\lambda_tX_t+(1-\lambda_t)Z_{t-1},\ A_t=\lVert Z_t\rVert^2_\Sigma/c_t")
            st.latex(r"\textbf{Adaptive-DHWMA: } \lambda_t=\lambda_{min}+(\lambda_{max}-\lambda_{min})\min(\lVert X_t-D_{t-1}\rVert_\Sigma/k,1)")
        with st.expander("ℹ️ Catatan teknis"):
            st.markdown("- Semua peta kecuali EWMA univariat *directionally invariant*: disimulasikan di ruang terstandar $\\tilde X=L^{-1}X$, sehingga UCL dikalibrasi sekali per $p$ dan berlaku untuk semua ρ. EWMA dikalibrasi per (p, ρ).\n"
                        "- Statistik AMEWMA-Exp dinormalisasi dengan $c_t$ agar konsisten dengan Konsep 1 & 2.\n- Aturan sinyal semua peta: statistik > UCL.\n- Simulasi berjalan paralel (`joblib`); hasil disimpan di sesi, jadi mengganti tab tidak mengulang komputasi.")
        if R:
            st.success(f"Hasil tersedia: {len(R['charts'])} peta, p = {R['C']['p_list']}, ρ = {R['C']['rho_list']} (selesai {R['elapsed']:.0f} detik).")
            quick = rmi_table(R["res"][R["res"]["shift"] > 0], "ARL", R["charts"]).mean().sort_values()
            interp(f"Ringkasan cepat (RMI zero-state): peta paling konsisten adalah **{quick.index[0]}**, diikuti **{quick.index[1] if len(quick) > 1 else '-'}**. Lihat tab *Peringkat* untuk detail.")
        else: st.info("Mulai dengan memilih preset di sidebar, lalu klik **🚀 Jalankan Simulasi**.")
    with tabs[1]:
        if not need_results(): tab_calib(R)
    with tabs[2]:
        if not need_results(): arl_tab(R["res"], "ARL", "SDRL", "zs", R, True)
    with tabs[3]:
        if not need_results():
            if "ss" in R: arl_tab(R["ss"], "ARL_ss", "SDRL_ss", "ss", R, False)
            else: st.info("ARL steady-state tidak dijalankan (aktifkan di sidebar → Tahapan).")
    with tabs[4]:
        if not need_results(): tab_rank(R)
    with tabs[5]:
        if not need_results(): tab_rl(R)
    with tabs[6]: tab_anim(C["cfg"])
    with tabs[7]:
        if not need_results(): tab_vsi(R)
    with tabs[8]:
        if not need_results(): tab_download(R)


if __name__ == "__main__":
    main()

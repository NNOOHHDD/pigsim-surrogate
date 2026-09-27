"""Coupled solver: upstream flow + downstream flow + pig dynamics.

Per time step the unknowns shared by the three sub-problems are the pig
velocity ``Vp`` and the by-pass volumetric flow rate ``Qh``.  They are found by
a Newton iteration on the two coupling residuals

    R1 = m (Vp^{n+1} - Vp^n)/dt - [(p1 - p2) A - m g sin(a) - Fc]      (eq. 4)
    R2 = (p1 - p2) - (rho K/2) (Qh/Ah)^2                               (eq. 5)

where ``p1`` and ``p2`` come from solving the two flow domains with the pig
face velocities  V|up = Vp + Qh/A_up  and  V|down = Vp + Qh/A_down.
A sealing pig (Ah = 0) drops ``Qh`` and R2; a pig held by static friction
drops ``Vp`` and R1.
"""

import numpy as np

from .domain import Domain
from .pig import G


class PigFlowSolver:
    def __init__(self, route, pig, fluid_up, fluid_down=None,
                 N_up=80, N_down=80, cluster=1.5,
                 bc_inlet=None, bc_outlet=None,
                 scheme_p="central", scheme_h="upwind", scheme_V="central",
                 picard_iter=6, picard_tol=1e-6, picard_relax=1.0, T_iso=None):
        self.route = route
        self.pig = pig
        self.fluid_up = fluid_up
        self.fluid_down = fluid_down if fluid_down is not None else fluid_up
        self.bc_inlet = bc_inlet
        self.bc_outlet = bc_outlet
        self.picard = dict(n_iter=picard_iter, tol=picard_tol, relax=picard_relax)

        self.T_iso = T_iso
        kw = dict(cluster=cluster, scheme_p=scheme_p, scheme_h=scheme_h,
                  scheme_V=scheme_V, T_iso=T_iso)
        self.up = Domain(self.fluid_up, N_up, route, "up", **kw)
        self.dn = Domain(self.fluid_down, N_down, route, "down", **kw)

        self.t = 0.0
        self.Qh = 0.0
        self.hist = {k: [] for k in
                     ("t", "s_pig", "V_pig", "p1", "p2", "Qh", "dt", "stopped")}
        self.snapshots = []

    # ------------------------------------------------------------------
    def initialise(self, s_pig, p_of_s, T_up_of_s, T_dn_of_s=None, V_pig=0.0):
        self.pig.s = float(s_pig)
        self.pig.V = float(V_pig)
        self.pig.stopped = (V_pig == 0.0)
        self.up.set_mesh(self.pig.s, 0.0)
        self.dn.set_mesh(self.pig.s, 0.0)
        self.up.init_uniform(p_of_s, T_up_of_s)
        self.dn.init_uniform(p_of_s, T_dn_of_s or T_up_of_s)
        self.Qh = 0.0

    # ------------------------------------------------------------------
    # state bookkeeping
    # ------------------------------------------------------------------
    def _snapshot_state(self):
        return (self.up.p.copy(), self.up.h.copy(), self.up.V.copy(),
                self.dn.p.copy(), self.dn.h.copy(), self.dn.V.copy())

    def _restore_state(self, st):
        (self.up.p, self.up.h, self.up.V,
         self.dn.p, self.dn.h, self.dn.V) = [a.copy() for a in st]

    def _resolve_bc(self, spec, domain, face):
        """Translate a user boundary spec into the form Domain.solve_step wants."""
        out = {}
        kind = spec["type"]
        ic = 0 if face == "left" else domain.N - 1
        iface = 0 if face == "left" else domain.N
        if "T" in spec:
            out["h"] = float(domain.fluid.h_of_T(spec["T"]))
        elif "h" in spec:
            out["h"] = float(spec["h"])
        if kind == "p":
            out["type"] = "p"
            out["p"] = float(spec["p"])
        elif kind == "mdot":
            st = domain.fluid.props(domain.p[ic], out.get("h", domain.h[ic]))
            out["type"] = "V"
            out["V"] = float(spec["mdot"]) / (float(st.rho) * domain.A_f[iface])
        elif kind == "V":
            out["type"] = "V"
            out["V"] = float(spec["V"])
        elif kind == "valve":
            out["type"] = "valve"
            out.update(CdA=spec["CdA"], chi=spec["chi"], p_res=spec["p_res"])
        else:
            raise ValueError(f"unknown bc type {kind!r}")
        return out

    # ------------------------------------------------------------------
    # flow solve for a trial (Vp, Qh)
    # ------------------------------------------------------------------
    def _flow_solve(self, Vp, Qh, dt, s_old, old, t_new):
        s_new = float(np.clip(s_old + 0.5 * (self.pig.V + Vp) * dt,
                              1e-6, self.route.L - 1e-6))
        self.up.set_mesh(s_new, Vp)
        self.dn.set_mesh(s_new, Vp)

        A_up = self.up.A_f[-1]
        A_dn = self.dn.A_f[0]

        bcL = self._resolve_bc(self.bc_inlet(t_new), self.up, "left")
        bcR = {"type": "V", "V": Vp + Qh / A_up}
        self.up.solve_step(dt, old["pu"], old["hu"], old["Vu"], bcL, bcR,
                           **self.picard)

        # The enthalpy carried through the by-pass is the upstream enthalpy
        # (paper, "Fluid properties" paragraph).  A sealing pig is adiabatic and
        # passes no mass, so no enthalpy boundary value is imposed at all -- and
        # when the two domains hold different fluids the match must be made on
        # temperature, not on enthalpy, since the two cp's differ.
        bcL2 = {"type": "V", "V": Vp + Qh / A_dn}
        if not self.pig.sealing and Qh > 0.0:
            T_up = float(self.up.fluid.T_of_h(self.up.h[-1]))
            bcL2["h"] = float(self.dn.fluid.h_of_T(T_up))
        bcR2 = self._resolve_bc(self.bc_outlet(t_new), self.dn, "right")
        self.dn.solve_step(dt, old["pd"], old["hd"], old["Vd"], bcL2, bcR2,
                           **self.picard)

        return float(self.up.p_pig_face()), float(self.dn.p_pig_face()), s_new

    # ------------------------------------------------------------------
    def step(self, dt, newton_max=10, tol=1e-3):
        t_new = self.t + dt
        s_old = self.pig.s
        Vp_old = self.pig.V
        old = dict(pu=self.up.p.copy(), hu=self.up.h.copy(), Vu=self.up.V.copy(),
                   pd=self.dn.p.copy(), hd=self.dn.h.copy(), Vd=self.dn.V.copy())
        base = self._snapshot_state()

        sealing = self.pig.sealing
        m = self.pig.mass
        # direction used for the dynamic contact force during this step
        dir_ref = 1.0 if Vp_old >= 0.0 else -1.0

        for attempt in (0, 1):
            moving = not self.pig.stopped
            free = ([] if not moving else ["V"]) + ([] if sealing else ["Q"])
            z0 = []
            if "V" in free:
                z0.append(Vp_old)
            if "Q" in free:
                z0.append(self.Qh)
            z = np.array(z0, dtype=float)

            def unpack(zv):
                i = 0
                Vp = 0.0
                if "V" in free:
                    Vp = zv[i]
                    i += 1
                Qh = 0.0 if sealing else zv[i]
                return Vp, Qh

            def resid(zv):
                Vp, Qh = unpack(zv)
                self._restore_state(base)
                p1, p2, s_new = self._flow_solve(Vp, Qh, dt, s_old, old, t_new)
                A = float(self.up.A_f[-1])
                sin_a = float(self.route.sin_alpha(s_new))
                driving = (p1 - p2) * A - m * G * sin_a
                R = []
                if "V" in free:
                    Fc = self.pig.contact_force(s_new, dir_ref, driving)
                    R.append(m * (Vp - Vp_old) / dt - (driving - Fc))
                if "Q" in free:
                    st = self.up.fluid.props(self.up.p[-1], self.up.h[-1])
                    R.append((p1 - p2) - self.pig.dp_bypass(float(st.rho), A, Qh))
                return np.array(R), (p1, p2, s_new)

            if len(free) == 0:
                self._restore_state(base)
                p1, p2, s_new = self._flow_solve(0.0, 0.0, dt, s_old, old, t_new)
                Vp, Qh = 0.0, 0.0
            else:
                r, extra = resid(z)
                for _ in range(newton_max):
                    J = np.zeros((len(z), len(z)))
                    for j in range(len(z)):
                        dz = max(1e-3 * abs(z[j]), 1e-4 if free[j] == "V" else 1e-6)
                        zp = z.copy()
                        zp[j] += dz
                        rp, _ = resid(zp)
                        J[:, j] = (rp - r) / dz
                    try:
                        step_z = np.linalg.solve(J, -r)
                    except np.linalg.LinAlgError:
                        break
                    scale = np.array([max(abs(z[j]), 1e-2 if free[j] == "V" else 1e-4)
                                      for j in range(len(z))])
                    step_z = np.clip(step_z, -5 * scale, 5 * scale)
                    z = z + step_z
                    r, extra = resid(z)
                    if np.all(np.abs(step_z) < tol * scale):
                        break
                p1, p2, s_new = extra
                Vp, Qh = unpack(z)

            A = float(self.up.A_f[-1])
            sin_a = float(self.route.sin_alpha(s_new))
            driving = (p1 - p2) * A - m * G * sin_a

            if self.pig.stopped and self.pig.can_break_free(s_new, driving) \
                    and attempt == 0:
                self.pig.stopped = False       # break free, redo the step
                self._restore_state(base)
                continue

            if moving and Vp_old != 0.0 and Vp * Vp_old < 0.0 and attempt == 0:
                self.pig.stopped = True        # came to rest, redo the step
                self._restore_state(base)
                continue
            break

        self.pig.V = float(Vp)
        self.pig.s = float(s_new)
        self.Qh = float(Qh)
        self.t = t_new

        for k, v in (("t", self.t), ("s_pig", self.pig.s), ("V_pig", self.pig.V),
                     ("p1", p1), ("p2", p2), ("Qh", self.Qh), ("dt", dt),
                     ("stopped", self.pig.stopped)):
            self.hist[k].append(v)
        return p1, p2

    # ------------------------------------------------------------------
    def _save(self):
        return (self._snapshot_state(), self.pig.s, self.pig.V,
                self.pig.stopped, self.Qh, self.t,
                {k: len(v) for k, v in self.hist.items()})

    def _load(self, saved):
        state, s, V, stopped, Qh, t, nrec = saved
        self._restore_state(state)
        self.pig.s, self.pig.V, self.pig.stopped = s, V, stopped
        self.Qh, self.t = Qh, t
        for k, n in nrec.items():
            del self.hist[k][n:]
        self.up.set_mesh(s, V)
        self.dn.set_mesh(s, V)

    def run(self, t_end, dt0=0.05, dt_max=2.0, dt_min=1e-5,
            cfl_pig=0.02, dV_max=5.0, snapshot_times=(), verbose=True,
            s_stop=None, progress_every=200, max_reject=10):
        """March in time with an adaptive, rejecting time step.

        A step is rejected (and retried with dt/4) if the flow solve fails or
        if the pig velocity changes by more than ``dV_max``; this is what keeps
        the break-away from static friction -- where the pig can see several
        hundred m/s^2 -- from destroying the solution.
        """
        dt = dt0
        snap = sorted(snapshot_times)
        s_stop = s_stop if s_stop is not None else self.route.L * 0.999
        self._record_snapshot()
        self._record_stations()
        n = 0
        while self.t < t_end and self.pig.s < s_stop:
            dt = min(dt, t_end - self.t)
            if snap and self.t < snap[0] < self.t + dt:
                dt = snap[0] - self.t
            V_before = self.pig.V
            saved = self._save()

            for _ in range(max_reject):
                try:
                    self.step(dt)
                    ok = (np.all(np.isfinite(self.up.p)) and np.all(np.isfinite(self.dn.p))
                          and np.isfinite(self.pig.V)
                          and abs(self.pig.V - V_before) <= max(dV_max, 0.5 * abs(V_before)))
                except (FloatingPointError, np.linalg.LinAlgError):
                    ok = False
                if ok or dt <= dt_min * 1.001:
                    break
                self._load(saved)
                dt = max(dt * 0.25, dt_min)

            n += 1
            self._record_stations()
            if snap and self.t >= snap[0] - 1e-9:
                self._record_snapshot()
                snap.pop(0)

            L1 = max(self.pig.s, 1e-3)
            L2 = max(self.route.L - self.pig.s, 1e-3)
            v = max(abs(self.pig.V), 1e-6)
            dt_geo = cfl_pig * min(L1, L2) / v
            dV = abs(self.pig.V - V_before)
            acc = dV / dt
            dt_acc = 0.5 * dV_max / max(acc, 1e-9)
            dt = float(np.clip(min(dt_geo, dt_acc, dt * 1.3, dt_max), dt_min, dt_max))
            if verbose and n % progress_every == 0:
                print(f"  t={self.t:9.2f}s  s={self.pig.s:10.2f}m  "
                      f"Vp={self.pig.V:8.3f}m/s  Qh={self.Qh:7.4f}m3/s  dt={dt:8.5f}s"
                      f"{'  [stopped]' if self.pig.stopped else ''}")
        self._record_snapshot()
        return self.hist

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def _record_snapshot(self):
        s = np.concatenate([self.up.sc, self.dn.sc])
        p = np.concatenate([self.up.p, self.dn.p])
        T = np.concatenate([self.up.fluid.T_of_h(self.up.h),
                            self.dn.fluid.T_of_h(self.dn.h)])
        Vc_u = 0.5 * (self.up.V[:-1] + self.up.V[1:])
        Vc_d = 0.5 * (self.dn.V[:-1] + self.dn.V[1:])
        V = np.concatenate([Vc_u, Vc_d])
        rho = np.concatenate([self.up.fluid.props(self.up.p, self.up.h).rho,
                              self.dn.fluid.props(self.dn.p, self.dn.h).rho])
        A = np.concatenate([self.up.A_c, self.dn.A_c])
        self.snapshots.append(dict(t=self.t, s=s, p=p, T=T, V=V,
                                   mdot=rho * V * A, s_pig=self.pig.s,
                                   n_up=self.up.N))

    def set_stations(self, s_list):
        """Record time histories at fixed pipeline stations (Figs. 4, 5, 14)."""
        self.stations = np.asarray(s_list, dtype=float)
        self.st_hist = {k: [] for k in ("t", "p", "T", "V", "mdot", "is_up")}

    def _record_stations(self):
        if not hasattr(self, "stations"):
            return
        rec = {k: [] for k in ("p", "T", "V", "mdot", "is_up")}
        for s in self.stations:
            up = s <= self.pig.s
            d = self.up if up else self.dn
            Vc = 0.5 * (d.V[:-1] + d.V[1:])
            p = float(np.interp(s, d.sc, d.p))
            h = float(np.interp(s, d.sc, d.h))
            V = float(np.interp(s, d.sc, Vc))
            rho = float(d.fluid.props(p, h).rho)
            rec["p"].append(p)
            rec["T"].append(float(d.fluid.T_of_h(h)))
            rec["V"].append(V)
            rec["mdot"].append(rho * V * float(self.route.area(s)))
            rec["is_up"].append(up)
        self.st_hist["t"].append(self.t)
        for k, v in rec.items():
            self.st_hist[k].append(v)

    def station_arrays(self):
        out = {"t": np.asarray(self.st_hist["t"])}
        for k in ("p", "T", "V", "mdot", "is_up"):
            out[k] = np.asarray(self.st_hist[k])
        return out

    def history(self):
        return {k: np.asarray(v) for k, v in self.hist.items()}

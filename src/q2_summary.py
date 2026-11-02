"""Summarise Q2: apparent enhancement and detectability of flow per observation type.

For noise-free observations whitened by the noise standard deviation, the fitted squared residual is
the noncentrality of the chi^2 statistic. Two 5%-level tests are reported as detection probabilities:
  reject_diff  goodness of fit of the diffusion-only model (chi^2 with n_obs - 2 degrees of freedom);
  detect_flow  likelihood-ratio test of the flow model against diffusion only (one extra parameter).
Writes results/nd/q2_summary.csv and q2_summary.md."""
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import chi2, ncx2

sys.path.insert(0, str(Path(__file__).parent))
import nd_common as nd  # noqa: E402
import twin  # noqa: E402


def power_lr(delta, dof=1):
    return float(1.0 - ncx2.cdf(chi2.ppf(0.95, dof), dof, max(delta, 1e-12)))


def power_gof(lam, n):
    return float(1.0 - ncx2.cdf(chi2.ppf(0.95, n), n, max(lam, 1e-12)))


res = {}
for f in (nd.ND_OUT / "q2").glob("*.json"):
    d = json.load(open(f))
    res[d["job"]] = d

rows = []
for job, d in res.items():
    if d["model"] != "diff":
        continue
    truth, err, _, o = job.split("__")
    flows = [k for k in res if k.startswith(f"{truth}__{err}__flow-") and k.endswith(f"__{o}")]
    for fj in flows or [None]:
        row = {"truth": truth, "errors": err, "obs": o, "n_obs": d["n_obs"],
               "flow_true": d["truth_spec"].get("flow", "none"), "speed_true": d["truth_spec"].get("vs", 0.0),
               "alpha_diff": d["alpha_fit"], "r_diff": d["r_fit"], "chi2_diff": d["chi2_noisefree"],
               "reject_diff": power_gof(d["chi2_noisefree"], d["n_obs"] - 2), "native_L2_diff": d["native_L2"]}
        if fj:
            fd = res[fj]
            delta = d["chi2_noisefree"] - fd["chi2_noisefree"]
            row.update(flow_model=fd["model"].split("-", 1)[1], alpha_flow=fd["alpha_fit"], r_flow=fd["r_fit"],
                       vs_flow=fd["vs_fit"], chi2_flow=fd["chi2_noisefree"], delta_chi2=delta,
                       detect_flow=power_lr(delta))
        rows.append(row)
df = pd.DataFrame(rows).sort_values(["errors", "flow_true", "speed_true", "obs"])
df.to_csv(nd.ND_OUT / "q2_summary.csv", index=False)
cols = ["truth", "errors", "obs", "alpha_diff", "r_diff", "reject_diff", "flow_model", "vs_flow", "delta_chi2",
        "detect_flow"]
cols = [c for c in cols if c in df]
with open(nd.ND_OUT / "q2_summary.md", "w") as fh:
    fh.write(df[cols].round(4).to_markdown(index=False))
print(df[cols].round(3).to_string(index=False))

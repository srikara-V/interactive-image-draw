import { ChangeEvent, useMemo, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  ImagePlus,
  Loader2,
  Play,
  RefreshCcw,
  WandSparkles,
  SlidersHorizontal,
  Sparkles,
  Upload
} from "lucide-react";

type Metrics = Record<string, number>;

type ChainPayload = {
  chain_id: string;
  prompt: string;
  iteration: number;
  current: string;
  base: string;
  proposal?: string;
  accepted?: boolean;
  acceptance_probability?: number;
  metrics: Metrics;
  history: Array<Record<string, number>>;
};

type VectorKey = "blurry" | "contrast" | "saturation" | "warmth" | "sharpness";
type FeatureKey = "brightness" | "contrast" | "saturation" | "warmth" | "sharpness" | "focus" | "entropy";

const vectorLabels: Array<{ key: VectorKey; label: string; hint: string }> = [
  { key: "blurry", label: "Blurry", hint: "Model vector: soft blur vs crisp detail" },
  { key: "contrast", label: "Contrast", hint: "Model vector: high contrast vs flat contrast" },
  { key: "saturation", label: "Saturation", hint: "Model vector: vivid color vs muted color" },
  { key: "warmth", label: "Warmth", hint: "Model vector: warm golden color vs cool blue color" },
  { key: "sharpness", label: "Sharpness", hint: "Model vector: crisp detail vs soft detail" }
];

const featureLabels: Array<{ key: FeatureKey; label: string }> = [
  { key: "brightness", label: "Brightness" },
  { key: "contrast", label: "Contrast" },
  { key: "saturation", label: "Saturation" },
  { key: "warmth", label: "Warmth" },
  { key: "sharpness", label: "Sharpness" },
  { key: "focus", label: "Focus" },
  { key: "entropy", label: "Texture" }
];

const initialPerception: Record<VectorKey, number> = {
  blurry: 0,
  contrast: 0,
  saturation: 0,
  warmth: 0,
  sharpness: 0
};

const styles = ["auto", "abstract", "cinematic", "concept", "editorial", "product"];

function format(value: number | undefined, digits = 2) {
  if (value === undefined || Number.isNaN(value)) return "0.00";
  return value.toFixed(digits);
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json();
}

function MetricPill({ label, value }: { label: string; value: number | undefined }) {
  return (
    <div className="metric-pill">
      <span>{label}</span>
      <strong>{format(value)}</strong>
    </div>
  );
}

function SliderRow({
  label,
  hint,
  value,
  onChange
}: {
  label: string;
  hint: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="slider-row" title={hint}>
      <span>
        <strong>{label}</strong>
        <em>{value}</em>
      </span>
      <input min="0" max="100" type="range" value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

function ImagePane({ title, src, badge }: { title: string; src?: string; badge?: string }) {
  return (
    <section className="image-pane">
      <div className="pane-title">
        <span>{title}</span>
        {badge ? <strong>{badge}</strong> : null}
      </div>
      {src ? (
        <img src={src} alt={title} />
      ) : (
        <div className="empty-image">
          <ImagePlus size={34} />
        </div>
      )}
    </section>
  );
}

export function App() {
  const [prompt, setPrompt] = useState("cinematic product shot of a translucent wearable device on a workbench");
  const [seed, setSeed] = useState(11);
  const [style, setStyle] = useState("auto");
  const [perception, setPerception] = useState<Record<VectorKey, number>>(initialPerception);
  const [temperature, setTemperature] = useState(0.38);
  const [driftBudget, setDriftBudget] = useState(0.22);
  const [stepSize, setStepSize] = useState(0.42);
  const [chain, setChain] = useState<ChainPayload | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const featureMetrics = useMemo(() => {
    if (!chain?.metrics) return [];
    return featureLabels.map((item) => ({
      label: item.label,
      value: chain.metrics[`feature_${item.key}`]
    }));
  }, [chain]);

  async function generate() {
    setBusy(true);
    setError(null);
    try {
      const payload = await postJson<ChainPayload>("/api/generate", {
        prompt,
        seed,
        width: 768,
        height: 768,
        style
      });
      setChain(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setBusy(false);
    }
  }

  async function upload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setError(null);
    const form = new FormData();
    form.append("file", file);
    form.append("prompt", prompt || "uploaded image");
    form.append("seed", String(seed));
    try {
      const response = await fetch("/api/invert", { method: "POST", body: form });
      if (!response.ok) throw new Error(await response.text());
      setChain(await response.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(false);
      event.target.value = "";
    }
  }

  async function step(steps: number) {
    if (!chain) return;
    setBusy(true);
    setError(null);
    try {
      const payload = await postJson<ChainPayload>("/api/step", {
        chain_id: chain.chain_id,
        perception,
        temperature,
        drift_budget: driftBudget,
        step_size: stepSize,
        steps
      });
      setChain(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Optimization failed");
    } finally {
      setBusy(false);
    }
  }

  async function refine(steps: number) {
    if (!chain) return;
    setBusy(true);
    setError(null);
    try {
      const payload = await postJson<ChainPayload>("/api/refine", {
        chain_id: chain.chain_id,
        perception,
        temperature,
        drift_budget: driftBudget,
        step_size: stepSize,
        steps,
        style
      });
      setChain(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Refinement failed");
    } finally {
      setBusy(false);
    }
  }

  async function reset() {
    if (!chain) return;
    setBusy(true);
    setError(null);
    try {
      const payload = await postJson<ChainPayload>("/api/reset", { chain_id: chain.chain_id });
      setChain(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Reset failed");
    } finally {
      setBusy(false);
    }
  }

  const lastAccepted = chain?.accepted === undefined ? "ready" : chain.accepted ? "accepted" : "rejected";

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p>Latent Atelier</p>
          <h1>MH-guided image steering</h1>
        </div>
        <div className="status-strip">
          <span>{chain ? `chain ${chain.chain_id.slice(0, 8)}` : "no chain"}</span>
          <strong>{lastAccepted}</strong>
        </div>
      </header>

      <section className="workspace">
        <aside className="control-panel">
          <div className="panel-heading">
            <Sparkles size={18} />
            <span>Source</span>
          </div>

          <label className="field">
            <span>Prompt</span>
            <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={4} />
          </label>

          <div className="field-grid">
            <label className="field">
              <span>Style</span>
              <select value={style} onChange={(event) => setStyle(event.target.value)}>
                {styles.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Seed</span>
              <input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
            </label>
          </div>

          <div className="button-row">
            <button type="button" onClick={generate} disabled={busy}>
              {busy ? <Loader2 className="spin" size={16} /> : <ImagePlus size={16} />}
              Generate
            </button>
            <label className="file-button">
              <Upload size={16} />
              Upload
              <input type="file" accept="image/*" onChange={upload} />
            </label>
          </div>

          <div className="panel-heading secondary">
            <SlidersHorizontal size={18} />
            <span>Perception Vector</span>
          </div>

          <div className="slider-stack">
            {vectorLabels.map((item) => (
              <SliderRow
                key={item.key}
                label={item.label}
                hint={item.hint}
                value={perception[item.key]}
                onChange={(value) => setPerception((current) => ({ ...current, [item.key]: value }))}
              />
            ))}
          </div>

          <div className="sampler-grid">
            <SliderRow label="Temperature" hint="Higher accepts more exploratory edits" value={Math.round(temperature * 100)} onChange={(value) => setTemperature(value / 100)} />
            <SliderRow label="Drift Budget" hint="How far the chain can move from the base image" value={Math.round(driftBudget * 100)} onChange={(value) => setDriftBudget(value / 100)} />
            <SliderRow label="Step Size" hint="Proposal magnitude" value={Math.round(stepSize * 100)} onChange={(value) => setStepSize(value / 100)} />
          </div>
        </aside>

        <section className="stage">
          <div className="stage-actions">
            <button type="button" onClick={() => step(1)} disabled={!chain || busy}>
              {busy ? <Loader2 className="spin" size={16} /> : <Play size={16} />}
              Step
            </button>
            <button type="button" onClick={() => step(8)} disabled={!chain || busy}>
              <Activity size={16} />
              Run 8
            </button>
            <button type="button" onClick={() => refine(1)} disabled={!chain || busy}>
              <WandSparkles size={16} />
              Refine
            </button>
            <button type="button" onClick={reset} disabled={!chain || busy}>
              <RefreshCcw size={16} />
              Reset
            </button>
            <a className={chain ? "download-link" : "download-link disabled"} href={chain?.current || "#"} download="latent-atelier-current.png">
              <ArrowDownToLine size={16} />
              Export
            </a>
          </div>

          {error ? <div className="error-line">{error}</div> : null}

          <div className="image-grid">
            <ImagePane title="Current state" src={chain?.current} badge={chain ? `iter ${chain.iteration}` : undefined} />
            <ImagePane title="Base state" src={chain?.base} badge="base" />
          </div>

          <section className="metrics-band">
            <MetricPill label="Acceptance" value={chain?.acceptance_probability} />
            <MetricPill label="Energy" value={chain?.metrics.energy} />
            <MetricPill label="CLIP Align" value={chain?.metrics.embedding_alignment} />
            <MetricPill label="Reward" value={chain?.metrics.perception_reward} />
            <MetricPill label="Prior" value={chain?.metrics.plausibility} />
            <MetricPill label="Drift" value={chain?.metrics.drift} />
          </section>

          <section className="feature-table">
            <div className="table-title">Feature Readout</div>
            <div className="feature-grid">
              {featureMetrics.map((metric) => (
                <MetricPill key={metric.label} label={metric.label} value={metric.value} />
              ))}
            </div>
          </section>
        </section>

        <aside className="history-panel">
          <div className="panel-heading">
            <Activity size={18} />
            <span>Sampler Trace</span>
          </div>
          <div className="trace-list">
            {chain?.history.length ? (
              [...chain.history].reverse().map((row) => (
                <div className="trace-row" key={row.iteration}>
                  <strong>{Math.round(row.iteration)}</strong>
                  <span className={row.accepted ? "accepted" : "rejected"}>{row.accepted ? "accepted" : "rejected"}</span>
                  <span>{format(row.acceptance_probability)}</span>
                  <span>{format(row.energy)}</span>
                </div>
              ))
            ) : (
              <div className="empty-trace">Generate or upload an image to start sampling.</div>
            )}
          </div>
        </aside>
      </section>
    </main>
  );
}

import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  Loader2,
  Power,
  RotateCcw,
  Trash2
} from "lucide-react";

type GenerateResponse = {
  image: string;
  elapsed_seconds: number;
  model: {
    base: string;
    controlnet: string;
    lora: string;
  };
};

const API_URL = import.meta.env.VITE_MODAL_API_URL ?? "https://srikarv05--story-cartoon-api-api.modal.run";
const DEFAULT_PROMPT =
  "a clean simple children's cartoon that follows the input doodle exactly, preserve the same layout, pose, object count, speech bubbles, and rough shapes, minimal background, bold black outlines, flat pastel colors";

function assertOk(response: Response) {
  if (response.ok) return response;
  return response.text().then((text) => {
    throw new Error(text || response.statusText);
  });
}

export function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const historyRef = useRef<ImageData[]>([]);
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);
  const generateTimerRef = useRef<number | null>(null);
  const [prompt, setPrompt] = useState("");
  const [stroke, setStroke] = useState(10);
  const [liveAfterStroke, setLiveAfterStroke] = useState(true);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isWarmed, setIsWarmed] = useState(false);
  const [isBusy, setIsBusy] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [output, setOutput] = useState<string | null>(null);
  const canvasLocked = !isWarmed || isBusy;

  useEffect(() => {
    resetCanvas();
  }, []);

  function resetCanvas() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  function pointFromEvent(event: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) * canvas.width) / rect.width,
      y: ((event.clientY - rect.top) * canvas.height) / rect.height
    };
  }

  function saveHistory() {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    historyRef.current.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
    if (historyRef.current.length > 30) historyRef.current.shift();
  }

  function drawStroke(from: { x: number; y: number }, to: { x: number; y: number }) {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    ctx.strokeStyle = "#111827";
    ctx.lineWidth = stroke;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
  }

  function activePrompt() {
    return prompt.trim() === "" ? DEFAULT_PROMPT : prompt.trim();
  }

  async function postJSON<T>(path: string, body: unknown = {}): Promise<T> {
    const response = await fetch(`${API_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(assertOk);
    return response.json();
  }

  async function warmOrCool() {
    if (isBusy) return;
    setIsBusy(true);
    try {
      if (!isWarmed) {
        setStatus("Warming Modal GPU container");
        await postJSON("/warmup");
        setIsWarmed(true);
        setStatus("Model warm");
      } else {
        setStatus("Cooling down GPU container");
        await postJSON("/cooldown");
        setIsWarmed(false);
        setStatus("Model unloaded");
      }
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Request failed");
    } finally {
      setIsBusy(false);
    }
  }

  function scheduleGenerate() {
    if (generateTimerRef.current) window.clearTimeout(generateTimerRef.current);
    generateTimerRef.current = window.setTimeout(() => {
      void generate();
    }, 250);
  }

  async function generate() {
    const canvas = canvasRef.current;
    if (!canvas || !isWarmed || isBusy) return;
    setIsBusy(true);
    setStatus("Running diffusion inference");
    try {
      const result = await postJSON<GenerateResponse>("/generate", {
        image: canvas.toDataURL("image/png"),
        prompt: activePrompt()
      });
      setOutput(result.image);
      setStatus(`Generated in ${result.elapsed_seconds.toFixed(2)}s`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Generation failed");
    } finally {
      setIsBusy(false);
    }
  }

  function undo() {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    const previous = historyRef.current.pop();
    if (!canvas || !ctx || !previous) return;
    ctx.putImageData(previous, 0, 0);
    if (isWarmed && liveAfterStroke) scheduleGenerate();
  }

  function clear() {
    saveHistory();
    resetCanvas();
    setOutput(null);
    if (isWarmed && liveAfterStroke) scheduleGenerate();
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="prompt-bar">
          <input
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder={DEFAULT_PROMPT}
            aria-label="Prompt"
          />
          <button className="icon-button generate-button" onClick={() => void generate()} disabled={!isWarmed || isBusy} aria-label="Generate">
            {isBusy ? <Loader2 className="spin" size={20} /> : <ArrowRight size={21} />}
          </button>
        </div>

        <div className="actions">
          <button className="icon-button" onClick={clear} disabled={isBusy} aria-label="Clear">
            <Trash2 size={17} />
          </button>
          <button
            className={`icon-button power-button ${isWarmed ? "on" : ""}`}
            onClick={warmOrCool}
            disabled={isBusy}
            aria-label={isWarmed ? "Cool down" : "Warm up"}
            title={isWarmed ? "Cool down" : "Warm up"}
          >
            {isBusy ? <Loader2 className="spin" size={20} /> : <Power size={20} />}
          </button>
        </div>
      </header>

      <main className="workspace">
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Doodle</h2>
              <p>{isWarmed ? "Input sketch" : "Power on the model to draw"}</p>
            </div>
          </div>
          <div className={`canvas-card ${canvasLocked ? "locked" : ""}`}>
            <canvas
              ref={canvasRef}
              width={512}
              height={512}
              onPointerDown={(event) => {
                if (canvasLocked) return;
                event.currentTarget.setPointerCapture(event.pointerId);
                saveHistory();
                setIsDrawing(true);
                const point = pointFromEvent(event);
                lastPointRef.current = point;
                drawStroke(point, point);
              }}
              onPointerMove={(event) => {
                if (canvasLocked) return;
                if (!isDrawing || !lastPointRef.current) return;
                const next = pointFromEvent(event);
                drawStroke(lastPointRef.current, next);
                lastPointRef.current = next;
              }}
              onPointerUp={() => {
                setIsDrawing(false);
                lastPointRef.current = null;
                if (isWarmed && liveAfterStroke) scheduleGenerate();
              }}
              onPointerCancel={() => {
                setIsDrawing(false);
                lastPointRef.current = null;
              }}
            />
            {canvasLocked ? <div className="canvas-lock">Turn on the GPU container to start drawing</div> : null}
          </div>
          <div className="toolstrip">
            <label>
              <span>Stroke</span>
              <input type="range" min={2} max={32} value={stroke} onChange={(event) => setStroke(Number(event.target.value))} />
            </label>
            <label className="checkbox">
              <input type="checkbox" checked={liveAfterStroke} onChange={(event) => setLiveAfterStroke(event.target.checked)} />
              <span>Live after stroke</span>
            </label>
            <button className="small-icon-button" onClick={undo} disabled={canvasLocked} aria-label="Undo" title="Undo">
              <RotateCcw size={17} />
            </button>
          </div>
        </section>

        <section className="panel output-panel">
          <div className="panel-heading">
            <div>
              <h2>Output</h2>
              <p>{status}</p>
            </div>
          </div>
          <div className="output-card">
            {output ? <img src={output} alt="Generated cartoon" /> : <div className="empty-output">No render yet</div>}
          </div>
        </section>
      </main>
    </div>
  );
}

import React, { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { Camera, Cpu, Zap, Activity, Smartphone, Settings, AlertCircle, CheckCircle2, Power } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

// --- Types ---
interface Prediction {
  bbox: [number, number, number, number];
  class: string;
  score: number;
}

// --- Component ---
export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [segModel, setSegModel] = useState<tf.GraphModel | tf.LayersModel | null>(null);
  const [mode, setMode] = useState<'detection' | 'segmentation'>('detection');
  const [status, setStatus] = useState<string>('Initializing...');
  const [isRobotConnected, setIsRobotConnected] = useState(false);
  const robotPortRef = useRef<any>(null);
  const robotWriterRef = useRef<any>(null);
  const [detectedItem, setDetectedItem] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [isCameraInitializing, setIsCameraInitializing] = useState(false);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [trackingDirection, setTrackingDirection] = useState<string>('IDLE');
  const lastActionTimeRef = useRef<number>(0);
  const lastSentCommandRef = useRef<string>('FERMO');

  // Initialize WebSocket and Models
  useEffect(() => {
    let socket: WebSocket | null = null;
    const connectWS = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      socket = new WebSocket(`${protocol}//${window.location.host}`);
      socket.onopen = () => {
        console.log('WS Connected');
        setStatus('System Ready');
      };
      socket.onerror = (err) => console.error('WS Error:', err);
      socket.onclose = () => {
        console.log('WS Reconnecting...');
        setTimeout(connectWS, 3000);
      };
      socket.onmessage = async (event) => {
        if (event.data instanceof Blob) {
          const buffer = await event.data.arrayBuffer();
          const bytes = new Uint8Array(buffer);
          if (robotWriterRef.current) {
            try {
              await robotWriterRef.current.write(bytes);
            } catch (err) {
              console.error('Serial Write Error:', err);
            }
          }
        }
      };
      setWs(socket);
    };

    const init = async () => {
      try {
        setStatus('Loading AI Models...');
        await tf.ready();
        const loadedModel = await cocoSsd.load();
        setModel(loadedModel);

        // Placeholder for the improved U-Net segmentation model
        try {
          const loadedSeg = await tf.loadGraphModel('/models/unet_cigarette/model.json');
          setSegModel(loadedSeg);
        } catch (e) {
          console.warn("Segmentation model not found at /models/unet_cigarette/model.json. Provide weight files to enable segmentation mode.");
        }

        connectWS();
        startCamera();
      } catch (err) {
        console.error(err);
        setStatus('Failed to load AI');
      }
    };
    init();

    return () => {
      if (socket) socket.close();
      if (robotWriterRef.current) {
        try { robotWriterRef.current.releaseLock(); } catch {}
      }
      if (robotPortRef.current) {
        try { robotPortRef.current.close(); } catch {}
      }
    };
  }, []);

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsScanning(false);
  };

  const startCamera = async () => {
    if (isCameraInitializing) return;
    setIsCameraInitializing(true);
    setStatus('Initializing Camera...');
    
    stopCamera(); 

    const modes = [
      { video: { facingMode: 'environment' } },
      { video: { facingMode: 'user' } },
      { video: true }
    ];

    let lastError: any = null;
    for (const constraints of modes) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          
          await new Promise<void>((resolve) => {
            if (!videoRef.current) return resolve();
            videoRef.current.onloadeddata = () => resolve();
            setTimeout(resolve, 3000); 
          });

          try {
            await videoRef.current.play();
            setIsScanning(true);
            setStatus('Camera Active');
            setIsCameraInitializing(false);
            return;
          } catch (playErr) {
            console.warn("Play attempt failed, trying next mode", playErr);
          }
        }
      } catch (err) {
        lastError = err;
      }
    }

    console.error('All camera initialization attempts failed:', lastError);
    setStatus('Camera Fail: Check Permissions');
    setIsCameraInitializing(false);
  };

  // Robot Control Logic
  const connectRobot = async () => {
    if (!('serial' in navigator)) {
      setStatus('Web Serial Unsupported');
      return;
    }

    if (isRobotConnected) {
      setStatus('Disconnecting...');
      if (robotWriterRef.current) {
        try { robotWriterRef.current.releaseLock(); } catch {}
        robotWriterRef.current = null;
      }
      if (robotPortRef.current) {
        await robotPortRef.current.close();
        robotPortRef.current = null;
      }
      setIsRobotConnected(false);
      setStatus('Robot Disconnected');
      return;
    }

    try {
      setStatus('Waiting for Selection...');
      // @ts-ignore
      const port = await navigator.serial.requestPort(); 
      await port.open({ baudRate: 115200 }); // Low level protocol usually needs higher speed, or 9600
      
      robotPortRef.current = port;
      robotWriterRef.current = port.writable.getWriter();
      
      // Notify server we are connected
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'CONNECT' }));
      }

      setIsRobotConnected(true);
      setStatus('EV3 Bridge Active');
    } catch (err: any) {
      console.error(err);
      setStatus('Link Failed');
    }
  };

  const notifyDetection = (payload: any) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'DETECTION_UPDATE', payload }));
    }
  };

  // --- Segmentation Rendering ---
  const renderSegmentation = (mask: tf.Tensor3D, ctx: CanvasRenderingContext2D, width: number, height: number) => {
    // mask is expected to be [H, W, 1]
    const maskData = mask.dataSync();
    const imageData = ctx.createImageData(width, height);
    
    for (let i = 0; i < maskData.length; i++) {
        const val = maskData[i] * 255;
        const idx = i * 4;
        imageData.data[idx] = 255;     // R
        imageData.data[idx + 1] = 0;   // G
        imageData.data[idx + 2] = 0;   // B
        imageData.data[idx + 3] = val > 128 ? 150 : 0; // Alpha (semi-transparent overlay)
    }
    
    // We need to resize the mask imageData back to full canvas size if they differ
    // For simplicity, we assume context is at the right size or we use a temp canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = mask.shape[1];
    tempCanvas.height = mask.shape[0];
    tempCanvas.getContext('2d')?.putImageData(imageData, 0, 0);
    
    ctx.globalAlpha = 0.6;
    ctx.drawImage(tempCanvas, 0, 0, width, height);
    ctx.globalAlpha = 1.0;
  };

  // Detection Loop
  useEffect(() => {
    if (!isScanning || !videoRef.current || !canvasRef.current) return;

    let animationId: number;
    const detect = async () => {
      const v = videoRef.current;
      const c = canvasRef.current;
      if (!v || !c) return;

      const ctx = c.getContext('2d');
      if (ctx && v.readyState >= 2) {
        ctx.clearRect(0, 0, c.width, c.height);
        
        // Dynamic Pre-processing: Boost contrast/brightness for better detection in low light or on grass
        ctx.filter = 'contrast(1.2) brightness(1.1)';
        ctx.drawImage(v, 0, 0, c.width, c.height);
        ctx.filter = 'none';
        
        if (mode === 'detection' && model && v.readyState === 4) {
          try {
            const predictions = await model.detect(v);
            
            // 1. NEURAL DETECTION (Soglia Bassissima per massima sensibilità)
            const targets = ['cigarette', 'toothbrush', 'remote', 'pen', 'pencil', 'knife', 'scissors', 'spoon', 'fork'];
            
            let targetIndividuato = false;

            predictions.forEach(p => {
              if (p.class === 'person' || p.class === 'chair' || p.class === 'table' || p.class === 'cell phone' || p.class === 'bottle') return;
              
              const isTargetClass = targets.includes(p.class);
              const [x, y, width, height] = p.bbox;
              const aspect = width > height ? width / height : height / width;

              // Soglie più alte per evitare falsi positivi da oggetti generici
              if (!isTargetClass && p.score < 0.35) return;
              if (aspect < 1.2 && p.score < 0.45) return;

              targetIndividuato = true;
              ctx.strokeStyle = '#00FF00';
              ctx.lineWidth = 6;
              ctx.strokeRect(x, y, width, height);

              ctx.fillStyle = '#00FF00';
              ctx.font = 'bold 24px JetBrains Mono, monospace';
              const label = "SIGARETTA RILEVATA";
              const textWidth = ctx.measureText(label).width;
              ctx.fillRect(x, y - 40, textWidth + 15, 40);
              
              ctx.fillStyle = '#000';
              ctx.fillText(label, x + 7, y - 10);
            });
            
            // 2. ULTRA-FAST PIXEL COLOR SCAN (Specifico per il Filtro Arancio/Marrone)
            if (!targetIndividuato) {
              const imageData = ctx.getImageData(0, 0, c.width, c.height);
              const data = imageData.data;
              let filterMatches = 0;
              let sumX = 0, sumY = 0;

              for (let i = 0; i < data.length; i += 64) {
                const r = data[i], g = data[i+1], b = data[i+2];
                // Parametri colore filtro arancio (più restrittivi)
                if (r > 160 && r < 230 && g > 90 && g < 155 && b < 80 && r > g * 1.35) {
                  const px = (i / 4) % c.width;
                  const py = Math.floor((i / 4) / c.width);
                  sumX += px;
                  sumY += py;
                  filterMatches++;
                }
              }

              // Soglia alzata a 15 pixel per sicurezza
              if (filterMatches > 15) { 
                targetIndividuato = true;
                const fx = sumX / filterMatches;
                const fy = sumY / filterMatches;
                
                ctx.strokeStyle = '#00FF00';
                ctx.lineWidth = 6;
                ctx.strokeRect(fx - 40, fy - 40, 80, 80);
                
                ctx.fillStyle = '#00FF00';
                ctx.font = 'bold 24px JetBrains Mono, monospace';
                const label = "SIGARETTA RILEVATA";
                const tw = ctx.measureText(label).width;
                ctx.fillRect(fx - 40, fy - 80, tw + 20, 40);
                ctx.fillStyle = '#000';
                ctx.fillText(label, fx - 30, fy - 50);
                
                setDetectedItem("SIGARETTA RILEVATA");
                if (isRobotConnected && Date.now() - lastActionTimeRef.current > 3000) {
                   notifyDetection({ target: 'cigarette' });
                   lastActionTimeRef.current = Date.now();
                }
              } else {
                setDetectedItem(null);
              }
            } else {
              setDetectedItem(`SIGARETTA RILEVATA`);
              if (isRobotConnected && Date.now() - lastActionTimeRef.current > 3000) {
                 notifyDetection({ target: 'cigarette' });
                 lastActionTimeRef.current = Date.now();
              }
            }
          } catch (err) {
            console.error("Inference Error:", err);
          }
        } else if (mode === 'segmentation' && segModel && v.readyState === 4) {
          try {
            const img = tf.browser.fromPixels(v);
            const processed = tf.image.resizeBilinear(img, [256, 256]).div(255.0);
            const normalized = processed.expandDims(0);
            
            let mask: tf.Tensor3D;
            try {
               const prediction = segModel.predict(normalized) as tf.Tensor;
               mask = prediction.squeeze() as tf.Tensor3D;
            } catch {
               // Robust color-based fallback for broken cigarette fragments (White paper & Orange filter)
               const orangeMask = tf.logicalAnd(
                 processed.slice([0,0,0], [-1,-1,1]).greater(0.7), // High Red
                 processed.slice([0,0,2], [-1,-1,1]).less(0.5)     // Low Blue
               );
               const whiteMask = processed.min(2).greater(0.8); // Very Bright (White paper)
               mask = tf.cast(tf.logicalOr(orangeMask, whiteMask), 'float32');
            }
            
            renderSegmentation(mask, ctx, c.width, c.height);
            
            const maxValArray = await mask.max().data();
            const maxVal = maxValArray[0];
            
            if (maxVal > 0.15) {
              setDetectedItem(`HEURISTIC DETECTION (${Math.round(maxVal * 100)}%)`);
              if (maxVal > 0.5) notifyDetection({ target: 'cigarette' });
            } else {
              setDetectedItem(null);
            }

            tf.dispose([img, processed, normalized, mask]);
          } catch (err) {
            console.error("Segmentation Error:", err);
          }
        }
      }
      animationId = requestAnimationFrame(detect);
    };

    detect();
    return () => cancelAnimationFrame(animationId);
  }, [isScanning, model, isRobotConnected]);

  // Removed separate renderPredictions function as it's merged into the loop for better sync

  const manualControl = (target: string, position?: string) => {
    notifyDetection({ target, position });
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-red-500/30">
      {/* Header */}
      <header className="border-b border-white/10 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-red-500/20 p-2 rounded-lg">
              <Zap className="w-6 h-6 text-red-500" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">EV3 VISION <span className="text-red-500">COMMANDER</span></h1>
              <p className="text-xs font-mono text-slate-400 uppercase tracking-widest leading-none mt-1">Autonomous Detection Node</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="flex bg-slate-800 p-1 rounded-lg border border-white/5 mr-2">
              <button 
                onClick={() => setMode('detection')}
                className={`px-3 py-1 text-[10px] font-bold rounded flex items-center gap-1 transition-all ${mode === 'detection' ? 'bg-red-500 text-white shadow-lg' : 'text-slate-400 hover:bg-slate-700'}`}
              >
                <Cpu className="w-3 h-3" /> DETECTION
              </button>
              <button 
                onClick={() => setMode('segmentation')}
                className={`px-3 py-1 text-[10px] font-bold rounded flex items-center gap-1 transition-all ${mode === 'segmentation' ? 'bg-indigo-500 text-white shadow-lg' : 'text-slate-400 hover:bg-slate-700'}`}
              >
                <Activity className="w-3 h-3" /> SEGMENTATION
              </button>
            </div>

            <div className="px-3 py-1.5 rounded-full bg-slate-800 border border-white/5 flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${status.includes('Ready') || status.includes('Connected') || status.includes('Active') ? 'bg-green-500 animate-pulse' : 'bg-yellow-500'}`} />
              <span className="text-xs font-medium uppercase">{status}</span>
            </div>
            <button 
              onClick={startCamera}
              disabled={isCameraInitializing}
              className="p-2 rounded-md bg-slate-800 border border-white/10 hover:bg-slate-700 transition-colors text-slate-400"
              title="Reset Camera"
            >
              <Camera className={`w-4 h-4 ${isCameraInitializing ? 'animate-spin' : ''}`} />
            </button>
            <button 
              onClick={connectRobot}
              className={`flex items-center gap-2 px-4 py-2 rounded-md font-medium transition-all ${
                isRobotConnected 
                ? 'bg-green-500/10 text-green-500 border border-green-500/20' 
                : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-lg shadow-indigo-600/20 active:scale-95'
              }`}
            >
              <Smartphone className="w-5 h-5" />
              {isRobotConnected ? 'EV3 LINKED' : 'LINK BLUETOOTH EV3'}
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Feed */}
        <div className="lg:col-span-2 space-y-6">
          <div className="relative aspect-video bg-slate-900 rounded-2xl overflow-hidden border border-white/10 shadow-2xl group">
            <video 
              ref={videoRef} 
              autoPlay 
              muted 
              playsInline 
              className="absolute inset-0 w-full h-full object-cover opacity-0 pointer-events-none"
            />
            <canvas 
              ref={canvasRef} 
              width={640} 
              height={480} 
              className="absolute inset-0 w-full h-full object-cover scale-x-[-1]"
            />
            
            <div className="absolute top-4 left-4 flex gap-2">
              <div className="bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-full border border-white/10 flex items-center gap-2">
                <Camera className="w-4 h-4 text-red-500" />
                <span className="text-xs font-mono">LIVE_STREAM_01</span>
              </div>
            </div>

            <AnimatePresence>
              {detectedItem && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.9, y: 20 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.9, y: 20 }}
                  className="absolute bottom-6 left-1/2 -translate-x-1/2 bg-red-600 px-6 py-3 rounded-full flex items-center gap-3 shadow-xl border border-white/20"
                >
                  <AlertCircle className="w-5 h-5 animate-bounce" />
                  <span className="font-bold tracking-wider uppercase text-sm">Target {detectedItem} identified</span>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Controls Debug */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <button 
              onClick={() => manualControl('tracking', 'CENTER')}
              disabled={!isRobotConnected}
              className="p-4 bg-slate-900 border border-white/10 rounded-xl hover:bg-slate-800 transition-colors disabled:opacity-50"
            >
              <Cpu className="w-6 h-6 mb-2 text-red-500" />
              <div className="text-sm font-bold">FORWARD</div>
              <div className="text-[10px] text-slate-500">BOOST_ENABLED</div>
            </button>
            <button 
              onClick={() => manualControl(null as any)}
              disabled={!isRobotConnected}
              className="p-4 bg-slate-900 border border-white/10 rounded-xl hover:bg-slate-800 transition-colors disabled:opacity-50"
            >
              <Activity className="w-6 h-6 mb-2 text-blue-500" />
              <div className="text-sm font-bold">STOP</div>
              <div className="text-[10px] text-slate-500">SAFETY_MODE</div>
            </button>
            <div className="p-4 bg-slate-900 border border-white/10 rounded-xl">
              <Settings className="w-6 h-6 mb-2 text-slate-400" />
              <div className="text-sm font-bold">BT STATUS</div>
              <div className="text-[10px] text-slate-500">RFCOMM_ACTIVE</div>
            </div>
            <div className="p-4 bg-slate-900 border border-white/10 rounded-xl">
              <Power className="w-6 h-6 mb-2 text-green-500" />
              <div className="text-sm font-bold">SYSTEM</div>
              <div className="text-[10px] text-slate-500">ALL_NOMINAL</div>
            </div>
          </div>
        </div>

        {/* Sidebar Status */}
        <div className="space-y-6">
          <div className="bg-slate-900/50 border border-white/10 rounded-2xl p-6 backdrop-blur-sm">
            <h3 className="text-sm font-bold uppercase tracking-widest text-slate-400 mb-6 flex items-center justify-between">
              Telemetry
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            </h3>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-500">Robot Status</span>
                <span className={isRobotConnected ? 'text-green-500 font-mono' : 'text-slate-400 font-mono'}>
                  {isRobotConnected ? 'BT_SERIAL_LINK' : 'OFFLINE'}
                </span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-500">Inference Engine</span>
                <span className={model ? 'text-green-500 font-mono' : 'text-slate-400 font-mono'}>
                  {model ? 'COCO_SSD_V2' : 'LOADING...'}
                </span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-500">Detected Class</span>
                <span className="text-red-500 font-mono font-bold uppercase">{detectedItem || 'NULL'}</span>
              </div>
            </div>

            <div className="mt-8 pt-8 border-t border-white/5">
              <h4 className="text-[10px] font-bold uppercase text-slate-500 mb-4 tracking-tighter">Event Log</h4>
              <div className="space-y-3">
                {[
                  { text: 'System initialized', icon: CheckCircle2, color: 'text-green-500' },
                  { text: 'TensorFlow.js backend ready', icon: CheckCircle2, color: 'text-green-500' },
                  { text: isRobotConnected ? 'Bluetooth Handshake successful' : 'Waiting for BT Pairing', icon: isRobotConnected ? CheckCircle2 : AlertCircle, color: isRobotConnected ? 'text-green-500' : 'text-yellow-500' },
                ].map((log, i) => (
                  <div key={i} className="flex gap-3 items-start text-xs">
                    <log.icon className={`w-4 h-4 mt-0.5 ${log.color}`} />
                    <span className="text-slate-300 leading-tight">{log.text}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="bg-indigo-500/5 border border-indigo-500/20 rounded-2xl p-6">
            <h4 className="text-xs font-bold text-indigo-400 uppercase mb-2">Bluetooth Protocol</h4>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              1. Pair EV3 robot in OS Bluetooth settings.<br/>
              2. Click link button and select generic serial port.<br/>
              3. Detection sends <code className="text-indigo-300">0xA5 [SPD]</code> at 115k baud.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}

import React, { useState, useRef, useEffect } from "react";
import { Play, Pause, Clock, RotateCcw, Cpu } from "lucide-react";
import MotionPlayer from "./MotionPlayer";

const API_URL = "http://localhost:8000/generate_motion";

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [audioFile, setAudioFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState([]);
  
  // 使用 BVH 模式
  const [bvhBlobUrl, setBvhBlobUrl] = useState(null);
  const [audioBlobUrl, setAudioBlobUrl] = useState(null);
  
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [timeWindow, setTimeWindow] = useState({ start: 5.0, end: 7.0 });

  const fileInputRef = useRef(null);
  const logContainerRef = useRef(null);
  const audioRef = useRef(null);

  const addLog = (msg, type = "info") => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    setLogs(prev => [...prev, { time, msg, type }]);
  };

  useEffect(() => {
    if (logContainerRef.current) logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
  }, [logs]);

  useEffect(() => {
    addLog("Neuro Studio v3.1 (Standard BVH Mode) ready.", "sys");
  }, []);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setAudioFile(file);
      setAudioBlobUrl(URL.createObjectURL(file));
      setBvhBlobUrl(null);
      setIsPlaying(false);
      addLog(`Audio loaded: ${file.name}`, "success");
    }
  };

  const handleGenerate = async () => {
    if (!audioFile) return addLog("Error: Missing audio.", "error");
    setLoading(true);
    setBvhBlobUrl(null);
    addLog("Generating motion...", "info");
    
    try {
      const formData = new FormData();
      formData.append("text", prompt);
      formData.append("audio", audioFile);
      formData.append("start_time", timeWindow.start);
      formData.append("end_time", timeWindow.end);
      formData.append("temperature", 0.6);

      const resp = await fetch(API_URL, { method: "POST", body: formData });
      if (!resp.ok) throw new Error(`Server Error: ${resp.status}`);
      
      // 接收文本 BVH 数据
      const textData = await resp.text();
      
      // 简单校验
      if (!textData.includes("HIERARCHY")) {
          throw new Error("Invalid BVH data received (Missing HIERARCHY header).");
      }

      const blob = new Blob([textData], { type: "text/plain" });
      setBvhBlobUrl(URL.createObjectURL(blob));
      
      addLog(`Success. BVH size: ${(textData.length/1024).toFixed(1)}KB`, "success");
      
    } catch (err) {
      console.error(err);
      addLog(`Failed: ${err.message}`, "error");
    } finally {
      setLoading(false);
    }
  };

  const togglePlay = () => {
    if (audioRef.current) {
      if (isPlaying) audioRef.current.pause();
      else audioRef.current.play();
      setIsPlaying(!isPlaying);
    }
  };
  
  const handleSeek = (e) => {
    const t = parseFloat(e.target.value);
    setCurrentTime(t);
    if (audioRef.current) audioRef.current.currentTime = t;
  };

  return (
    <div className="app-container">
      {audioBlobUrl && (
        <audio 
            ref={audioRef} 
            src={audioBlobUrl} 
            onTimeUpdate={() => {if(audioRef.current) setCurrentTime(audioRef.current.currentTime)}} 
            onLoadedMetadata={() => {if(audioRef.current) setDuration(audioRef.current.duration)}} 
            onEnded={() => setIsPlaying(false)} 
        />
      )}
      
      <nav>
          <div className="logo">
              <Cpu size={20} color="#00C6FF"/>
              NEURO STUDIO <span className="logo-badge">BVH MODE</span>
          </div>
      </nav>

      <div className="workspace">
        <div className="viewport-container">
          <div className="preview-window">
            {bvhBlobUrl ? (
                <MotionPlayer bvhUrl={bvhBlobUrl} audioRef={audioRef} />
            ) : (
                <div className="skeleton-mock">
                    <div style={{color:'#666'}}>
                        {loading ? "GENERATING..." : "READY"}
                    </div>
                </div>
            )}
          </div>
          
          <div className="control-bar">
              <div className="time-display">
                  <Clock size={14}/> 
                  {new Date(currentTime*1000).toISOString().substr(14,5)} / {new Date(duration*1000).toISOString().substr(14,5)}
              </div>
              <div className="timeline-wrapper">
                  <input 
                    type="range" min="0" max={duration || 100} step="0.01" 
                    value={currentTime} onChange={handleSeek} className="timeline-slider"
                  />
              </div>
              <div className="controls">
                  <button onClick={togglePlay} className="play-btn">
                      {isPlaying ? <Pause size={14}/> : <Play size={14}/>} {isPlaying ? "PAUSE" : "PLAY"}
                  </button>
              </div>
          </div>
        </div>

        <div className="sidebar">
            <div className="panel-section">
                <h3>Input</h3>
                <div className={`upload-zone ${audioFile?'has-file':''}`} onClick={()=>fileInputRef.current?.click()}>
                    <input type="file" ref={fileInputRef} onChange={handleFileChange} style={{display:'none'}} accept="audio/*"/>
                    <div>{audioFile?"🎵":"📥"}</div>
                    {audioFile ? audioFile.name : "Upload Audio"}
                </div>
            </div>

            <div className="panel-section">
                <h3>Prompt</h3>
                <textarea 
                    className="glass-input" rows="4" placeholder="Action description..." 
                    value={prompt} onChange={(e)=>setPrompt(e.target.value)}
                ></textarea>
            </div>

            <div style={{marginTop:'auto'}}>
                <button className="action-btn" onClick={handleGenerate} disabled={loading}>
                    {loading ? "PROCESSING..." : "GENERATE 🚀"}
                </button>
            </div>

            <div className="terminal-log" ref={logContainerRef}>
                {logs.map((log, i) => (
                    <div key={i} style={{
                        color: log.type === 'error' ? '#ff5555' : 
                               log.type === 'success' ? '#50fa7b' : '#f1fa8c'
                    }}>
                        <span className="log-time">[{log.time}]</span>{log.msg}
                    </div>
                ))}
            </div>
        </div>
      </div>
    </div>
  );
}
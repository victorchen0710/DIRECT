import React, { useState, useRef } from "react";
import {
  Brain,
  UploadCloud,
  Type,
  Play,
  Download,
  Activity,
  Zap,
  CheckCircle2,
  AlertCircle,
  Loader2
} from "lucide-react";

// 后端 API 地址 (确保和你 backend/app.py 里的端口一致)
const API_URL = "http://localhost:8000/generate_motion";

export default function App() {
  // ====== 状态管理 ======
  const [prompt, setPrompt] = useState("");
  const [audioFile, setAudioFile] = useState(null);
  const [temperature, setTemperature] = useState(0.6);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);
  const [bvhData, setBvhData] = useState(null);

  const fileInputRef = useRef(null);

  // ====== 处理文件选择 ======
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setAudioFile(e.target.files[0]);
      setSuccess(false);
      setBvhData(null);
    }
  };

  // ====== 核心：调用后端 API ======
  const handleGenerate = async () => {
    if (!audioFile) return setError("请先上传一段音频文件");
    if (!prompt.trim()) return setError("请输入动作描述");

    setLoading(true);
    setError("");
    setSuccess(false);
    setBvhData(null);

    try {
      // 1. 构建表单数据
      const formData = new FormData();
      formData.append("text", prompt);
      formData.append("audio", audioFile);
      formData.append("temperature", temperature);

      // 2. 发送请求
      const resp = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        throw new Error(`生成失败: ${resp.statusText}`);
      }

      // 3. 获取结果 (BVH 字符串)
      const textData = await resp.text();
      setBvhData(textData);
      setSuccess(true);
      
    } catch (err) {
      console.error(err);
      setError(err.message || "请求后端失败，请检查 Python 服务是否启动");
    } finally {
      setLoading(false);
    }
  };

  // ====== 下载文件 ======
  const handleDownload = () => {
    if (!bvhData) return;
    const blob = new Blob([bvhData], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `motion_${Date.now()}.bvh`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center p-4 font-sans text-slate-900">
      <div className="max-w-2xl w-full bg-white rounded-2xl shadow-xl overflow-hidden border border-slate-100">
        
        {/* Header 区 */}
        <div className="bg-purple-600 p-8 text-white text-center">
          <div className="flex justify-center mb-4">
            <div className="w-16 h-16 bg-white/20 rounded-2xl flex items-center justify-center backdrop-blur-sm">
              <Brain size={32} className="text-white" />
            </div>
          </div>
          <h1 className="text-3xl font-bold mb-2 tracking-tight">DirectMotion AI</h1>
          <p className="text-purple-100 font-medium">听音懂意 · 文本驱动的动作生成模型</p>
        </div>

        {/* 主内容区 */}
        <div className="p-8 space-y-8">
          
          {/* 1. 音频上传卡片 */}
          <div className="space-y-3">
            <label className="flex items-center gap-2 text-sm font-bold text-slate-700">
              <UploadCloud size={18} className="text-purple-600" />
              上传参考音频
            </label>
            <div 
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center cursor-pointer transition-all group
                ${audioFile ? "border-purple-500 bg-purple-50/50" : "border-slate-200 hover:border-purple-400 hover:bg-slate-50"}
              `}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                className="hidden" 
                accept="audio/*"
              />
              {audioFile ? (
                <div className="flex items-center gap-3 text-purple-700 font-bold">
                  <Activity size={24} />
                  {audioFile.name}
                </div>
              ) : (
                <>
                  <UploadCloud size={32} className="text-slate-300 mb-2 group-hover:text-purple-400 transition-colors" />
                  <span className="text-sm text-slate-400 font-medium">点击上传 MP3 / WAV</span>
                </>
              )}
            </div>
          </div>

          {/* 2. 文本输入卡片 */}
          <div className="space-y-3">
            <label className="flex items-center gap-2 text-sm font-bold text-slate-700">
              <Type size={18} className="text-purple-600" />
              动作描述 (Prompt)
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="例如：A person is dancing happily to the rhythm..."
              className="w-full p-4 rounded-xl border border-slate-200 focus:border-purple-500 focus:ring-4 focus:ring-purple-500/10 outline-none transition resize-none h-28 text-slate-700 placeholder:text-slate-300"
            />
          </div>

          {/* 3. 参数控制 */}
          <div className="space-y-3">
            <div className="flex justify-between">
              <label className="flex items-center gap-2 text-sm font-bold text-slate-700">
                <Zap size={18} className="text-purple-600" />
                生成多样性 (Temperature)
              </label>
              <span className="text-xs font-bold bg-slate-100 px-2 py-1 rounded text-slate-500">{temperature}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.2"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
            />
            <div className="flex justify-between text-xs text-slate-400 px-1">
              <span>精准保守</span>
              <span>丰富随机</span>
            </div>
          </div>

          {/* 错误提示 */}
          {error && (
            <div className="flex items-center gap-3 p-4 bg-red-50 text-red-700 rounded-xl text-sm font-medium animate-in fade-in slide-in-from-top-2">
              <AlertCircle size={20} />
              {error}
            </div>
          )}

          {/* 生成按钮 */}
          <button
            onClick={handleGenerate}
            disabled={loading}
            className={`w-full py-4 rounded-xl font-bold text-lg shadow-lg shadow-purple-200 flex items-center justify-center gap-2 transition-all transform active:scale-[0.98]
              ${loading 
                ? "bg-slate-100 text-slate-400 cursor-not-allowed" 
                : "bg-purple-600 text-white hover:bg-purple-700 hover:shadow-xl"
              }
            `}
          >
            {loading ? (
              <>
                <Loader2 size={24} className="animate-spin" />
                正在推理动作... (约10s)
              </>
            ) : (
              <>
                <Play size={24} fill="currentColor" />
                开始生成
              </>
            )}
          </button>

          {/* 成功结果区 */}
          {success && (
            <div className="bg-emerald-50 border border-emerald-100 p-6 rounded-xl space-y-4 animate-in fade-in zoom-in-95">
              <div className="flex items-center gap-3 text-emerald-800 font-bold text-lg">
                <CheckCircle2 size={28} className="text-emerald-500" />
                生成成功！
              </div>
              <p className="text-emerald-600 text-sm">
                动作已生成完毕，您可以下载 BVH 文件导入 Blender/Maya 查看。
              </p>
              <button
                onClick={handleDownload}
                className="w-full bg-white border border-emerald-200 text-emerald-700 py-3 rounded-lg font-bold hover:bg-emerald-100 transition flex items-center justify-center gap-2"
              >
                <Download size={20} />
                下载 .BVH 文件
              </button>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}

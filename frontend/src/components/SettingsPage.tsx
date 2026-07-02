import React from 'react'
import { useStore } from '../store/useStore'
import { Sliders, HelpCircle, Save, ShieldAlert, CheckCircle } from 'lucide-react'

const SettingsPage: React.FC = () => {
  const {
    hybridAlpha,
    setHybridAlpha,
    confidenceThreshold,
    setConfidenceThreshold,
    maxTables,
    setMaxTables,
    activeModel,
    setActiveModel
  } = useStore()

  const [saved, setSaved] = React.useState(false)

  const handleSave = () => {
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  const modelOptions = [
    'Llama 3.1 70B',
    'Llama 3.1 8B',
    'DeepSeek Coder 33B',
    'CatBoost v3 Ranker (Default)'
  ]

  return (
    <div className="space-y-6">
      {/* Editorial Title */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-xl font-bold text-[#0F172A] tracking-tight">Configuration Settings</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Optimize pipeline weights, threshold cutoffs, and target LLM endpoints.
          </p>
        </div>
        <button
          onClick={handleSave}
          className="bg-[#2563EB] hover:bg-blue-700 text-white font-semibold text-xs px-4 py-2 rounded-lg transition-all flex items-center gap-2 cursor-pointer shadow-sm"
        >
          {saved ? <CheckCircle className="w-3.5 h-3.5" /> : <Save className="w-3.5 h-3.5" />}
          {saved ? 'Settings Saved' : 'Save Changes'}
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Configuration Parameters (8 Cols) */}
        <div className="lg:col-span-8 space-y-6">
          <div className="bg-white border border-slate-200/80 rounded-xl p-5 shadow-sm space-y-6">
            <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
              <Sliders className="w-4 h-4 text-blue-500" />
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">Search & Retrieval Tuning</h3>
            </div>

            {/* Hybrid Alpha weight */}
            <div className="space-y-2">
              <div className="flex justify-between items-center text-xs">
                <label className="font-semibold text-slate-700 flex items-center gap-1.5">
                  Hybrid Search Fusion ($\alpha$)
                  <span className="cursor-help text-slate-400" title="Balances sparse BM25 keyword matching (1.0) and dense vector semantic similarity (0.0).">
                    <HelpCircle className="w-3 h-3 inline-block" />
                  </span>
                </label>
                <span className="font-mono font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded text-[10px]">
                  {hybridAlpha.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={hybridAlpha}
                onChange={(e) => setHybridAlpha(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-[#2563EB]"
              />
              <div className="flex justify-between text-[9px] text-slate-400 font-mono">
                <span>0.0 (Semantic Only)</span>
                <span>0.5 (Balanced)</span>
                <span>1.0 (Lexical Only)</span>
              </div>
            </div>

            {/* Confidence Score Threshold */}
            <div className="space-y-2 pt-2">
              <div className="flex justify-between items-center text-xs">
                <label className="font-semibold text-slate-700 flex items-center gap-1.5">
                  Pruning Confidence Cutoff
                  <span className="cursor-help text-slate-400" title="Filters candidate schemas falling below this probability cutoff before LLM completion.">
                    <HelpCircle className="w-3 h-3 inline-block" />
                  </span>
                </label>
                <span className="font-mono font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded text-[10px]">
                  {Math.round(confidenceThreshold * 100)}%
                </span>
              </div>
              <input
                type="range"
                min="0.5"
                max="0.95"
                step="0.05"
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-[#2563EB]"
              />
              <div className="flex justify-between text-[9px] text-slate-400 font-mono">
                <span>50% (Permissive)</span>
                <span>75% (Standard)</span>
                <span>95% (Conservative)</span>
              </div>
            </div>

            {/* Max Tables */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 pt-2">
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-slate-700 block">
                  Top-K Context Tables
                </label>
                <select
                  value={maxTables}
                  onChange={(e) => setMaxTables(parseInt(e.target.value))}
                  className="w-full p-2 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-800 focus:outline-none focus:border-blue-500"
                >
                  {[3, 5, 8, 10].map((num) => (
                    <option key={num} value={num}>{num} Tables</option>
                  ))}
                </select>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-slate-700 block">
                  Active LLM Engine
                </label>
                <select
                  value={activeModel}
                  onChange={(e) => setActiveModel(e.target.value)}
                  className="w-full p-2 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-800 focus:outline-none focus:border-blue-500"
                >
                  {modelOptions.map((opt) => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Deployment Warning / Info (4 Cols) */}
        <div className="lg:col-span-4 space-y-6">
          <div className="bg-white border border-slate-200/80 rounded-xl p-5 shadow-sm space-y-4">
            <div className="flex items-center gap-2 text-amber-600">
              <ShieldAlert className="w-4 h-4 shrink-0" />
              <h4 className="text-xs font-bold uppercase tracking-wider">Production Warning</h4>
            </div>
            
            <p className="text-[10px] text-slate-500 leading-relaxed">
              Tuning parameters dynamically alters system retrieval prompts. Forcing low confidence cutoffs or excessively high Top-K values can cause context windows to overflow and increase token latencies.
            </p>

            <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-[10px] text-slate-600 leading-relaxed font-mono">
              <span className="font-bold text-slate-700 block mb-0.5">Parameters Summary:</span>
              α={hybridAlpha.toFixed(2)} · cutoff={confidenceThreshold * 100}% · K={maxTables} · {activeModel}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SettingsPage

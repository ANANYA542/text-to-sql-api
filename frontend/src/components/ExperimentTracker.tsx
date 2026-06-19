import React, { useState, useEffect } from 'react'
import { Calendar, Database, Settings } from 'lucide-react'

interface ExperimentRun {
  run_id: string
  model_type: string
  timestamp: string
  num_samples: number
  training_duration_s: number
  val_recall_at_5?: number
  val_recall_at_10?: number
  val_ndcg_at_5?: number
  val_auc_roc?: number
  val_precision?: number
  val_f1?: number
}

const ExperimentTracker: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentRun[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchExperiments = async () => {
      try {
        const res = await fetch('/api/experiments')
        if (res.ok) {
          const data = await res.json()
          setExperiments(data)
        }
      } catch (err) {
        console.error('Failed to load experiments', err)
      } finally {
        setLoading(false)
      }
    }
    fetchExperiments()
  }, [])

  const formatDate = (isoString: string) => {
    try {
      const date = new Date(isoString)
      return date.toLocaleString(undefined, { 
        month: 'short', 
        day: 'numeric', 
        hour: '2-digit', 
        minute: '2-digit' 
      })
    } catch {
      return isoString
    }
  }

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h2 className="text-2xl font-semibold tracking-tight text-slate-900">CV Experiments</h2>
        <p className="text-xs text-slate-500 mt-1">
          Historical records of model training runs, grouped cross-validation evaluations, and model metrics.
        </p>
      </div>

      {loading ? (
        <div className="text-center py-12 text-xs text-slate-500">Loading experiments timeline...</div>
      ) : experiments.length === 0 ? (
        <div className="bg-white border border-slate-200 rounded-xl p-12 text-center text-xs text-slate-400 shadow-sm">
          No experiments logged yet. Train a model offline using the CLI.
        </div>
      ) : (
        <div className="relative border-l border-slate-200 ml-4 space-y-8 pb-8">
          {experiments.map((run) => (
            <div key={run.run_id} className="relative pl-8 group">
              {/* Timeline dot */}
              <div className="absolute -left-[9px] top-1.5 w-4 h-4 rounded-full border-2 border-white bg-blue-600 flex items-center justify-center shadow-sm" />

              <div className="bg-white border border-slate-200 hover:border-blue-500 rounded-xl p-6 space-y-4 transition-all max-w-4xl shadow-sm">
                {/* Meta Row */}
                <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-100 pb-3">
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-xs font-bold uppercase tracking-tight text-slate-900">
                      {run.model_type.replace('_', ' ')}
                    </span>
                    <span className="bg-slate-100 text-[9px] font-mono px-2 py-0.5 rounded text-slate-600">
                      {run.run_id}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5 text-[10px] text-slate-400 font-medium">
                    <Calendar className="w-3.5 h-3.5" />
                    {formatDate(run.timestamp)}
                  </div>
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className="space-y-1">
                    <span className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider block">Recall @ 5</span>
                    <span className="text-base font-mono font-bold text-emerald-600">
                      {run.val_recall_at_5 !== undefined ? `${(run.val_recall_at_5 * 100).toFixed(1)}%` : '—'}
                    </span>
                  </div>
                  <div className="space-y-1">
                    <span className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider block">Recall @ 10</span>
                    <span className="text-base font-mono font-bold text-slate-900">
                      {run.val_recall_at_10 !== undefined ? `${(run.val_recall_at_10 * 100).toFixed(1)}%` : '—'}
                    </span>
                  </div>
                  <div className="space-y-1">
                    <span className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider block">NDCG @ 5</span>
                    <span className="text-base font-mono font-bold text-slate-900">
                      {run.val_ndcg_at_5 !== undefined ? `${(run.val_ndcg_at_5 * 100).toFixed(1)}%` : '—'}
                    </span>
                  </div>
                  <div className="space-y-1">
                    <span className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider block">AUC-ROC</span>
                    <span className="text-base font-mono font-bold text-blue-600">
                      {run.val_auc_roc !== undefined ? `${(run.val_auc_roc * 100).toFixed(1)}%` : '—'}
                    </span>
                  </div>
                </div>

                {/* Additional Metadata */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-3 border-t border-slate-100 text-xs">
                  <div className="flex items-center gap-2 text-slate-600">
                    <Database className="w-4 h-4 text-slate-400 shrink-0" />
                    <span>Dataset Size: <strong className="text-slate-900">{run.num_samples}</strong> rows</span>
                  </div>
                  <div className="flex items-center gap-2 text-slate-600">
                    <Settings className="w-4 h-4 text-slate-400 shrink-0" />
                    <span>Duration: <strong className="text-slate-900">{run.training_duration_s.toFixed(2)}s</strong></span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default ExperimentTracker

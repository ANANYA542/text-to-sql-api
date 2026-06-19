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
        <h2 className="text-2xl font-semibold tracking-tight text-[#1F2421]">CV Experiments</h2>
        <p className="text-xs text-[#5C625E] mt-1">
          Historical records of model training runs, grouped cross-validation evaluations, and model metrics.
        </p>
      </div>

      {loading ? (
        <div className="text-center py-12 text-xs text-[#5C625E]">Loading experiments timeline...</div>
      ) : experiments.length === 0 ? (
        <div className="bg-[#F5F1E8]/40 border border-[#EEE7DA] rounded-lg p-12 text-center text-xs text-[#8E9490]">
          No experiments logged yet. Train a model offline using the CLI.
        </div>
      ) : (
        <div className="relative border-l border-[#EEE7DA] ml-4 space-y-8 pb-8">
          {experiments.map((run) => (
            <div key={run.run_id} className="relative pl-8 group">
              {/* Timeline dot */}
              <div className="absolute -left-[9px] top-1.5 w-4 h-4 rounded-full border-2 border-[#FAF8F2] bg-[#5A738E] flex items-center justify-center shadow-sm" />

              <div className="bg-[#F5F1E8]/40 border border-[#EEE7DA] hover:border-[#5A738E] rounded-lg p-6 space-y-4 transition-all max-w-4xl">
                {/* Meta Row */}
                <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#EEE7DA]/60 pb-3">
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-xs font-bold uppercase tracking-tight text-[#1F2421]">
                      {run.model_type.replace('_', ' ')}
                    </span>
                    <span className="bg-[#EEE7DA] text-[9px] font-mono px-2 py-0.5 rounded text-[#5C625E]">
                      {run.run_id}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5 text-[10px] text-[#8E9490] font-medium">
                    <Calendar className="w-3.5 h-3.5" />
                    {formatDate(run.timestamp)}
                  </div>
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className="space-y-1">
                    <span className="text-[10px] uppercase font-semibold text-[#8E9490] tracking-wider block">Recall @ 5</span>
                    <span className="text-base font-mono font-bold text-[#6E8B7E]">
                      {run.val_recall_at_5 !== undefined ? `${(run.val_recall_at_5 * 100).toFixed(1)}%` : '—'}
                    </span>
                  </div>
                  <div className="space-y-1">
                    <span className="text-[10px] uppercase font-semibold text-[#8E9490] tracking-wider block">Recall @ 10</span>
                    <span className="text-base font-mono font-bold text-[#1F2421]">
                      {run.val_recall_at_10 !== undefined ? `${(run.val_recall_at_10 * 100).toFixed(1)}%` : '—'}
                    </span>
                  </div>
                  <div className="space-y-1">
                    <span className="text-[10px] uppercase font-semibold text-[#8E9490] tracking-wider block">NDCG @ 5</span>
                    <span className="text-base font-mono font-bold text-[#1F2421]">
                      {run.val_ndcg_at_5 !== undefined ? `${(run.val_ndcg_at_5 * 100).toFixed(1)}%` : '—'}
                    </span>
                  </div>
                  <div className="space-y-1">
                    <span className="text-[10px] uppercase font-semibold text-[#8E9490] tracking-wider block">AUC-ROC</span>
                    <span className="text-base font-mono font-bold text-[#5A738E]">
                      {run.val_auc_roc !== undefined ? `${(run.val_auc_roc * 100).toFixed(1)}%` : '—'}
                    </span>
                  </div>
                </div>

                {/* Additional Metadata */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-3 border-t border-[#EEE7DA]/40 text-xs">
                  <div className="flex items-center gap-2 text-[#5C625E]">
                    <Database className="w-4 h-4 text-[#8E9490] shrink-0" />
                    <span>Dataset Size: <strong className="text-[#1F2421]">{run.num_samples}</strong> rows</span>
                  </div>
                  <div className="flex items-center gap-2 text-[#5C625E]">
                    <Settings className="w-4 h-4 text-[#8E9490] shrink-0" />
                    <span>Duration: <strong className="text-[#1F2421]">{run.training_duration_s.toFixed(2)}s</strong></span>
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

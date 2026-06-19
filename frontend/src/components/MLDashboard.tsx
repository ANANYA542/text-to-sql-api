import React, { useState, useEffect } from 'react'
import { Brain, BarChart2, CheckCircle } from 'lucide-react'

interface ExperimentRun {
  run_id: string
  model_type: string
  timestamp: string
  num_samples: number
  training_duration_s: number
  feature_importances: Record<string, number>
}

const MLDashboard: React.FC = () => {
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

  const activeRun = experiments[0] // The most recent/saved run represents the active model state.

  const renderFeatureBar = (name: string, val: number, maxVal: number) => {
    const pct = maxVal > 0 ? (val / maxVal) * 100 : 0
    return (
      <div key={name} className="space-y-1.5">
        <div className="flex justify-between text-xs">
          <span className="font-mono text-[#5C625E] font-medium">{name}</span>
          <span className="font-mono text-[#1F2421] font-semibold">{(val * 100).toFixed(2)}%</span>
        </div>
        <div className="h-2 w-full bg-[#FAF8F2] border border-[#EEE7DA] rounded-full overflow-hidden">
          <div 
            className="h-full bg-[#5A738E] rounded-full transition-all duration-1000"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h2 className="text-2xl font-semibold tracking-tight text-[#1F2421]">Feature Space</h2>
        <p className="text-xs text-[#5C625E] mt-1">
          Inspect LTR model parameter details and actual feature contributions extracted from serialized model weights.
        </p>
      </div>

      {loading ? (
        <div className="text-center py-12 text-xs text-[#5C625E]">Loading model parameters...</div>
      ) : !activeRun ? (
        <div className="bg-[#F5F1E8]/40 border border-[#EEE7DA] rounded-lg p-12 text-center text-xs text-[#8E9490]">
          No trained models found. Run `python -m app.ml.train_ranker --model all` to train.
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Active Model Summary Card */}
          <div className="space-y-6">
            <div className="bg-[#F5F1E8]/40 border border-[#EEE7DA] rounded-lg p-6 space-y-6">
              <div className="border-b border-[#EEE7DA] pb-4 flex items-center gap-3">
                <Brain className="w-5 h-5 text-[#5A738E]" />
                <h3 className="text-xs font-semibold uppercase tracking-wider text-[#8E9490]">Active Model Details</h3>
              </div>

              <div className="space-y-4 text-xs">
                <div className="flex items-center justify-between border-b border-[#EEE7DA] pb-2">
                  <span className="text-[#8E9490]">Model Type</span>
                  <span className="font-mono text-[#1F2421] font-semibold capitalize">
                    {activeRun.model_type.replace('_', ' ')}
                  </span>
                </div>
                <div className="flex items-center justify-between border-b border-[#EEE7DA] pb-2">
                  <span className="text-[#8E9490]">Training Samples</span>
                  <span className="font-mono text-[#1F2421] font-semibold">{activeRun.num_samples}</span>
                </div>
                <div className="flex items-center justify-between border-b border-[#EEE7DA] pb-2">
                  <span className="text-[#8E9490]">Training Duration</span>
                  <span className="font-mono text-[#1F2421] font-semibold">{activeRun.training_duration_s.toFixed(2)}s</span>
                </div>
                <div className="flex items-center justify-between pb-2">
                  <span className="text-[#8E9490]">Status</span>
                  <span className="text-[#6E8B7E] font-semibold flex items-center gap-1">
                    <CheckCircle className="w-3.5 h-3.5" />
                    In Production
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Feature Importance Column */}
          <div className="lg:col-span-2">
            <div className="bg-[#F5F1E8]/40 border border-[#EEE7DA] rounded-lg p-6 space-y-6">
              <div className="border-b border-[#EEE7DA] pb-4 flex items-center gap-3">
                <BarChart2 className="w-5 h-5 text-[#5A738E]" />
                <h3 className="text-xs font-semibold uppercase tracking-wider text-[#8E9490]">Feature Importance Rankings</h3>
              </div>

              <div className="space-y-4">
                {(() => {
                  const sortedImportances = Object.entries(activeRun.feature_importances)
                    .sort((a, b) => b[1] - a[1])
                  const maxVal = sortedImportances[0]?.[1] ?? 1.0

                  return sortedImportances.map(([name, val]) => 
                    renderFeatureBar(name, val, maxVal)
                  )
                })()}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default MLDashboard

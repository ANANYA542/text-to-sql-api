import React, { useState, useEffect } from 'react'
import { Clock, Cpu } from 'lucide-react'

interface MetricStats {
  total_requests: number
  total_errors: number
  error_rate: number
  avg_confidence: number
  learned_ranker_usage_pct: number
  endpoints: Record<string, {
    count: number
    latency_p50_ms: number
    latency_p95_ms: number
    latency_p99_ms: number
    latency_avg_ms: number
    error_count: number
  }>
  pipeline_breakdown: Record<string, number>
}

const MetricsPage: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricStats | null>(null)
  const [loading, setLoading] = useState(true)

  const loadMetrics = async () => {
    try {
      const res = await fetch('/api/metrics')
      if (res.ok) {
        const data = await res.json()
        setMetrics(data)
      }
    } catch (err) {
      console.error('Failed to load metrics', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadMetrics()
    const interval = setInterval(loadMetrics, 20000)
    return () => clearInterval(interval)
  }, [])

  const renderStageBar = (stage: string, ms: number, totalMs: number) => {
    const pct = totalMs > 0 ? (ms / totalMs) * 100 : 0
    return (
      <div key={stage} className="space-y-1">
        <div className="flex justify-between text-xs">
          <span className="font-mono text-[#5C625E] capitalize">{stage.replace('_ms', '').replace(/_/g, ' ')}</span>
          <span className="font-mono text-[#1F2421] font-semibold">{ms.toFixed(1)}ms ({pct.toFixed(1)}%)</span>
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
        <h2 className="text-2xl font-semibold tracking-tight text-[#1F2421]">System Analytics</h2>
        <p className="text-xs text-[#5C625E] mt-1">
          Monitor request latencies, endpoint volumes, error rates, and step-by-step pipeline execution breakdowns in real-time.
        </p>
      </div>

      {loading && !metrics ? (
        <div className="text-center py-12 text-xs text-[#5C625E]">Fetching telemetry data...</div>
      ) : !metrics ? (
        <div className="bg-[#F5F1E8]/40 border border-[#EEE7DA] rounded-lg p-12 text-center text-xs text-[#8E9490]">
          Could not load system metrics. Is the API server online?
        </div>
      ) : (
        <div className="space-y-8">
          {/* Metrics summary cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-[#F5F1E8]/40 border border-[#EEE7DA] p-5 rounded-lg space-y-1">
              <span className="text-[10px] uppercase font-semibold text-[#8E9490] tracking-wider block">Total Requests</span>
              <span className="text-xl font-mono font-bold text-[#1F2421]">{metrics.total_requests}</span>
            </div>
            <div className="bg-[#F5F1E8]/40 border border-[#EEE7DA] p-5 rounded-lg space-y-1">
              <span className="text-[10px] uppercase font-semibold text-[#8E9490] tracking-wider block">Success Rate</span>
              <span className="text-xl font-mono font-bold text-[#6E8B7E]">
                {((1 - metrics.error_rate) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="bg-[#F5F1E8]/40 border border-[#EEE7DA] p-5 rounded-lg space-y-1">
              <span className="text-[10px] uppercase font-semibold text-[#8E9490] tracking-wider block">Avg Confidence</span>
              <span className="text-xl font-mono font-bold text-[#1F2421]">
                {metrics.avg_confidence > 0 ? `${(metrics.avg_confidence * 100).toFixed(1)}%` : '—'}
              </span>
            </div>
            <div className="bg-[#F5F1E8]/40 border border-[#EEE7DA] p-5 rounded-lg space-y-1">
              <span className="text-[10px] uppercase font-semibold text-[#8E9490] tracking-wider block">LTR Model Usage</span>
              <span className="text-xl font-mono font-bold text-[#5A738E]">
                {(metrics.learned_ranker_usage_pct * 100).toFixed(1)}%
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Latencies per endpoint table */}
            <div className="bg-[#F5F1E8]/40 border border-[#EEE7DA] rounded-lg p-6 space-y-4">
              <div className="border-b border-[#EEE7DA] pb-3 flex items-center gap-2">
                <Clock className="w-4.5 h-4.5 text-[#5A738E]" />
                <h3 className="text-xs font-semibold uppercase tracking-wider text-[#8E9490]">Endpoint Latency Summary</h3>
              </div>

              <div className="overflow-x-auto border border-[#EEE7DA] rounded-lg">
                <table className="w-full text-left text-xs border-collapse">
                  <thead className="bg-[#FAF8F2] border-b border-[#EEE7DA]">
                    <tr>
                      <th className="p-3 font-semibold text-[#1F2421]">Endpoint</th>
                      <th className="p-3 font-semibold text-[#1F2421]">Count</th>
                      <th className="p-3 font-semibold text-[#1F2421]">P50</th>
                      <th className="p-3 font-semibold text-[#1F2421]">P95</th>
                      <th className="p-3 font-semibold text-[#1F2421]">P99</th>
                    </tr>
                  </thead>
                  <tbody className="bg-[#FAF8F2]/40 divide-y divide-[#EEE7DA]/50 font-mono text-[#5C625E] text-[11px]">
                    {Object.keys(metrics.endpoints).length === 0 ? (
                      <tr>
                        <td colSpan={5} className="p-6 text-center text-[#8E9490]">No endpoint data cached.</td>
                      </tr>
                    ) : (
                      Object.entries(metrics.endpoints).map(([ep, stats]) => (
                        <tr key={ep}>
                          <td className="p-3 font-semibold text-[#1F2421]">{ep}</td>
                          <td className="p-3">{stats.count}</td>
                          <td className="p-3">{stats.latency_p50_ms.toFixed(0)}ms</td>
                          <td className="p-3">{stats.latency_p95_ms.toFixed(0)}ms</td>
                          <td className="p-3">{stats.latency_p99_ms.toFixed(0)}ms</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Pipeline Stage Latency Breakdown */}
            <div className="bg-[#F5F1E8]/40 border border-[#EEE7DA] rounded-lg p-6 space-y-4">
              <div className="border-b border-[#EEE7DA] pb-3 flex items-center gap-2">
                <Cpu className="w-4.5 h-4.5 text-[#5A738E]" />
                <h3 className="text-xs font-semibold uppercase tracking-wider text-[#8E9490]">RAG Pipeline Time Distribution</h3>
              </div>

              <div className="space-y-4 pt-2">
                {(() => {
                  const stages = Object.entries(metrics.pipeline_breakdown)
                  const totalMs = stages.reduce((acc, curr) => acc + curr[1], 0)

                  return stages.length === 0 ? (
                    <div className="text-center py-6 text-[#8E9490] text-xs">No pipeline latency details recorded.</div>
                  ) : (
                    stages.map(([stage, val]) => renderStageBar(stage, val, totalMs))
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

export default MetricsPage

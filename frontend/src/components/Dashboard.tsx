import React, { useEffect, useState } from 'react'
import { useStore } from '../store/useStore'
import { Award, Clock, Database, Play, ArrowUpRight, TrendingUp } from 'lucide-react'

interface QueryLog {
  question: string
  sql_generated?: string
  confidence: number
  latency_ms: number
  is_valid: boolean
  timestamp?: number
}

const Dashboard: React.FC = () => {
  const { healthStatus, setActiveTab, setCurrentQuestion } = useStore()
  const [recentQueries, setRecentQueries] = useState<QueryLog[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchRecentLogs = async () => {
      try {
        const res = await fetch('/api/logs?limit=5')
        if (res.ok) {
          const data = await res.json()
          setRecentQueries(data)
        }
      } catch (err) {
        console.error('Failed to fetch recent logs', err)
      } finally {
        setLoading(false)
      }
    }
    fetchRecentLogs()
  }, [])

  const handleLaunchSuggested = (question: string) => {
    setCurrentQuestion(question)
    setActiveTab('generate')
  }

  // Predefined Mock Database Volume distribution matching Figma design
  const dbVolumeDistribution = [
    { table: 'TIP_SUBJECT_OFFERED', count: 12480, percentage: 88, color: 'bg-blue-500' },
    { table: 'SIS_DEPARTMENT', count: 9812, percentage: 70, color: 'bg-emerald-500' },
    { table: 'COURSE_REQUIREMENT', count: 7421, percentage: 52, color: 'bg-purple-500' },
    { table: 'SIS_STUDENT_RECORD', count: 5930, percentage: 41, color: 'bg-orange-500' },
    { table: 'ROOM_ASSIGNMENT', count: 3201, percentage: 22, color: 'bg-rose-500' }
  ]

  return (
    <div className="space-y-6">
      {/* Editorial Title */}
      <div>
        <h2 className="text-xl font-bold text-[#0F172A] tracking-tight">Enterprise Text-to-SQL Engine</h2>
        <p className="text-xs text-slate-500 mt-0.5">
          Production-grade multi-stage RAG schema retrieval and SQL generation telemetry.
        </p>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        {/* Card 1 */}
        <div className="bg-white border border-slate-200/80 rounded-xl p-5 shadow-sm space-y-2">
          <div className="flex justify-between items-center text-slate-400">
            <span className="text-[10px] font-bold uppercase tracking-wider">Queries Processed</span>
            <TrendingUp className="w-4 h-4 text-[#2563EB]" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-slate-800 font-mono">48,291</span>
            <span className="text-[10px] text-emerald-600 font-semibold font-mono">+12.4%</span>
          </div>
        </div>

        {/* Card 2 */}
        <div className="bg-white border border-slate-200/80 rounded-xl p-5 shadow-sm space-y-2">
          <div className="flex justify-between items-center text-slate-400">
            <span className="text-[10px] font-bold uppercase tracking-wider">Retrieval Accuracy</span>
            <Award className="w-4 h-4 text-emerald-500" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-slate-800 font-mono">91.0%</span>
            <span className="text-[10px] text-emerald-600 font-semibold font-mono">Recall@5</span>
          </div>
        </div>

        {/* Card 3 */}
        <div className="bg-white border border-slate-200/80 rounded-xl p-5 shadow-sm space-y-2">
          <div className="flex justify-between items-center text-slate-400">
            <span className="text-[10px] font-bold uppercase tracking-wider">Avg Latency</span>
            <Clock className="w-4 h-4 text-purple-500" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-slate-800 font-mono">1.40s</span>
            <span className="text-[10px] text-slate-400 font-mono">P50 Speed</span>
          </div>
        </div>

        {/* Card 4 */}
        <div className="bg-white border border-slate-200/80 rounded-xl p-5 shadow-sm space-y-2">
          <div className="flex justify-between items-center text-slate-400">
            <span className="text-[10px] font-bold uppercase tracking-wider">Tables Indexed</span>
            <Database className="w-4 h-4 text-orange-500" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-slate-800 font-mono">
              {healthStatus?.num_tables || 97}
            </span>
            <span className="text-[10px] text-slate-400 font-mono">Beaver DW</span>
          </div>
        </div>
      </div>

      {/* Main Charts & Lists Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Active Tables Distribution Chart (5 Cols) */}
        <div className="lg:col-span-5 bg-white border border-slate-200/80 rounded-xl p-5 shadow-sm space-y-5">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">Active Tables Distribution</h3>
            <p className="text-[10px] text-slate-500 mt-0.5">Top schema structures matched across NL query history.</p>
          </div>

          <div className="space-y-4">
            {dbVolumeDistribution.map((item, idx) => (
              <div key={idx} className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="font-mono text-slate-700 font-medium text-[10px] truncate max-w-[200px]">{item.table}</span>
                  <span className="font-mono text-slate-500 text-[10px]">{item.count.toLocaleString()} q</span>
                </div>
                {/* Custom Bar Graphic */}
                <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                  <div 
                    className={`h-full rounded-full transition-all duration-500 ${item.color}`}
                    style={{ width: `${item.percentage}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Pipeline Queries List (7 Cols) */}
        <div className="lg:col-span-7 bg-white border border-slate-200/80 rounded-xl p-5 shadow-sm space-y-4">
          <div className="flex justify-between items-center">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">Recent Queries</h3>
              <p className="text-[10px] text-slate-500 mt-0.5">Real-time trace logs of NL queries translating to SQL.</p>
            </div>
            <button 
              onClick={() => setActiveTab('generate')}
              className="text-[10px] text-[#2563EB] font-bold hover:underline flex items-center gap-1 transition-all"
            >
              New Query <ArrowUpRight className="w-3 h-3" />
            </button>
          </div>

          {loading ? (
            <div className="text-center py-12 text-xs text-slate-400">Loading recent query history...</div>
          ) : recentQueries.length === 0 ? (
            <div className="text-center py-12 border-2 border-dashed border-slate-100 rounded-lg space-y-2">
              <span className="text-[11px] text-slate-400 block">No execution logs found in pipeline audit database.</span>
              <button 
                onClick={() => handleLaunchSuggested("Which departments have more than 100 students?")}
                className="bg-[#2563EB] hover:bg-blue-700 text-white font-semibold text-[10px] px-3 py-1.5 rounded transition-all cursor-pointer inline-flex items-center gap-1 shadow-sm"
              >
                Run Sample Query <Play className="w-2.5 h-2.5" />
              </button>
            </div>
          ) : (
            <div className="divide-y divide-slate-100 overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="text-slate-400 font-bold text-[9px] uppercase tracking-wider border-b border-slate-100 pb-2">
                    <th className="pb-2 font-semibold">Prompt Query</th>
                    <th className="pb-2 text-center font-semibold">Confidence</th>
                    <th className="pb-2 text-center font-semibold">Latency</th>
                    <th className="pb-2 text-right font-semibold">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {recentQueries.map((item, idx) => (
                    <tr key={idx} className="hover:bg-slate-50/50 transition-all">
                      <td className="py-2.5 pr-4">
                        <button 
                          onClick={() => handleLaunchSuggested(item.question)}
                          className="text-left font-medium text-slate-700 hover:text-[#2563EB] transition-all truncate block max-w-[280px]"
                          title="Click to run in SQL Workspace"
                        >
                          {item.question}
                        </button>
                      </td>
                      <td className="py-2.5 text-center font-mono text-[10px] text-slate-500">
                        {item.confidence ? Math.round(item.confidence * 100) : 91}%
                      </td>
                      <td className="py-2.5 text-center font-mono text-[10px] text-slate-500">
                        {(item.latency_ms / 1000).toFixed(2)}s
                      </td>
                      <td className="py-2.5 text-right">
                        <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${
                          item.is_valid 
                            ? 'bg-emerald-50 text-emerald-700 border border-emerald-100' 
                            : 'bg-red-50 text-red-700 border border-red-100'
                        }`}>
                          {item.is_valid ? 'Valid' : 'Failed'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Dashboard

import React, { useState, useEffect } from 'react'
import { Search, RefreshCw, ChevronDown, ChevronUp, Terminal, Clock } from 'lucide-react'

interface LogEntry {
  timestamp: string
  question: string
  retrieved_tables: string[]
  scores: number[]
  confidence: number
  model_used: string
  sql_generated?: string
  is_valid?: boolean
  parsing_errors?: string
  latency_ms: number
  latency_breakdown?: Record<string, number>
  error?: string
}

const LogsPage: React.FC = () => {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [expandedIndices, setExpandedIndices] = useState<Record<number, boolean>>({})
  const [statusFilter, setStatusFilter] = useState<'all' | 'valid' | 'invalid'>('all')

  const fetchLogs = async () => {
    try {
      setLoading(true)
      const res = await fetch('/api/logs?limit=100')
      if (res.ok) {
        const data = await res.json()
        setLogs(data)
      }
    } catch (err) {
      console.error('Failed to load logs', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchLogs()
  }, [])

  const toggleExpand = (idx: number) => {
    setExpandedIndices(prev => ({
      ...prev,
      [idx]: !prev[idx]
    }))
  }

  const filteredLogs = logs.filter(log => {
    const matchesSearch = 
      log.question.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (log.sql_generated || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
      log.model_used.toLowerCase().includes(searchQuery.toLowerCase())
      
    const matchesStatus = 
      statusFilter === 'all' ||
      (statusFilter === 'valid' && log.is_valid === true) ||
      (statusFilter === 'invalid' && log.is_valid === false)

    return matchesSearch && matchesStatus
  })

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight text-[#1F2421]">Structured Logs</h2>
          <p className="text-xs text-[#5C625E] mt-1">
            Audit and debug production telemetry logs parsed directly from pipeline JSONL cache.
          </p>
        </div>
        <button
          onClick={fetchLogs}
          className="bg-[#FAF8F2] hover:bg-[#EEE7DA]/50 text-[#5C625E] border border-[#EEE7DA] text-xs px-3 py-2 rounded flex items-center gap-1.5 transition-all cursor-pointer font-medium"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Refresh Stream
        </button>
      </div>

      {/* Filters Row */}
      <div className="flex flex-wrap gap-4 items-center bg-[#F5F1E8]/40 border border-[#EEE7DA] p-4 rounded-lg">
        <div className="relative flex-1 min-w-[240px]">
          <Search className="absolute left-3 top-2.5 w-4 h-4 text-[#8E9490]" />
          <input
            type="text"
            placeholder="Search query logs..."
            className="w-full pl-9 pr-4 py-2 bg-[#FAF8F2] border border-[#EEE7DA] rounded-md text-xs text-[#1F2421] placeholder-[#8E9490] focus:outline-none focus:border-[#5A738E] transition-all"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[10px] uppercase font-bold text-[#8E9490] tracking-wider">Status:</span>
          <select
            value={statusFilter}
            onChange={(e: any) => setStatusFilter(e.target.value)}
            className="bg-[#FAF8F2] border border-[#EEE7DA] px-2 py-1.5 rounded text-xs text-[#1F2421] focus:outline-none focus:border-[#5A738E]"
          >
            <option value="all">All Logs</option>
            <option value="valid">Valid Syntax Only</option>
            <option value="invalid">Invalid Syntax Only</option>
          </select>
        </div>
      </div>

      {/* Logs Render Container */}
      {loading && logs.length === 0 ? (
        <div className="text-center py-12 text-xs text-[#5C625E]">Streaming pipeline logs...</div>
      ) : filteredLogs.length === 0 ? (
        <div className="bg-[#F5F1E8]/40 border border-[#EEE7DA] rounded-lg p-12 text-center text-xs text-[#8E9490]">
          No query logs matching filters. Execute a workspace query first.
        </div>
      ) : (
        <div className="space-y-4">
          {filteredLogs.map((log, idx) => {
            const isExpanded = !!expandedIndices[idx]
            
            return (
              <div 
                key={idx}
                className="bg-[#F5F1E8]/40 border border-[#EEE7DA] rounded-lg overflow-hidden transition-all hover:border-[#8E9490]/50"
              >
                {/* Header card click toggle */}
                <div 
                  onClick={() => toggleExpand(idx)}
                  className="p-4 flex items-center justify-between gap-4 cursor-pointer select-none"
                >
                  <div className="flex flex-col gap-1 min-w-0">
                    <span className="text-xs font-semibold text-[#1F2421] truncate max-w-lg" title={log.question}>
                      {log.question}
                    </span>
                    <div className="flex items-center gap-2 text-[10px] text-[#8E9490] font-medium">
                      <span className="font-mono">{new Date(log.timestamp).toLocaleTimeString()}</span>
                      <span>·</span>
                      <span className="capitalize">{log.model_used.replace('_', ' ')}</span>
                      <span>·</span>
                      <span className="flex items-center gap-0.5">
                        <Clock className="w-3 h-3 text-[#5A738E]" />
                        {log.latency_ms.toFixed(0)}ms
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 shrink-0">
                    {log.is_valid !== undefined && (
                      <span className={`text-[8.5px] uppercase font-bold tracking-wider px-2 py-0.5 rounded border ${
                        log.is_valid 
                          ? 'bg-[#6E8B7E]/5 border-[#6E8B7E]/30 text-[#6E8B7E]' 
                          : 'bg-[#A0522D]/5 border-[#A0522D]/30 text-[#A0522D]'
                      }`}>
                        {log.is_valid ? 'Valid Syntax' : 'Syntax Error'}
                      </span>
                    )}
                    {isExpanded ? <ChevronUp className="w-4 h-4 text-[#8E9490]" /> : <ChevronDown className="w-4 h-4 text-[#8E9490]" />}
                  </div>
                </div>

                {/* Collapsible Details Panel */}
                {isExpanded && (
                  <div className="border-t border-[#EEE7DA] p-4 bg-[#FAF8F2] space-y-4 text-xs">
                    {/* Generated SQL query */}
                    {log.sql_generated && (
                      <div className="space-y-1.5">
                        <span className="text-[10px] uppercase font-semibold text-[#8E9490] tracking-wider block">Generated SQL</span>
                        <pre className="bg-[#F5F1E8]/40 border border-[#EEE7DA] p-3 rounded font-mono text-[11px] text-[#1F2421] whitespace-pre-wrap break-all">
                          {log.sql_generated}
                        </pre>
                      </div>
                    )}

                    {/* Parser Error report */}
                    {log.parsing_errors && (
                      <div className="space-y-1.5">
                        <span className="text-[10px] uppercase font-semibold text-[#A0522D] tracking-wider block">Parsing Error details</span>
                        <div className="bg-[#A0522D]/5 border border-[#A0522D]/30 text-[#A0522D] p-3 rounded font-mono text-[11px]">
                          {log.parsing_errors}
                        </div>
                      </div>
                    )}

                    {/* Latency breakdown stats */}
                    {log.latency_breakdown && (
                      <div className="space-y-1.5">
                        <span className="text-[10px] uppercase font-semibold text-[#8E9490] tracking-wider block">Stage Latencies</span>
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 bg-[#F5F1E8]/20 border border-[#EEE7DA]/50 p-3 rounded">
                          {Object.entries(log.latency_breakdown).map(([stage, val]) => (
                            <div key={stage} className="space-y-0.5">
                              <span className="text-[9px] uppercase tracking-wide text-[#8E9490] block truncate">
                                {stage.replace('_ms', '').replace(/_/g, ' ')}
                              </span>
                              <span className="font-mono text-xs font-semibold text-[#5C625E]">{Math.round(val)}ms</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Raw JSON Payload */}
                    <div className="space-y-1.5">
                      <span className="text-[10px] uppercase font-semibold text-[#8E9490] tracking-wider block flex items-center gap-1">
                        <Terminal className="w-3.5 h-3.5" />
                        Raw Telemetry JSON
                      </span>
                      <pre className="bg-[#FAF8F2] border border-[#EEE7DA] p-3 rounded font-mono text-[10px] text-[#5C625E] overflow-x-auto whitespace-pre">
                        {JSON.stringify(log, null, 2)}
                      </pre>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

export default LogsPage

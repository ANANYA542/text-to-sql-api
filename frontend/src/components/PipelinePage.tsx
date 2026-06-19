import React, { useState, useEffect } from 'react'
import { useStore } from '../store/useStore'
import { Search as SearchIcon, Info, Network, Columns, HelpCircle } from 'lucide-react'

interface TableSchema {
  table_name: string
  columns: string[]
  relations: number
  description: string
}

const PipelinePage: React.FC = () => {
  const { selectedTable, setSelectedTable, executionResult, currentQuestion } = useStore()
  const [schemas, setSchemas] = useState<TableSchema[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadSchema = async () => {
      try {
        const res = await fetch('/api/schema')
        if (res.ok) {
          const data = await res.json()
          setSchemas(data)
          if (data.length > 0 && !selectedTable) {
            setSelectedTable(data[0].table_name)
          }
        }
      } catch (err) {
        console.error('Failed to load schema', err)
      } finally {
        setLoading(false)
      }
    }
    loadSchema()
  }, [selectedTable, setSelectedTable])

  const filteredSchemas = schemas.filter(s => 
    s.table_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.columns.some(c => c.toLowerCase().includes(searchQuery.toLowerCase()))
  )

  const activeSchema = schemas.find(s => s.table_name === selectedTable)

  // Find if selected table was scored in current execution result
  const queryDetail = executionResult?.details?.[selectedTable || '']
  const queryRankIdx = executionResult?.retrieved_tables?.indexOf(selectedTable || '') ?? -1
  const isSelectedTableInTop5 = queryRankIdx !== -1

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h2 className="text-2xl font-semibold tracking-tight text-slate-900">Retrieval Funnel</h2>
        <p className="text-xs text-slate-500 mt-1">
          Explore the 97-table schema space, foreign-key relationships, and learned reranker prediction features.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Side: Tables List */}
        <div className="lg:col-span-1 space-y-4">
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm space-y-4">
            <div className="relative">
              <SearchIcon className="absolute left-3 top-2.5 w-4 h-4 text-slate-400" />
              <input
                type="text"
                placeholder="Search tables or columns..."
                className="w-full pl-9 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>

            {loading ? (
              <div className="text-center py-12 text-xs text-slate-500">Loading schemas...</div>
            ) : (
              <div className="space-y-1.5 overflow-y-auto max-h-[500px] pr-1">
                {filteredSchemas.map((s) => {
                  const isSelected = selectedTable === s.table_name
                  const isInTop5 = executionResult?.retrieved_tables?.includes(s.table_name)
                  
                  return (
                    <button
                      key={s.table_name}
                      onClick={() => setSelectedTable(s.table_name)}
                      className={`w-full text-left px-3 py-2.5 rounded-lg transition-all flex flex-col gap-1 border ${
                        isSelected 
                          ? 'bg-blue-600 text-white border-blue-600 shadow-sm' 
                          : 'bg-slate-50/60 hover:bg-slate-100 text-slate-900 border-slate-200'
                      }`}
                    >
                      <div className="flex items-center justify-between w-full">
                        <span className="font-mono text-xs font-semibold truncate max-w-[170px]">{s.table_name}</span>
                        {isInTop5 && (
                          <span className={`text-[8px] uppercase tracking-wider font-semibold px-1 rounded ${
                            isSelected ? 'bg-white text-blue-600' : 'bg-emerald-50 text-emerald-700'
                          }`}>
                            Top 5
                          </span>
                        )}
                      </div>
                      <div className={`text-[10px] truncate max-w-[210px] ${isSelected ? 'text-white/80' : 'text-slate-500'}`}>
                        {s.columns.length} columns · {s.relations} relations
                      </div>
                    </button>
                  )
                })}
              </div>
            )}
          </div>
        </div>

        {/* Right Side: Details View */}
        <div className="lg:col-span-2">
          {activeSchema ? (
            <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6">
              {/* Table Identity */}
              <div className="border-b border-slate-100 pb-4 space-y-2">
                <div className="flex items-center gap-3">
                  <h3 className="text-lg font-mono font-semibold text-slate-900">{activeSchema.table_name}</h3>
                  <span className="bg-slate-100 px-2 py-0.5 rounded text-[10px] text-slate-600 font-medium font-mono">
                    Relations: {activeSchema.relations}
                  </span>
                </div>
                <p className="text-xs text-slate-600 leading-relaxed">
                  {activeSchema.description || "No description provided."}
                </p>
              </div>

              {/* Live Query Scores Overlay */}
              {queryDetail ? (
                <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-3.5">
                  <div className="flex items-center gap-2 text-xs font-semibold text-slate-900">
                    <Info className="w-4 h-4 text-blue-500" />
                    <span>ML Pipeline Relevance for Current Query</span>
                  </div>
                  <div className="text-[11px] text-slate-600">
                    <span className="font-semibold text-slate-700">Question:</span> "{currentQuestion}"
                  </div>

                  <div className="grid grid-cols-2 gap-4 pt-2 border-t border-slate-200">
                    <div className="space-y-1">
                      <span className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider block">Relevance Probability</span>
                      <span className="text-base font-mono font-bold text-emerald-600">
                        {Math.round(queryDetail.relevance_score * 100)}%
                      </span>
                    </div>
                    <div className="space-y-1">
                      <span className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider block">Funnel Position</span>
                      <span className="text-base font-mono font-bold text-slate-900">
                        {isSelectedTableInTop5 ? `#${queryRankIdx + 1} of 97` : 'Filtered Out (Rank > 5)'}
                      </span>
                    </div>
                  </div>

                  <div className="space-y-1 bg-white p-2.5 rounded-lg border border-slate-100">
                    <span className="text-[9px] uppercase font-bold text-slate-400 tracking-wider block">Decision Reason</span>
                    <p className="text-[10px] text-slate-600 italic leading-relaxed">
                      "{queryDetail.reason}"
                    </p>
                  </div>
                </div>
              ) : currentQuestion && (
                <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-xs text-slate-400 flex items-center gap-2">
                  <HelpCircle className="w-4 h-4 shrink-0 text-slate-400" />
                  <span>This table was not evaluated in the top 25 candidates for the current question.</span>
                </div>
              )}

              {/* Columns List */}
              <div className="space-y-3">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                  <Columns className="w-3.5 h-3.5" />
                  Columns ({activeSchema.columns.length})
                </h4>
                <div className="flex flex-wrap gap-2 max-h-60 overflow-y-auto pr-1">
                  {activeSchema.columns.map((col) => {
                    let typeClass = 'border-slate-200 text-slate-600 bg-slate-50'
                    if (col.toLowerCase().endsWith('_id') || col.toLowerCase().endsWith('_key')) {
                      typeClass = 'border-amber-200 text-amber-700 bg-amber-50'
                    } else if (col.toLowerCase().includes('count') || col.toLowerCase().includes('num') || col.toLowerCase().includes('total')) {
                      typeClass = 'border-blue-200 text-blue-700 bg-blue-50'
                    }
                    
                    return (
                      <span key={col} className={`border px-2.5 py-1 rounded-lg text-xs font-mono font-medium ${typeClass}`}>
                        {col}
                      </span>
                    )
                  })}
                </div>
              </div>

              {/* Table Relationships Graph Meta */}
              <div className="space-y-3">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                  <Network className="w-3.5 h-3.5" />
                  FK Relationships Degree
                </h4>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Connected to <span className="font-semibold text-slate-900">{activeSchema.relations}</span> other tables in the physical schema graph. Relationship degree is propagated during stage 2 retrieval boosts.
                </p>
              </div>
            </div>
          ) : (
            <div className="bg-white border border-slate-200 rounded-xl p-12 text-center text-xs text-slate-400 shadow-sm">
              Select a table from the sidebar list to inspect schemas.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default PipelinePage

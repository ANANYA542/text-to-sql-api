import React, { useState, useEffect } from 'react'
import { useStore } from '../store/useStore'
import { Search as SearchIcon, Columns, Network, Activity, Clock, Compass, HelpCircle, Info } from 'lucide-react'

interface TableSchema {
  table_name: string
  columns: string[]
  relations: number
  description: string
}

const PipelinePage: React.FC = () => {
  const { selectedTable, setSelectedTable, executionResult, currentQuestion, hybridAlpha } = useStore()
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
  const queryDetail = executionResult?.details?.[selectedTable || '']
  const queryRankIdx = executionResult?.retrieved_tables?.indexOf(selectedTable || '') ?? -1
  const isSelectedTableInTop5 = queryRankIdx !== -1

  // Dynamic phase timings or fallback to figma prototype defaults
  const timeMs = executionResult?.latency_breakdown || {
    expand_ms: 23,
    retrieval_ms: 89,
    reranking_ms: 142,
    generation_ms: 891
  }

  const expandVal = timeMs.expand_ms || (timeMs.retrieval_ms ? timeMs.retrieval_ms * 0.1 : 23)
  const retrievalVal = timeMs.retrieval_ms ? timeMs.retrieval_ms * 0.9 : 89
  const rerankVal = timeMs.reranking_ms || 142
  const genVal = timeMs.generation_ms || 891
  const totalVal = expandVal + retrievalVal + rerankVal + genVal

  const getPercentage = (val: number) => {
    return Math.max(2, Math.round((val / totalVal) * 100))
  }

  return (
    <div className="space-y-8">
      {/* Editorial Title */}
      <div>
        <h2 className="text-xl font-bold text-[#0F172A] tracking-tight">Retrieval Pipeline</h2>
        <p className="text-xs text-slate-500 mt-0.5">
          Timing metrics across query expansion, vector search, cross-encoder reranking, and generation.
        </p>
      </div>

      {/* Latency Phase Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        {/* Stage 1 */}
        <div className="bg-white border border-slate-200/80 rounded-xl p-4 shadow-sm space-y-2.5">
          <div className="flex justify-between items-center text-[10px] uppercase font-bold text-slate-400">
            <span>Stage 1: Query Expansion</span>
            <Activity className="w-3.5 h-3.5 text-emerald-500" />
          </div>
          <div>
            <div className="text-lg font-bold text-slate-800 font-mono">{Math.round(expandVal)}ms</div>
            <p className="text-[10px] text-slate-400 mt-0.5 leading-relaxed">
              Expands NL query with domain synonyms and entity recognition to improve recall.
            </p>
          </div>
          <div className="text-[9px] text-slate-400 border-t border-slate-50 pt-2 font-mono flex justify-between">
            <span>P95 Limit: 92ms</span>
            <span>Scale: T5-Small</span>
          </div>
        </div>

        {/* Stage 2 */}
        <div className="bg-white border border-slate-200/80 rounded-xl p-4 shadow-sm space-y-2.5">
          <div className="flex justify-between items-center text-[10px] uppercase font-bold text-slate-400">
            <span>Stage 2: Hybrid Search</span>
            <Compass className="w-3.5 h-3.5 text-blue-500" />
          </div>
          <div>
            <div className="text-lg font-bold text-slate-800 font-mono">{Math.round(retrievalVal)}ms</div>
            <p className="text-[10px] text-slate-400 mt-0.5 leading-relaxed">
              Fuses BM25 & dense vector search. α={hybridAlpha} balances lexical and semantic signals.
            </p>
          </div>
          <div className="text-[9px] text-slate-400 border-t border-slate-50 pt-2 font-mono flex justify-between">
            <span>P95 Limit: 356ms</span>
            <span>BGE-Small-v1.5</span>
          </div>
        </div>

        {/* Stage 3 */}
        <div className="bg-white border border-slate-200/80 rounded-xl p-4 shadow-sm space-y-2.5">
          <div className="flex justify-between items-center text-[10px] uppercase font-bold text-slate-400">
            <span>Stage 3: ML Reranking</span>
            <Network className="w-3.5 h-3.5 text-purple-500" />
          </div>
          <div>
            <div className="text-lg font-bold text-slate-800 font-mono">{Math.round(rerankVal)}ms</div>
            <p className="text-[10px] text-slate-400 mt-0.5 leading-relaxed">
              Cross-encoder scores candidate table pairs. LightGBM ranker applies learned feature weights.
            </p>
          </div>
          <div className="text-[9px] text-slate-400 border-t border-slate-50 pt-2 font-mono flex justify-between">
            <span>P95 Limit: 256ms</span>
            <span>Ranker v3 Active</span>
          </div>
        </div>

        {/* Stage 4 */}
        <div className="bg-white border border-slate-200/80 rounded-xl p-4 shadow-sm space-y-2.5">
          <div className="flex justify-between items-center text-[10px] uppercase font-bold text-slate-400">
            <span>Stage 4: SQL Generation</span>
            <Clock className="w-3.5 h-3.5 text-orange-500" />
          </div>
          <div>
            <div className="text-lg font-bold text-slate-800 font-mono">{Math.round(genVal)}ms</div>
            <p className="text-[10px] text-slate-400 mt-0.5 leading-relaxed">
              Schema-augmented generation with retrieved tables injected into system context.
            </p>
          </div>
          <div className="text-[9px] text-slate-400 border-t border-slate-50 pt-2 font-mono flex justify-between">
            <span>P95 Limit: 1604ms</span>
            <span>LLaMA 3.1 70B</span>
          </div>
        </div>
      </div>

      {/* Stacked Timing Breakdown Graph */}
      <div className="bg-white border border-slate-200/80 rounded-xl p-5 shadow-sm space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">End-to-End Pipeline Latency Breakdown</span>
          <span className="text-[10px] font-mono font-semibold text-slate-700">Total: {Math.round(totalVal)}ms</span>
        </div>
        
        {/* Horizontal Stacked Bar */}
        <div className="w-full h-3 rounded-full overflow-hidden flex bg-slate-100">
          <div className="h-full bg-emerald-500 transition-all duration-300" style={{ width: `${getPercentage(expandVal)}%` }} title={`Expansion: ${Math.round(expandVal)}ms`} />
          <div className="h-full bg-blue-500 transition-all duration-300" style={{ width: `${getPercentage(retrievalVal)}%` }} title={`Search: ${Math.round(retrievalVal)}ms`} />
          <div className="h-full bg-purple-500 transition-all duration-300" style={{ width: `${getPercentage(rerankVal)}%` }} title={`Reranking: ${Math.round(rerankVal)}ms`} />
          <div className="h-full bg-orange-500 transition-all duration-300" style={{ width: `${getPercentage(genVal)}%` }} title={`SQL Generation: ${Math.round(genVal)}ms`} />
        </div>

        {/* Legend */}
        <div className="flex flex-wrap gap-x-6 gap-y-2 pt-1.5 text-[10px] text-slate-500 justify-center">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <span>Expansion ({getPercentage(expandVal)}%)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-blue-500" />
            <span>Hybrid Search ({getPercentage(retrievalVal)}%)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-purple-500" />
            <span>Reranking ({getPercentage(rerankVal)}%)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-orange-500" />
            <span>SQL Generation ({getPercentage(genVal)}%)</span>
          </div>
        </div>
      </div>

      {/* Database Table Schema Registry Split Layout */}
      <div className="space-y-4">
        <div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">Database Table Registry</h3>
          <p className="text-[10px] text-slate-500 mt-0.5">Explore physical column layouts, primary/foreign constraints, and data descriptions.</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Tables Search / Selector (4 Cols) */}
          <div className="lg:col-span-4 bg-white border border-slate-200/80 rounded-xl p-4 shadow-sm space-y-4">
            <div className="relative">
              <SearchIcon className="absolute left-3 top-2.5 w-3.5 h-3.5 text-slate-400" />
              <input
                type="text"
                placeholder="Search tables or fields..."
                className="w-full pl-9 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>

            {loading ? (
              <div className="text-center py-12 text-xs text-slate-500">Loading tables...</div>
            ) : (
              <div className="space-y-1.5 overflow-y-auto max-h-[360px] pr-1">
                {filteredSchemas.map((s) => {
                  const isSelected = selectedTable === s.table_name
                  const isInTop5 = executionResult?.retrieved_tables?.includes(s.table_name)
                  
                  return (
                    <button
                      key={s.table_name}
                      onClick={() => setSelectedTable(s.table_name)}
                      className={`w-full text-left px-3 py-2 rounded-lg transition-all border ${
                        isSelected 
                          ? 'bg-[#2563EB] text-white border-blue-600 shadow-sm' 
                          : 'bg-slate-50/60 hover:bg-slate-100 text-slate-700 border-slate-200'
                      }`}
                    >
                      <div className="flex items-center justify-between w-full">
                        <span className="font-mono text-[10px] font-bold truncate max-w-[190px]">{s.table_name}</span>
                        {isInTop5 && (
                          <span className={`text-[8px] uppercase tracking-wider font-semibold px-1 rounded ${
                            isSelected ? 'bg-white text-blue-600' : 'bg-emerald-50 text-emerald-700'
                          }`}>
                            Top 5
                          </span>
                        )}
                      </div>
                      <div className={`text-[9px] mt-0.5 ${isSelected ? 'text-white/80' : 'text-slate-400'}`}>
                        {s.columns.length} columns · {s.relations} links
                      </div>
                    </button>
                  )
                })}
              </div>
            )}
          </div>

          {/* Table Properties Panel (8 Cols) */}
          <div className="lg:col-span-8">
            {activeSchema ? (
              <div className="bg-white border border-slate-200/80 rounded-xl p-5 shadow-sm space-y-5">
                <div className="border-b border-slate-100 pb-3 space-y-1">
                  <div className="flex items-center gap-3">
                    <h3 className="text-sm font-mono font-bold text-slate-800">{activeSchema.table_name}</h3>
                    <span className="bg-slate-100 px-2 py-0.5 rounded text-[9px] text-slate-500 font-semibold font-mono">
                      Degree: {activeSchema.relations}
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 leading-relaxed">
                    {activeSchema.description || "Physical warehouse table indexed within the Beaver framework."}
                  </p>
                </div>

                {queryDetail ? (
                  <div className="bg-slate-50 border border-slate-200 rounded-lg p-3.5 space-y-2">
                    <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-800">
                      <Info className="w-3.5 h-3.5 text-blue-500" />
                      <span>LTR Retrieval Weighting</span>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-[10px]">
                      <div>
                        <span className="text-slate-400 block">Relevance Logit Score</span>
                        <span className="font-mono font-bold text-emerald-600 text-sm">
                          {(queryDetail.relevance_score * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-400 block">Funnel Decision</span>
                        <span className="font-bold text-slate-700">
                          {isSelectedTableInTop5 ? `Selected (Rank #${queryRankIdx + 1})` : 'Pruned / Filtered'}
                        </span>
                      </div>
                    </div>
                    <p className="text-[10px] text-slate-500 italic bg-white p-2 border border-slate-100 rounded leading-relaxed">
                      "{queryDetail.reason}"
                    </p>
                  </div>
                ) : currentQuestion && (
                  <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-[10px] text-slate-400 flex items-center gap-2">
                    <HelpCircle className="w-3.5 h-3.5 text-slate-400" />
                    <span>Table filtered during candidate retrieval.</span>
                  </div>
                )}

                <div className="space-y-2">
                  <h4 className="text-[10px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                    <Columns className="w-3.5 h-3.5" />
                    Columns Properties ({activeSchema.columns.length})
                  </h4>
                  <div className="flex flex-wrap gap-2 max-h-48 overflow-y-auto pr-1">
                    {activeSchema.columns.map((col) => {
                      let typeClass = 'border-slate-200 text-slate-600 bg-slate-50'
                      if (col.toLowerCase().endsWith('_id') || col.toLowerCase().endsWith('_key')) {
                        typeClass = 'border-amber-200 text-amber-700 bg-amber-50'
                      } else if (col.toLowerCase().includes('count') || col.toLowerCase().includes('num') || col.toLowerCase().includes('total')) {
                        typeClass = 'border-blue-200 text-blue-700 bg-blue-50'
                      }
                      
                      return (
                        <span key={col} className={`border px-2 py-0.5 rounded text-[10px] font-mono font-medium ${typeClass}`}>
                          {col}
                        </span>
                      )
                    })}
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-white border border-slate-200/80 rounded-xl p-12 text-center text-xs text-slate-400 shadow-sm">
                Select a database table to inspect structural schema metadata.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default PipelinePage

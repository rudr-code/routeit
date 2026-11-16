import React, { useState, useEffect, useCallback } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Bar,
} from 'recharts';
import { api } from '../../api/client';
import {
  Cpu,
  Zap,
  BarChart3,
  TrendingUp,
  Award,
  Clock,
  Shield,
  Activity,
  RotateCw,
} from '../Icons/LucideIcons';

interface AlgorithmResult {
  algorithm: string;
  display_name: string;
  is_feasible: boolean;
  num_vehicles_used: number;
  total_cost: number;
  total_distance_km: number;
  total_duration_min: number;
  traffic_delay_min: number;
  operational_cost_inr: number;
  execution_time_ms: number;
  convergence_history: Array<{ iter: number; cost: number }>;
  relative_distance_diff_pct: number;
  relative_cost_diff_pct: number;
}

const ALGO_COLORS: Record<string, string> = {
  qpso: '#6366f1', // Indigo
  pso: '#f59e0b',  // Amber
  ga: '#10b981',   // Emerald
  dijkstra: '#06b6d4', // Cyan
  ortools: '#ef4444', // Rose / Red
};

export function BenchmarkPortal() {
  const [loading, setLoading] = useState<boolean>(false);
  const [scalabilityLoading, setScalabilityLoading] = useState<boolean>(false);
  const [benchmarkData, setBenchmarkData] = useState<any | null>(null);
  const [scalabilityData, setScalabilityData] = useState<any[] | null>(null);
  const [selectedAlgos, setSelectedAlgos] = useState<string[]>([
    'qpso',
    'pso',
    'ga',
    'dijkstra',
    'ortools',
  ]);
  const [swarmSize, setSwarmSize] = useState<number>(30);
  const [iterations, setIterations] = useState<number>(35);
  const [activeTab, setActiveTab] = useState<'comparison' | 'convergence' | 'scalability'>('comparison');
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  // Initial load
  useEffect(() => {
    loadLatest();
  }, []);

  const loadLatest = async () => {
    setLoading(true);
    setStatusMsg('Loading latest empirical benchmark metrics...');
    try {
      const data = await api.getLatestBenchmark();
      setBenchmarkData(data);
      setStatusMsg(null);
    } catch (err: any) {
      setStatusMsg(`Notice: ${err.message || 'Could not load pre-cached benchmark'}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRunBenchmark = async () => {
    setLoading(true);
    setStatusMsg('Executing multi-algorithm comparison across QPSO, PSO, GA, Dijkstra, and OR-Tools...');
    try {
      const data = await api.runBenchmark({
        algorithms: selectedAlgos,
        num_particles_or_pop: swarmSize,
        max_iters_or_gen: iterations,
        seed: 42,
      });
      setBenchmarkData(data);
      setStatusMsg('Benchmark execution finished successfully!');
      setTimeout(() => setStatusMsg(null), 4000);
    } catch (err: any) {
      setStatusMsg(`Error running benchmark: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadScalability = async () => {
    setScalabilityLoading(true);
    setActiveTab('scalability');
    try {
      const data = await api.getScalability();
      setScalabilityData(data);
    } catch (err: any) {
      setStatusMsg(`Error loading scalability: ${err.message}`);
    } finally {
      setScalabilityLoading(false);
    }
  };

  const toggleAlgo = (algo: string) => {
    setSelectedAlgos((prev) =>
      prev.includes(algo) ? prev.filter((a) => a !== algo) : [...prev, algo]
    );
  };

  // Prepare convergence multi-line chart dataset
  const prepareConvergenceChart = () => {
    if (!benchmarkData || !benchmarkData.results) return [];
    const maxIters = iterations;
    const chartPoints: any[] = [];

    for (let t = 1; t <= maxIters; t++) {
      const row: any = { iteration: t };
      for (const res of benchmarkData.results) {
        if (!res.convergence_history || res.convergence_history.length === 0) continue;
        // Find nearest point
        const pt = res.convergence_history.find((p: any) => p.iter === t);
        if (pt) {
          row[res.algorithm] = pt.cost;
        } else {
          // Nearest preceding
          const prevPts = res.convergence_history.filter((p: any) => p.iter <= t);
          if (prevPts.length > 0) {
            row[res.algorithm] = prevPts[prevPts.length - 1].cost;
          }
        }
      }
      chartPoints.push(row);
    }
    return chartPoints;
  };

  // Prepare bar chart dataset
  const barChartData = (benchmarkData?.results || []).map((r: AlgorithmResult) => ({
    name: r.display_name.split(' (')[0],
    algorithm: r.algorithm,
    distance_km: r.total_distance_km,
    duration_min: r.total_duration_min,
    traffic_delay_min: r.traffic_delay_min,
    operational_cost_inr: r.operational_cost_inr,
    latency_ms: r.execution_time_ms,
  }));

  const qpsoResult = benchmarkData?.results?.find((r: AlgorithmResult) => r.algorithm === 'qpso');
  const psoResult = benchmarkData?.results?.find((r: AlgorithmResult) => r.algorithm === 'pso');

  return (
    <div className="portal-main" style={{ padding: '24px 32px', height: 'calc(100vh - 54px)', overflowY: 'auto' }}>
      {/* Top Banner */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 20 }}>
        <div>
          <h2 style={{ margin: '0 0 6px 0', fontSize: 24, fontWeight: 700, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: 10 }}>
            <Cpu size={26} color="#818cf8" />
            Algorithm Benchmarking & Analytics Studio
          </h2>
          <p style={{ margin: 0, color: '#94a3b8', fontSize: 14 }}>
            Multi-objective optimization comparison between QPSO, Classical PSO, Genetic Algorithm (GA), Dijkstra Network Heuristic, and Google OR-Tools.
          </p>
        </div>

        <div style={{ display: 'flex', gap: 10 }}>
          <button
            onClick={handleRunBenchmark}
            disabled={loading}
            style={{
              padding: '10px 18px',
              backgroundColor: '#6366f1',
              color: '#fff',
              border: 'none',
              borderRadius: 6,
              fontWeight: 600,
              cursor: loading ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              boxShadow: '0 2px 8px rgba(99, 102, 241, 0.4)',
            }}
          >
            <Zap size={16} />
            {loading ? 'Solving Algorithms...' : 'Run Live Benchmark'}
          </button>

          <button
            onClick={handleLoadScalability}
            disabled={scalabilityLoading}
            style={{
              padding: '10px 16px',
              backgroundColor: '#1e293b',
              color: '#94a3b8',
              border: '1px solid #334155',
              borderRadius: 6,
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}
          >
            <TrendingUp size={16} />
            {scalabilityLoading ? 'Evaluating Scalability...' : 'Scalability Analysis'}
          </button>
        </div>
      </div>

      {statusMsg && (
        <div style={{ padding: '10px 16px', backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: 6, color: '#38bdf8', fontSize: 13, marginBottom: 18 }}>
          {statusMsg}
        </div>
      )}

      {/* Control Strip */}
      <div className="card-box" style={{ padding: '16px 20px', marginBottom: 20, display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: '#cbd5e1' }}>Algorithms:</span>
          {[
            { id: 'qpso', label: 'QPSO (Quantum)' },
            { id: 'pso', label: 'Classical PSO' },
            { id: 'ga', label: 'Genetic Algorithm' },
            { id: 'dijkstra', label: 'Dijkstra Heuristic' },
            { id: 'ortools', label: 'Google OR-Tools' },
          ].map((item) => (
            <label key={item.id} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, color: '#e2e8f0', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={selectedAlgos.includes(item.id)}
                onChange={() => toggleAlgo(item.id)}
                style={{ accentColor: ALGO_COLORS[item.id] || '#6366f1' }}
              />
              <span style={{ color: selectedAlgos.includes(item.id) ? ALGO_COLORS[item.id] : '#64748b', fontWeight: selectedAlgos.includes(item.id) ? 600 : 400 }}>
                {item.label}
              </span>
            </label>
          ))}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 18 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 12, color: '#94a3b8' }}>Swarm/Pop:</span>
            <input
              type="number"
              min={10}
              max={100}
              value={swarmSize}
              onChange={(e) => setSwarmSize(Number(e.target.value))}
              style={{ width: 60, padding: '4px 8px', backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: 4, color: '#fff', fontSize: 12 }}
            />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 12, color: '#94a3b8' }}>Iterations:</span>
            <input
              type="number"
              min={10}
              max={200}
              value={iterations}
              onChange={(e) => setIterations(Number(e.target.value))}
              style={{ width: 60, padding: '4px 8px', backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: 4, color: '#fff', fontSize: 12 }}
            />
          </div>
        </div>
      </div>

      {/* KPI Cards Strip */}
      {benchmarkData && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 20 }}>
          <div className="card-box" style={{ padding: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: 12, marginBottom: 6 }}>
              <span>TOP ALGORITHM</span>
              <Award size={16} color="#f59e0b" />
            </div>
            <div style={{ fontSize: 20, fontWeight: 700, color: '#38bdf8' }}>
              {benchmarkData.best_algorithm.toUpperCase()}
            </div>
            <div style={{ fontSize: 11, color: '#10b981', marginTop: 4 }}>
              Near-optimal multi-objective cost
            </div>
          </div>

          <div className="card-box" style={{ padding: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: 12, marginBottom: 6 }}>
              <span>QPSO DISTANCE</span>
              <TrendingUp size={16} color="#6366f1" />
            </div>
            <div style={{ fontSize: 20, fontWeight: 700, color: '#e2e8f0' }}>
              {qpsoResult ? `${qpsoResult.total_distance_km} km` : 'N/A'}
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
              {qpsoResult && psoResult ? `${Math.abs(qpsoResult.relative_distance_diff_pct)}% vs classical PSO` : 'Simulated road graph'}
            </div>
          </div>

          <div className="card-box" style={{ padding: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: 12, marginBottom: 6 }}>
              <span>FLEET DISPATCH COST</span>
              <Shield size={16} color="#10b981" />
            </div>
            <div style={{ fontSize: 20, fontWeight: 700, color: '#10b981' }}>
              {qpsoResult ? `₹${qpsoResult.operational_cost_inr}` : 'N/A'}
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
              Fixed dispatch + ₹15/km distance
            </div>
          </div>

          <div className="card-box" style={{ padding: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: 12, marginBottom: 6 }}>
              <span>SOLVE LATENCY</span>
              <Clock size={16} color="#ec4899" />
            </div>
            <div style={{ fontSize: 20, fontWeight: 700, color: '#f43f5e' }}>
              {qpsoResult ? `${qpsoResult.execution_time_ms} ms` : 'N/A'}
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
              Numba-accelerated evaluation
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 12, borderBottom: '1px solid #334155', marginBottom: 18 }}>
        <button
          className={`portal-tab-btn ${activeTab === 'comparison' ? 'active' : ''}`}
          onClick={() => setActiveTab('comparison')}
          style={{ padding: '8px 16px', background: 'none', border: 'none', color: activeTab === 'comparison' ? '#6366f1' : '#94a3b8', borderBottom: activeTab === 'comparison' ? '2px solid #6366f1' : 'none', fontWeight: 600, cursor: 'pointer' }}
        >
          📊 Head-to-Head Comparison
        </button>
        <button
          className={`portal-tab-btn ${activeTab === 'convergence' ? 'active' : ''}`}
          onClick={() => setActiveTab('convergence')}
          style={{ padding: '8px 16px', background: 'none', border: 'none', color: activeTab === 'convergence' ? '#6366f1' : '#94a3b8', borderBottom: activeTab === 'convergence' ? '2px solid #6366f1' : 'none', fontWeight: 600, cursor: 'pointer' }}
        >
          📈 Convergence Curves
        </button>
        <button
          className={`portal-tab-btn ${activeTab === 'scalability' ? 'active' : ''}`}
          onClick={() => setActiveTab('scalability')}
          style={{ padding: '8px 16px', background: 'none', border: 'none', color: activeTab === 'scalability' ? '#6366f1' : '#94a3b8', borderBottom: activeTab === 'scalability' ? '2px solid #6366f1' : 'none', fontWeight: 600, cursor: 'pointer' }}
        >
          ⚡ Scalability Scaling (10-100 Nodes)
        </button>
      </div>

      {/* Tab 1: Comparison Overview */}
      {activeTab === 'comparison' && (
        <div>
          {/* Chart Section */}
          <div className="card-box" style={{ padding: 20, marginBottom: 20 }}>
            <h3 style={{ margin: '0 0 16px 0', fontSize: 15, color: '#e2e8f0', display: 'flex', alignItems: 'center', gap: 8 }}>
              <BarChart3 size={18} color="#6366f1" />
              Multi-Objective Metrics Comparison
            </h3>
            <div style={{ height: 320, width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={barChartData} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                  <YAxis stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 6, color: '#fff' }} />
                  <Legend wrapperStyle={{ paddingTop: 10 }} />
                  <Bar dataKey="distance_km" name="Total Distance (km)" fill="#6366f1" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="duration_min" name="Travel Time (min)" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="traffic_delay_min" name="Congestion Delay (min)" fill="#ef4444" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Detailed Table */}
          <div className="card-box" style={{ padding: 20 }}>
            <h3 style={{ margin: '0 0 16px 0', fontSize: 15, color: '#e2e8f0', display: 'flex', alignItems: 'center', gap: 8 }}>
              <Activity size={18} color="#10b981" />
              Detailed Benchmark Ledger
            </h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #334155', color: '#94a3b8' }}>
                    <th style={{ padding: '10px 12px' }}>Algorithm</th>
                    <th style={{ padding: '10px 12px' }}>Feasibility</th>
                    <th style={{ padding: '10px 12px' }}>Vehicles</th>
                    <th style={{ padding: '10px 12px' }}>Distance (km)</th>
                    <th style={{ padding: '10px 12px' }}>Duration (min)</th>
                    <th style={{ padding: '10px 12px' }}>Traffic Delay</th>
                    <th style={{ padding: '10px 12px' }}>Fleet Cost (₹)</th>
                    <th style={{ padding: '10px 12px' }}>Solve Time (ms)</th>
                    <th style={{ padding: '10px 12px' }}>vs Baseline</th>
                  </tr>
                </thead>
                <tbody>
                  {(benchmarkData?.results || []).map((r: AlgorithmResult) => (
                    <tr key={r.algorithm} style={{ borderBottom: '1px solid #1e293b' }}>
                      <td style={{ padding: '12px 12px', fontWeight: 600, color: ALGO_COLORS[r.algorithm] || '#fff' }}>
                        {r.display_name}
                      </td>
                      <td style={{ padding: '12px 12px' }}>
                        <span style={{
                          padding: '2px 8px',
                          borderRadius: 4,
                          fontSize: 11,
                          fontWeight: 600,
                          backgroundColor: r.is_feasible ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                          color: r.is_feasible ? '#10b981' : '#ef4444',
                        }}>
                          {r.is_feasible ? 'FEASIBLE' : 'VIOLATION'}
                        </span>
                      </td>
                      <td style={{ padding: '12px 12px', color: '#cbd5e1' }}>{r.num_vehicles_used}</td>
                      <td style={{ padding: '12px 12px', color: '#cbd5e1', fontWeight: 600 }}>{r.total_distance_km}</td>
                      <td style={{ padding: '12px 12px', color: '#cbd5e1' }}>{r.total_duration_min}</td>
                      <td style={{ padding: '12px 12px', color: '#f87171' }}>+{r.traffic_delay_min} min</td>
                      <td style={{ padding: '12px 12px', color: '#10b981', fontWeight: 600 }}>₹{r.operational_cost_inr}</td>
                      <td style={{ padding: '12px 12px', color: '#cbd5e1' }}>{r.execution_time_ms} ms</td>
                      <td style={{ padding: '12px 12px', color: r.relative_cost_diff_pct < 0 ? '#10b981' : (r.relative_cost_diff_pct === 0 ? '#94a3b8' : '#f87171') }}>
                        {r.relative_cost_diff_pct < 0 ? `${r.relative_cost_diff_pct}%` : (r.relative_cost_diff_pct === 0 ? '0.0% (Base)' : `+${r.relative_cost_diff_pct}%`)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Convergence Curves */}
      {activeTab === 'convergence' && (
        <div className="card-box" style={{ padding: 24 }}>
          <h3 style={{ margin: '0 0 8px 0', fontSize: 16, color: '#e2e8f0', display: 'flex', alignItems: 'center', gap: 8 }}>
            <TrendingUp size={20} color="#6366f1" />
            Convergence Trajectory (Objective Fitness vs Iterations)
          </h3>
          <p style={{ margin: '0 0 20px 0', fontSize: 13, color: '#94a3b8' }}>
            Observes the quantum delta-potential well contraction in QPSO versus classical velocity-clamped PSO and Genetic Algorithm exploration.
          </p>
          <div style={{ height: 420, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={prepareConvergenceChart()} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="iteration" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} label={{ value: 'Iterations / Generations', position: 'insideBottom', offset: -10, fill: '#64748b' }} />
                <YAxis stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} label={{ value: 'Best Objective Cost', angle: -90, position: 'insideLeft', fill: '#64748b' }} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 6, color: '#fff' }} />
                <Legend wrapperStyle={{ paddingTop: 10 }} />
                {selectedAlgos.includes('qpso') && (
                  <Line type="monotone" dataKey="qpso" name="QPSO (Quantum)" stroke="#6366f1" strokeWidth={3} dot={false} />
                )}
                {selectedAlgos.includes('pso') && (
                  <Line type="monotone" dataKey="pso" name="Classical PSO" stroke="#f59e0b" strokeWidth={2} dot={false} />
                )}
                {selectedAlgos.includes('ga') && (
                  <Line type="monotone" dataKey="ga" name="Genetic Algorithm" stroke="#10b981" strokeWidth={2} dot={false} />
                )}
                {selectedAlgos.includes('ortools') && (
                  <Line type="stepAfter" dataKey="ortools" name="Google OR-Tools" stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Tab 3: Scalability Scaling */}
      {activeTab === 'scalability' && (
        <div className="card-box" style={{ padding: 24 }}>
          <h3 style={{ margin: '0 0 8px 0', fontSize: 16, color: '#e2e8f0', display: 'flex', alignItems: 'center', gap: 8 }}>
            <Cpu size={20} color="#10b981" />
            Scalability Analysis: 10, 25, 50, and 100 Delivery Nodes
          </h3>
          <p style={{ margin: '0 0 20px 0', fontSize: 13, color: '#94a3b8' }}>
            Empirical runtime latency scaling as customer density increases from suburban clusters to city-wide fleet routing.
          </p>

          {scalabilityData ? (
            <div style={{ height: 380, width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={scalabilityData} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="num_nodes" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} label={{ value: 'Number of Delivery Nodes', position: 'insideBottom', offset: -10, fill: '#64748b' }} />
                  <YAxis stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} label={{ value: 'Solve Latency (ms)', angle: -90, position: 'insideLeft', fill: '#64748b' }} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 6, color: '#fff' }} />
                  <Legend wrapperStyle={{ paddingTop: 10 }} />
                  <Line type="monotone" dataKey="qpso_ms" name="QPSO Latency (ms)" stroke="#6366f1" strokeWidth={3} />
                  <Line type="monotone" dataKey="pso_ms" name="PSO Latency (ms)" stroke="#f59e0b" strokeWidth={2} />
                  <Line type="monotone" dataKey="ga_ms" name="GA Latency (ms)" stroke="#10b981" strokeWidth={2} />
                  <Line type="monotone" dataKey="ortools_ms" name="OR-Tools Latency (ms)" stroke="#ef4444" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '40px 0' }}>
              <button
                onClick={handleLoadScalability}
                disabled={scalabilityLoading}
                style={{
                  padding: '12px 24px',
                  backgroundColor: '#6366f1',
                  color: '#fff',
                  border: 'none',
                  borderRadius: 6,
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                {scalabilityLoading ? 'Evaluating Scalability Matrix...' : 'Run Scalability Benchmark Profile'}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

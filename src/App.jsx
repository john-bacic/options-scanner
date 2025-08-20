import React, { useEffect, useState, useRef } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip as ReTooltip, ResponsiveContainer, CartesianGrid, ReferenceLine } from "recharts";

// Backend API base (for Render/Vercel). Prefer env; fallback to same-origin in non-localhost; else localhost.
const API_BASE = (() => {
  const env = import.meta?.env?.VITE_API_BASE;
  if (env && String(env).trim()) return String(env).replace(/\/$/, "");
  if (typeof window !== 'undefined') {
    const { origin, hostname } = window.location || {};
    if (hostname && hostname !== 'localhost' && hostname !== '127.0.0.1') {
      return String(origin).replace(/\/$/, "");
    }
  }
  return "http://localhost:8000";
})();
const apiUrl = (path) => `${API_BASE}${path.startsWith('/') ? '' : '/'}${path}`;

// Tooltip for mini line chart that shows full year and selected timeframe
function MiniTooltip({ active, payload, label, range, currency = "USD", rate = 1 }) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0]?.payload || {};
  const d = p.dateFull ? new Date(p.dateFull) : null;
  const dateStr = d && !isNaN(d.getTime())
    ? d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "2-digit" })
    : (p.dateFull || label || "");
  const priceVal = typeof p.close === 'number' && isFinite(p.close) ? Number(p.close) : null;
  const currCode = currency === "CAD" ? "CAD" : "USD";
  const disp = priceVal != null ? (currCode === "CAD" ? priceVal * Number(rate) : priceVal) : null;
  const price = disp != null
    ? new Intl.NumberFormat("en-US", { style: "currency", currency: currCode, minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(disp)
    : "";
  return (
    <div className="rounded border border-gray-200 bg-white p-2 text-xs shadow">
      <div className="flex items-center justify-between gap-2 mb-1">
        <div className="font-medium text-gray-800">{dateStr}</div>
        {range ? <span className="px-1.5 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200">{range}</span> : null}
      </div>
      <div className="text-gray-600">Close: <span className="font-semibold text-gray-800">{price}</span></div>
    </div>
  );
}

// Custom X-axis tick: align first label to start (Y-axis) and last to end to avoid clipping
function CustomXAxisTick({ x, y, payload, first = false, last = false }) {
  const label = formatDateMMDDYY(payload?.value);
  const anchor = first ? 'start' : (last ? 'end' : 'middle');
  const dx = last ? 10 : 0; // more right nudge for rightmost tick
  return (
    <g transform={`translate(${x},${y})`}>
      <text dy={9} dx={dx} textAnchor={anchor} fill="#6b7280" fontSize={10}>{label}</text>
    </g>
  );
}

// Minimal responsive candlestick mini chart using SVG
const CandleMiniChart = ({ data }) => {
  if (!Array.isArray(data) || data.length === 0) return null;
  const w = 600;
  const h = 160;
  const padL = 30;
  const padR = 10;
  const padT = 10;
  const padB = 20;
  const innerW = w - padL - padR;
  const innerH = h - padT - padB;

  const highs = data.map(d => Number(d.high));
  const lows = data.map(d => Number(d.low));
  const yMax = Math.max(...highs.filter(Number.isFinite));
  const yMin = Math.min(...lows.filter(Number.isFinite));
  const yRange = yMax - yMin || 1;

  const xStep = innerW / Math.max(data.length, 1);
  const candleW = Math.max(2, Math.min(10, xStep * 0.6));

  const yScale = (v) => padT + innerH - ((v - yMin) / yRange) * innerH;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="w-full h-full">
      {/* Y axis grid lines (light) */}
      {[0, 0.25, 0.5, 0.75, 1].map((t, i) => {
        const y = padT + innerH * (1 - t);
        return <line key={i} x1={padL} y1={y} x2={w - padR} y2={y} stroke="#eee" strokeDasharray="3 3" />;
      })}
      {/* Candles */}
      {data.map((d, i) => {
        const xCenter = padL + i * xStep + xStep / 2;
        const o = Number(d.open), c = Number(d.close), hi = Number(d.high), lo = Number(d.low);
        const isUp = c >= o;
        const color = isUp ? "#10b981" : "#ef4444"; // green-500 / red-500
        const yO = yScale(o);
        const yC = yScale(c);
        const yH = yScale(hi);
        const yL = yScale(lo);
        const top = Math.min(yO, yC);
        const bodyH = Math.max(1, Math.abs(yC - yO));
        return (
          <g key={i}>
            <line x1={xCenter} x2={xCenter} y1={yH} y2={yL} stroke={color} strokeWidth="1" />
            <rect x={xCenter - candleW / 2} y={top} width={candleW} height={bodyH} fill={color} opacity="0.8" />
          </g>
        );
      })}
      {/* X axis baseline */}
      <line x1={padL} y1={h - padB} x2={w - padR} y2={h - padB} stroke="#e5e7eb" />
    </svg>
  );
};

// Formatting helpers available to both App and SummaryBar
const formatMoney = (n) => {
  if (n === "" || n === null || n === undefined) return "";
  const num = Number(n);
  if (!Number.isFinite(num)) return "";
  try {
    return num.toLocaleString("en-US");
  } catch {
    return "";
  }
};
const parseMoney = (s) => {
  if (s === null || s === undefined) return "";
  const str = String(s);
  const cleaned = str.replace(/[^0-9.]/g, "");
  if (cleaned === "") return "";
  const num = Number(cleaned);
  return Number.isFinite(num) ? num : "";
};
const formatCurrency = (n, decimals = 2) => {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return `$${(0).toFixed(decimals)}`;
  return Number(n).toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
};

// Format ISO date string to MM-DD-YY for endpoint indicators
const formatDateMMDDYY = (s) => {
  if (!s) return "";
  const d = new Date(s);
  if (Number.isNaN(d.getTime())) return "";
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  const yy = String(d.getFullYear()).slice(-2);
  return `${mm}-${dd}-${yy}`;
};
const formatNumber = (n) => {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "0";
  return Number(n).toLocaleString("en-US");
};

// Integer input helpers: allow blanks while typing and avoid leading zeros
const parseInteger = (s) => {
  if (s === null || s === undefined) return "";
  const str = String(s);
  const cleaned = str.replace(/[^0-9]/g, "");
  if (cleaned === "") return "";
  const num = Number(cleaned);
  return Number.isFinite(num) ? num : "";
};
const formatInteger = (n) => {
  if (n === "" || n === null || n === undefined) return "";
  const num = Number(n);
  if (!Number.isFinite(num)) return "";
  return String(num);
};

// Decimal input helpers: allow blanks and in-progress strings like '.', '0.'
const parseDecimal = (s) => {
  if (s === null || s === undefined) return "";
  let str = String(s).replace(/[^0-9.]/g, "");
  if (str === "") return "";
  // Keep only the first dot
  const firstDot = str.indexOf('.');
  if (firstDot !== -1) {
    str = str.slice(0, firstDot + 1) + str.slice(firstDot + 1).replace(/\./g, "");
  }
  return str;
};
const formatDecimal = (n) => {
  if (n === "" || n === null || n === undefined) return "";
  if (typeof n === "string") return n;
  const num = Number(n);
  if (!Number.isFinite(num)) return "";
  return String(n);
};

// Simple error boundary to isolate rendering errors in child components (e.g., SummaryBar)
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, info) {
    console.error("ErrorBoundary caught: ", this.props.name || "", error, info);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="mb-4 p-3 text-xs text-red-700 bg-red-50 border border-red-200 rounded">
          A component failed to render{this.props.name ? `: ${this.props.name}` : ""}. Check console for details.
        </div>
      );
    }
    return this.props.children;
  }
}

function App() {
  const [symbol, setSymbol] = useState("TSLA");
  const [dteMin, setDteMin] = useState(1);
  const [dteMax, setDteMax] = useState(45);
  const [minOI, setMinOI] = useState(100);
  const [capital, setCapital] = useState(() => {
    if (typeof window === 'undefined') return 100000;
    try {
      const raw = localStorage.getItem('options.capital');
      if (raw === null) return 100000;
      if (raw === '') return '';
      const n = Number(raw);
      return Number.isFinite(n) ? n : 100000;
    } catch {
      return 100000;
    }
  }); // USD for puts
  const [shareCount, setShareCount] = useState(() => {
    if (typeof window === 'undefined') return 100;
    try {
      const sym = (typeof symbol === 'string' && symbol) ? symbol.toUpperCase() : null;
      let raw = sym ? localStorage.getItem(`options.shareCount.${sym}`) : null;
      if (raw === null) {
        // Back-compat: fall back to legacy global if present on initial load
        raw = localStorage.getItem('options.shareCount');
      }
      if (raw === null) return 100;
      if (raw === '') return '';
      const n = Number(raw);
      return Number.isFinite(n) ? n : 100;
    } catch {
      return 100;
    }
  }); // number of shares for covered calls (per symbol)
  const [targetIncome, setTargetIncome] = useState(() => {
    if (typeof window === 'undefined') return 10000;
    try {
      const raw = localStorage.getItem('options.targetIncome');
      if (raw === null) return 10000;
      if (raw === '') return '';
      const n = Number(raw);
      return Number.isFinite(n) ? n : 10000;
    } catch {
      return 10000;
    }
  }); // USD per month
  const [results, setResults] = useState([]);
  const [strikeLimit, setStrikeLimit] = useState('12'); // how many rows to display: '6' | '12' | '24' | 'all'
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [loading, setLoading] = useState(false);
  const [price, setPrice] = useState(null);
  const [companyName, setCompanyName] = useState(null);
  const [iv, setIv] = useState(null);
  const [priceSource, setPriceSource] = useState(null); // 'realtime' | 'last_close' | 'fallback'
  const [priceLoading, setPriceLoading] = useState(false);
  const [priceError, setPriceError] = useState(null);
  const priceAnchorRef = useRef(null);
  const [showStickyPrice, setShowStickyPrice] = useState(false);
  const [autoScanTimer, setAutoScanTimer] = useState(null);
  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState(null);
  const [historyPeriod, setHistoryPeriod] = useState("1d"); // 1d, 5d, 1mo, 3mo, 6mo, ytd, 1y, 3y, 5y, max
  const [historyInterval, setHistoryInterval] = useState("5m"); // 1m, 5m, 15m, 30m, 1h, 1d
  const [fxError, setFxError] = useState(null);
  const [chartRangeKey, setChartRangeKey] = useState("1D"); // UI label for selected timeframe
  const [chartType, setChartType] = useState("line"); // 'line' | 'candle'
  const [paramsInfoOpen, setParamsInfoOpen] = useState(false);
  const [resultsInfoOpen, setResultsInfoOpen] = useState(false);
  const [capitalInfoOpen, setCapitalInfoOpen] = useState(false);
  const [aboutInfoOpen, setAboutInfoOpen] = useState(false);
  // Backend tuning
  const [useBid, setUseBid] = useState(false);
  const [popOtmFallback, setPopOtmFallback] = useState(0.70);
  const [popItmFallback, setPopItmFallback] = useState(0.30);
  // Delta filters for covered calls
  const [deltaMin, setDeltaMin] = useState(0.0);
  const [deltaMax, setDeltaMax] = useState(0.30);
  // Section accordions
  const [isCapitalOpen, setIsCapitalOpen] = useState(true);
  const [isSymbolOpen, setIsSymbolOpen] = useState(true);
  const [isParamsOpen, setIsParamsOpen] = useState(true);
  // Strategy selection
  const [strategy, setStrategy] = useState("calls"); // "puts" | "calls"

  // Helpers moved to module scope

  // Currency and FX
  const [currency, setCurrency] = useState(() => {
    if (typeof window === 'undefined') return 'USD';
    try {
      const c = localStorage.getItem('options.currency');
      return c === 'CAD' ? 'CAD' : 'USD';
    } catch { return 'USD'; }
  });
  const [fxUsdToCad, setFxUsdToCad] = useState(() => {
    if (typeof window === 'undefined') return 1.39;
    try {
      const raw = localStorage.getItem('options.fxUsdToCad');
      const n = Number(raw);
      if (Number.isFinite(n) && n > 0) {
        // Migrate old default 1.35 to the updated rate 1.39
        return n === 1.35 ? 1.39 : n;
      }
      return 1.39;
    } catch { return 1.39; }
  });
  const currCode = currency === 'CAD' ? 'CAD' : 'USD';
  const toDisplay = (usd) => {
    const n = Number(usd);
    if (!Number.isFinite(n)) return 0;
    return currency === 'CAD' ? n * fxUsdToCad : n;
  };
  const fromDisplay = (val) => {
    const n = Number(val);
    if (!Number.isFinite(n)) return "";
    return currency === 'CAD' ? n / fxUsdToCad : n;
  };
  const formatCurrencyUI = (usd, decimals = 2) => {
    const n = toDisplay(usd);
    try {
      return Number(n).toLocaleString('en-US', {
        style: 'currency',
        currency: currCode,
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      });
    } catch {
      return `${currCode === 'CAD' ? 'CA$' : '$'}${Number(n || 0).toFixed(decimals)}`;
    }
  };

  const handleSymbolSelect = (selectedSymbol) => {
    setSymbol(selectedSymbol);
  };

  // Timeframe presets for mini price chart
  const chartRanges = {
    '1D':  { period: '1d',  interval: '5m' },
    '5D':  { period: '5d',  interval: '30m' },
    '1M':  { period: '1mo', interval: '1d' },
    '3M':  { period: '3mo', interval: '1d' },
    '6M':  { period: '6mo', interval: '1d' },
    'YTD': { period: 'ytd', interval: '1d' },
    '1Y':  { period: '1y',  interval: '1d' },
    '3Y':  { period: '3y',  interval: '1wk' },
    '5Y':  { period: '5y',  interval: '1wk' },
    'ALL': { period: 'max', interval: '1mo' },
  };
  const applyRange = (key) => {
    const cfg = chartRanges[key];
    if (!cfg) return;
    setHistoryPeriod(cfg.period);
    setHistoryInterval(cfg.interval);
    setChartRangeKey(key);
  };
  const isRangeActive = (key) => {
    const cfg = chartRanges[key];
    return cfg && historyPeriod === cfg.period && historyInterval === cfg.interval;
  };

  // Compute change for current chart data (first vs last valid close, or current price if available)
  const computeRangeChange = (series, endOverride = null) => {
    if (!Array.isArray(series) || series.length < 2) return null;
    let i = 0;
    let j = series.length - 1;
    while (i < series.length && !Number.isFinite(Number(series[i]?.close))) i++;
    while (j >= 0 && !Number.isFinite(Number(series[j]?.close))) j--;
    if (i >= j) return null;
    const start = Number(series[i].close);
    const end = Number.isFinite(Number(endOverride)) ? Number(endOverride) : Number(series[j].close);
    if (!Number.isFinite(start) || start === 0 || !Number.isFinite(end)) return null;
    const delta = end - start;
    const pct = (delta / start) * 100;
    return { delta, pct, start, end };
  };

  const rangeChange = computeRangeChange(history, price);

  // Expiry presets for quick DTE filter (for results table)
  const dtePresets = {
    '0-7d':   { min: 0,  max: 7 },
    '8-14d':  { min: 8,  max: 14 },
    '15-30d': { min: 15, max: 30 },
    '30-45d': { min: 30, max: 45 },
    '45-60d': { min: 45, max: 60 },
  };
  const applyDtePreset = (key) => {
    const p = dtePresets[key];
    if (!p) return;
    // Toggle off if already active -> revert to defaults (1–45)
    if (Number(dteMin) === p.min && Number(dteMax) === p.max) {
      setDteMin(1);
      setDteMax(45);
    } else {
      setDteMin(p.min);
      setDteMax(p.max);
    }
  };
  const isDtePresetActive = (key) => {
    const p = dtePresets[key];
    return !!p && Number(dteMin) === p.min && Number(dteMax) === p.max;
  };

  // Fetch current price when symbol changes
  useEffect(() => {
    const fetchPrice = async () => {
      if (!symbol) return;
      setPriceLoading(true);
      setPriceError(null);
      setPriceSource(null);
      try {
        const res = await fetch(apiUrl(`/price?symbol=${encodeURIComponent(symbol)}`));
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (data && typeof data.price === "number") {
          setPrice(data.price);
          setCompanyName(data.name || null);
          setIv(typeof data.iv === "number" ? data.iv : null);
          setPriceSource(data.source || null);
        } else if (data && data.price) {
          setPrice(Number(data.price));
          setCompanyName(data.name || null);
          setIv(typeof data.iv === "number" ? data.iv : null);
          setPriceSource(data.source || null);
        } else {
          setPrice(null);
          setCompanyName(null);
          setIv(null);
          setPriceSource(null);
        }
      } catch (e) {
        console.error("Price fetch error", e);
        setPriceError("Could not fetch price");
        setPrice(null);
        setCompanyName(null);
        setIv(null);
        setPriceSource(null);
      } finally {
        setPriceLoading(false);
      }
    };
    fetchPrice();
  }, [symbol]);

  // Clear selection when results change
  useEffect(() => {
    setSelectedIndex(null);
  }, [results]);

  // Fetch recent price history when symbol or period changes
  useEffect(() => {
    const fetchHistory = async () => {
      if (!symbol) return;
      setHistoryLoading(true);
      setHistoryError(null);
      try {
        const url = `/history?symbol=${encodeURIComponent(symbol)}&period=${encodeURIComponent(historyPeriod)}&interval=${encodeURIComponent(historyInterval)}`;
        const res = await fetch(apiUrl(url));
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const rows = Array.isArray(data?.data) ? data.data : [];
        // Map to recharts-friendly data with compact date labels
        const mapped = rows.map((r) => ({
          date: r.date?.slice(5, 10), // MM-DD for compact x-axis
          dateFull: r.date,            // ISO 8601 for tooltip with year
          open: Number(r.open || r.close || 0),
          high: Number(r.high || r.close || 0),
          low: Number(r.low || r.close || 0),
          close: Number(r.close || 0),
        }));
        setHistory(mapped);
      } catch (e) {
        console.error("History fetch error", e);
        setHistoryError("Could not fetch history");
        setHistory([]);
      } finally {
        setHistoryLoading(false);
      }
    };
    fetchHistory();
  }, [symbol, historyPeriod, historyInterval]);

  // Show the fixed top price bar only after scrolling past the original price location
  useEffect(() => {
    const anchor = priceAnchorRef.current;
    if (!anchor) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        // Reduce jitter by triggering slightly after the anchor passes the top
        setShowStickyPrice(!entry.isIntersecting);
      },
      { root: null, threshold: 0, rootMargin: "-8px 0px 0px 0px" }
    );
    observer.observe(anchor);
    return () => observer.disconnect();
  }, [priceAnchorRef]);

  const handleScan = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      params.set("symbol", String(symbol || ""));
      // Use defaults when fields are blank
      const dteMinSafe = dteMin === "" ? 1 : Number(dteMin);
      const dteMaxSafe = dteMax === "" ? 45 : Number(dteMax);
      const minOISafe = minOI === "" ? 100 : Number(minOI);
      params.set("dte_min", String(dteMinSafe));
      params.set("dte_max", String(dteMaxSafe));
      params.set("min_oi", String(minOISafe));
      // Conservative premium toggle
      params.set("use_bid", String(Boolean(useBid)));

      // Include optional numerics only if positive and finite to avoid 422s
      let capForBackend = Number(capital);
      if (strategy === "calls") {
        const shares = Number(shareCount);
        const px = Number(price);
        if (Number.isFinite(px) && Number.isFinite(shares) && shares > 0) {
          capForBackend = shares * px;
        } else {
          capForBackend = NaN;
        }
      }
      if (Number.isFinite(capForBackend) && capForBackend > 0) {
        params.set("capital", String(capForBackend));
      }
      const tgtNum = Number(targetIncome);
      if (Number.isFinite(tgtNum) && tgtNum > 0) {
        params.set("target_income", String(tgtNum));
      }
      // Backend tuning for POP fallbacks (only send if numeric)
      if (Number.isFinite(Number(popOtmFallback))) {
        params.set("pop_otm_fallback", String(popOtmFallback));
      }
      if (Number.isFinite(Number(popItmFallback))) {
        params.set("pop_itm_fallback", String(popItmFallback));
      }
      // Covered calls: explicitly request OTM-only by default (backend also defaults true)
      if (strategy === "calls") {
        params.set("otm_only", "true");
      }
      // Delta filters (apply to both puts and calls)
      let dMin = (deltaMin === "" || !Number.isFinite(Number(deltaMin))) ? 0.0 : Number(deltaMin);
      let dMax = (deltaMax === "" || !Number.isFinite(Number(deltaMax))) ? 0.30 : Number(deltaMax);
      dMin = Math.max(0, Math.min(1, dMin));
      dMax = Math.max(0, Math.min(1, dMax));
      if (dMin > dMax) {
        const tmp = dMin; dMin = dMax; dMax = tmp;
      }
      params.set("delta_min", String(dMin));
      params.set("delta_max", String(dMax));

      // Use different endpoints based on strategy
      const endpoint = strategy === "calls" ? "/scan-calls" : "/scan";
      const url = `${endpoint}?${params.toString()}`;
      console.log("Fetching URL:", apiUrl(url));
      
      const response = await fetch(apiUrl(url));
      console.log("Response status:", response.status);
      
      if (response.ok) {
        const data = await response.json();
        console.log("Response data", data);
        setResults(Array.isArray(data.results) ? data.results : []);
      } else {
        const errorText = await response.text();
        console.error("Scan failed:", response.status, errorText);
        setResults([]);
      }
    } catch (error) {
      console.error("Scan error:", error);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  // Auto-refresh scan when Capital/Shares, DTE, or Target Income change
  useEffect(() => {
    if (!symbol) return;
    // Debounce to avoid spamming the backend while typing
    if (autoScanTimer) clearTimeout(autoScanTimer);
    const t = setTimeout(() => {
      handleScan();
    }, 400);
    setAutoScanTimer(t);
    return () => clearTimeout(t);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [capital, shareCount, dteMin, dteMax, targetIncome, useBid, popOtmFallback, popItmFallback, deltaMin, deltaMax]);

  // Auto-run scan when symbol changes (debounced)
  useEffect(() => {
    if (!symbol) return;
    if (autoScanTimer) clearTimeout(autoScanTimer);
    const t = setTimeout(() => {
      handleScan();
    }, 300);
    setAutoScanTimer(t);
    return () => clearTimeout(t);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol]);

  // Auto-run scan when strategy changes so table updates between Puts and Calls
  useEffect(() => {
    if (!symbol) return;
    if (autoScanTimer) clearTimeout(autoScanTimer);
    const t = setTimeout(() => {
      handleScan();
    }, 300);
    setAutoScanTimer(t);
    return () => clearTimeout(t);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [strategy]);

  // When symbol changes, load Number of Shares for that symbol; default to 100 if none saved
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const sym = (typeof symbol === 'string' && symbol) ? symbol.toUpperCase() : null;
      const key = sym ? `options.shareCount.${sym}` : null;
      const raw = key ? localStorage.getItem(key) : null;
      if (raw === null || raw === '') {
        setShareCount(100);
      } else {
        const n = Number(raw);
        setShareCount(Number.isFinite(n) ? n : 100);
      }
    } catch {
      setShareCount(100);
    }
  }, [symbol]);

  // Persist key user inputs across refresh
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try { localStorage.setItem('options.capital', capital === '' ? '' : String(capital)); } catch {}
  }, [capital]);
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const val = shareCount === '' ? '' : String(shareCount);
      const sym = (typeof symbol === 'string' && symbol) ? symbol.toUpperCase() : null;
      if (sym) localStorage.setItem(`options.shareCount.${sym}`, val);
      // Keep legacy global key for backward compatibility
      localStorage.setItem('options.shareCount', val);
    } catch {}
  }, [shareCount]);
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try { localStorage.setItem('options.targetIncome', targetIncome === '' ? '' : String(targetIncome)); } catch {}
  }, [targetIncome]);

  // Persist currency & FX
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try { localStorage.setItem('options.currency', currency); } catch {}
  }, [currency]);
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try { localStorage.setItem('options.fxUsdToCad', String(fxUsdToCad)); } catch {}
  }, [fxUsdToCad]);

  // Fetch live FX rate USD->CAD and refresh periodically
  useEffect(() => {
    if (typeof window === 'undefined') return;
    let alive = true;
    const fetchFx = async () => {
      try {
        const res = await fetch(apiUrl('/fx/usd_cad'));
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const rate = Number(data?.rate);
        // Basic sanity bounds to avoid corrupting state on bad responses
        if (alive && Number.isFinite(rate) && rate > 0 && rate < 10) {
          setFxUsdToCad(rate);
          setFxError(null);
        }
      } catch (e) {
        console.warn('FX fetch error', e);
        if (alive) setFxError('FX rate unavailable');
      }
    };
    // Initial fetch on mount / when currency toggles
    fetchFx();
    // Refresh every 30 minutes
    const id = setInterval(fetchFx, 30 * 60 * 1000);
    return () => { alive = false; clearInterval(id); };
  }, [currency]);

  // Auto-run scan when price loads/changes in Calls to apply shares * price capital
  useEffect(() => {
    if (!symbol) return;
    if (strategy !== "calls") return;
    if (!Number.isFinite(Number(price))) return;
    if (autoScanTimer) clearTimeout(autoScanTimer);
    const t = setTimeout(() => {
      handleScan();
    }, 300);
    setAutoScanTimer(t);
    return () => clearTimeout(t);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [price]);

  // Find the row whose strike is closest to the current underlying price (computed later on displayResults)

  // Apply delta filter on client for both calls and puts to adjust displayed strikes to current delta settings (backend also filters)
  const resultsForDisplay = (strategy === "calls" || strategy === "puts")
    ? results.filter((r) => {
        const d = Number(r?.delta_abs);
        // If delta couldn't be computed on backend, keep the row (mimic backend behavior)
        if (!Number.isFinite(d)) return true;
        let dMin = Number(deltaMin);
        let dMax = Number(deltaMax);
        if (!Number.isFinite(dMin)) dMin = 0;
        if (!Number.isFinite(dMax)) dMax = 1;
        dMin = Math.max(0, Math.min(1, dMin));
        dMax = Math.max(0, Math.min(1, dMax));
        if (dMin > dMax) { const t = dMin; dMin = dMax; dMax = t; }
        return d >= dMin && d <= dMax;
      })
    : results;

  // Limit how many results (strikes) are shown in the table
  const displayResults = strikeLimit === 'all'
    ? resultsForDisplay
    : resultsForDisplay.slice(0, Math.max(0, Number(strikeLimit) || 0));

  // Compute closest-to-spot row within the currently displayed results
  const closestDisplayIndex = (price != null && Array.isArray(displayResults) && displayResults.length > 0)
    ? displayResults.reduce((bestI, r, i) => {
        const s = Number(r?.strike);
        if (!Number.isFinite(s)) return bestI;
        const bestS = Number(displayResults[bestI]?.strike);
        const d = Math.abs(s - Number(price));
        const bestD = Math.abs((Number.isFinite(bestS) ? bestS : Infinity) - Number(price));
        return d < bestD ? i : bestI;
      }, 0)
    : -1;

  // If visible strikes are reduced and previously selected row is now hidden, clear selection
  useEffect(() => {
    if (selectedIndex == null) return;
    if (strikeLimit === 'all') return;
    const limit = Number(strikeLimit) || 0;
    if (selectedIndex >= limit) {
      setSelectedIndex(null);
    }
  }, [strikeLimit, selectedIndex]);

  // If delta filter changes hide the selected row, clear selection to avoid mismatched indices
  useEffect(() => {
    if (selectedIndex == null) return;
    if (!Array.isArray(displayResults)) return;
    if (selectedIndex >= displayResults.length) {
      setSelectedIndex(null);
    }
  }, [displayResults.length, selectedIndex]);

  return (
    <div className={`min-h-screen bg-gray-50 text-gray-900 pt-16 px-2 md:px-8 pb-8`}>
          {price != null && showStickyPrice && (
            <div className="fixed top-0 left-0 right-0 z-40 bg-gray-900 border-b border-gray-800 shadow-sm">
              <div className="max-w-7xl mx-auto px-2 md:px-8 py-1 text-sm text-gray-100">
                <div>
                  <div>
                    Current price for <span className="font-semibold text-white">{symbol.toUpperCase()}</span>: <span className="font-semibold text-white">{formatCurrency(price)}</span>
                    {rangeChange ? (
                      <>
                        <span className={`ml-2 inline-flex items-center px-1.5 py-0.5 rounded ${rangeChange.delta >= 0 ? 'bg-green-900 text-green-100' : 'bg-red-900 text-red-100'}`}>
                          {`${rangeChange.delta >= 0 ? '+' : '-'}${Math.abs(rangeChange.pct).toFixed(2)}%`}
                        </span>
                        <span className={`ml-1 font-semibold ${rangeChange.delta >= 0 ? 'text-green-300' : 'text-red-300'}`}>
                          {`${rangeChange.delta >= 0 ? '+' : '-'}${formatCurrency(Math.abs(rangeChange.delta))}`}
                        </span>
                        <span className="ml-1 text-xs text-gray-300">({chartRangeKey})</span>
                      </>
                    ) : null}
                    {priceSource === 'last_close' ? (
                      <span className="ml-2 inline-flex items-center px-1.5 py-0.5 rounded border border-amber-300 bg-amber-100 text-amber-900">
                        Last close
                      </span>
                    ) : null}
                  </div>
                  {companyName ? <div className="text-gray-300">{companyName}</div> : null}
                </div>
              </div>
            </div>
          )}
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl md:text-3xl font-bold text-blue-600">Options Strategy Scanner</h1>
      </div>
      
      {/* Strategy Tabs */}
      <div className="mb-6">
        <div className="border-b border-gray-200">
          <div className="flex items-end justify-between">
            <nav className="-mb-px flex space-x-8">
              <button
                onClick={() => setStrategy("calls")}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  strategy === "calls"
                    ? "border-blue-500 text-blue-600"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                }`}
              >
                Covered Calls
              </button>
              <button
                onClick={() => setStrategy("puts")}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  strategy === "puts"
                    ? "border-blue-500 text-blue-600"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                }`}
              >
                Cash-Secured Puts
              </button>
            </nav>
            <div className="flex flex-col items-end gap-1">
              <span className="text-xs text-gray-600">1 USD = {fxUsdToCad.toFixed(4)} CAD</span>
              {fxError ? (
                <span className="text-xs text-red-600" title={String(fxError)}>FX offline</span>
              ) : null}
              <div className="inline-flex rounded-md border border-gray-300 overflow-hidden">
                <button
                  type="button"
                  className={`px-2 py-1 text-xs ${currency === 'USD' ? 'bg-blue-600 text-white' : 'bg-white text-gray-700 hover:bg-gray-50'}`}
                  onClick={() => setCurrency('USD')}
                >
                  USD
                </button>
                <button
                  type="button"
                  className={`px-2 py-1 text-xs ${currency === 'CAD' ? 'bg-blue-600 text-white' : 'bg-white text-gray-700 hover:bg-gray-50'}`}
                  onClick={() => setCurrency('CAD')}
                >
                  CAD
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Capital & Target (moved to top row) */}
      <div className="bg-white rounded-lg shadow mb-6 p-2 md:p-6">
        <div className="px-3 md:px-0">
        <div className="flex items-center justify-between mb-3">
          <button
            type="button"
            className="flex items-center gap-2 select-none"
            onClick={() => setIsCapitalOpen((v) => !v)}
            aria-expanded={isCapitalOpen}
            aria-controls="capital-section"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              className={`w-5 h-5 transition-transform ${isCapitalOpen ? 'rotate-90' : ''}`}
            >
              <path d="M9 5l7 7-7 7" />
            </svg>
            <h2 className="text-xl font-semibold">
              {strategy === "calls" ? "Shares & Target Monthly Income" : "Capital & Target Monthly Income"}
            </h2>
          </button>
          <div className="flex items-center gap-2">
            {strategy === "puts" && (
              <button
                type="button"
                aria-label="About cash-secured puts"
                className="w-8 h-8 flex items-center justify-center rounded-full border border-gray-300 text-gray-500 hover:text-gray-700 hover:border-gray-400"
                onClick={() => {
                  setIsCapitalOpen(true);
                  setAboutInfoOpen((prev) => {
                    const next = !prev;
                    if (next) setCapitalInfoOpen(false);
                    return next;
                  });
                }}
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                  <path d="M12 2a10 10 0 100 20 10 10 0 000-20zm.75 14.5h-1.5v-6h1.5v6zm0-8h-1.5V7h1.5v1.5z" />
                </svg>
              </button>
            )}
            <button
              type="button"
              aria-label="Capital & Target info"
              className="w-8 h-8 flex items-center justify-center rounded-full border border-gray-300 text-gray-500 hover:text-gray-700 hover:border-gray-400"
              onClick={() => {
                setIsCapitalOpen(true);
                setCapitalInfoOpen((prev) => {
                  const next = !prev;
                  if (next) setAboutInfoOpen(false);
                  return next;
                });
              }}
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                <path d="M12 2a10 10 0 100 20 10 10 0 000-20zm.75 14.5h-1.5v-6h1.5v6zm0-8h-1.5V7h1.5v1.5z" />
              </svg>
            </button>
          </div>
        </div>
        {isCapitalOpen && (
        <>
          {capitalInfoOpen && (
            <div className="mb-3 text-sm rounded-md border border-gray-200 bg-gray-50 p-3">
              {strategy === "calls" ? (
                <div className="space-y-2">
                  <div>
                    <div className="font-semibold">Number of Shares</div>
                    <p>How many shares you own (or plan to own). Covered calls are sized as 1 contract per 100 shares.</p>
                  </div>
                  <div>
                    <div className="font-semibold">Target Monthly Income ({currCode})</div>
                    <p>Your monthly income goal. Used to compute the “Progress to Target” metric in the summary; it does not affect ranking/sizing yet.</p>
                  </div>
                </div>
              ) : (
                <div className="space-y-2">
                  <div>
                    <div className="font-semibold">Capital ({currCode})</div>
                    <p>Cash available to secure puts. Contracts = floor(Capital ÷ (Strike × 100)). Capital Used and Income Total are based on this sizing.</p>
                  </div>
                  <div>
                    <div className="font-semibold">Target Monthly Income ({currCode})</div>
                    <p>Your monthly income goal. Used to compute the “Progress to Target” metric in the summary; it does not affect ranking/sizing yet.</p>
                  </div>
                </div>
              )}
            </div>
          )}
          {aboutInfoOpen && strategy === "puts" && (
            <div className="mb-3 text-sm rounded-md border border-gray-200 bg-gray-50 p-3">
              <h3 className="text-base font-semibold mb-2">How Cash-Secured Puts Work</h3>
              <div className="space-y-3 text-gray-700">
                <p>
                  A cash-secured put is an options strategy where you sell (write) a put option and set aside enough cash to buy the underlying shares if assigned.
                  Each contract controls 100 shares, so the required collateral is typically <span className="font-semibold">Strike × 100</span> (brokers may round or add fees).
                </p>
                <div>
                  <div className="font-semibold">Objective</div>
                  <p>Collect premium for agreeing to buy the stock at the strike price by expiration if the market price falls below that strike.</p>
                </div>
                <div>
                  <div className="font-semibold">Outcomes</div>
                  <ul className="list-disc list-inside space-y-1">
                    <li><span className="font-medium">Expires OTM (stock above strike):</span> you keep the premium; no shares are purchased.</li>
                    <li><span className="font-medium">Assigned (stock below strike):</span> you buy 100 shares per contract at the strike; effective cost basis is <span className="font-semibold">strike − premium</span>.</li>
                  </ul>
                </div>
                <div>
                  <div className="font-semibold">Risks</div>
                  <p>Downside similar to owning the stock from the strike price; the stock can keep falling after assignment. Premium received provides limited downside buffer.</p>
                </div>
                <div>
                  <div className="font-semibold">Sizing</div>
                  <p>Contracts are sized by available capital. This app computes Contracts = floor(Capital ÷ (Strike × 100)).</p>
                </div>
                <div>
                  <div className="font-semibold">Probability of Profit (POP)</div>
                  <p>Estimated probability of finishing profitable at expiration. Use alongside other metrics (liquidity, spreads, DTE, and your thesis).</p>
                </div>
                <div>
                  <div className="font-semibold mb-1">Illustration (Payoff at Expiration)</div>
                  <div className="rounded-md border border-gray-200 p-3 bg-gray-50">
                    <svg viewBox="0 0 520 260" className="w-full h-auto">
                      {/* Axes */}
                      <line x1="40" y1="220" x2="500" y2="220" stroke="#6b7280" strokeWidth="1" />
                      <line x1="40" y1="20" x2="40" y2="220" stroke="#6b7280" strokeWidth="1" />
                      {/* Labels */}
                      <text x="505" y="225" fontSize="12" fill="#374151">Stock Price →</text>
                      <text x="10" y="20" fontSize="12" fill="#374151" transform="rotate(-90 10,20)">Profit/Loss →</text>
                      {/* Reference zero P/L line */}
                      <line x1="40" y1="150" x2="500" y2="150" stroke="#d1d5db" strokeDasharray="4 4" />
                      <text x="45" y="162" fontSize="11" fill="#6b7280">Break-even P/L</text>
                      {/* Strike marker */}
                      <line x1="260" y1="20" x2="260" y2="220" stroke="#e5e7eb" />
                      <text x="250" y="235" fontSize="11" fill="#374151">Strike</text>
                      {/* Premium level text */}
                      <text x="420" y="130" fontSize="11" fill="#059669">= Premium kept</text>
                      {/* Payoff curve */}
                      <polyline
                        fill="none"
                        stroke="#10b981"
                        strokeWidth="2.5"
                        points="40,110 260,110 500,220"
                      />
                      {/* Annotations */}
                      <circle cx="260" cy="110" r="3" fill="#10b981" />
                      <text x="70" y="100" fontSize="11" fill="#065f46">Stock ≥ Strike → keep full premium</text>
                      <text x="265" y="195" fontSize="11" fill="#7f1d1d">Stock &lt; Strike → potential assignment; losses beyond break-even</text>
                    </svg>
                    <div className="mt-2 text-xs text-gray-600">
                      Example payoff of a short put at expiration. Above the strike, the option expires worthless and the premium is kept. Below the strike, you may be assigned and your P/L decreases as price falls (capped by owning shares at strike; premium reduces the loss by the amount collected).
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div id="capital-section" className={"grid grid-cols-1 " + (strategy === "calls" ? "md:grid-cols-3 " : "md:grid-cols-2 ") + "gap-4 items-end"}>
          <div className="flex flex-col h-full justify-end">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              <span className="inline-flex items-center gap-2 max-w-full">
                <span className="truncate min-w-0">
                  {strategy === "calls" ? "Number of Shares" : "Capital"}
                </span>
              </span>
            </label>
            <div className="relative">
              {strategy === "calls" ? (
                <input
                  type="text"
                  inputMode="numeric"
                  value={formatInteger(shareCount)}
                  onChange={(e) => setShareCount(parseInteger(e.target.value))}
                  placeholder="e.g., 100"
                  className="w-full pr-3 pl-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              ) : (
                <>
                  <span aria-hidden="true" className="pointer-events-none absolute inset-y-0 left-3 flex items-center text-gray-400">$</span>
                  <input
                    type="text"
                    inputMode="decimal"
                    value={capital === '' ? '' : formatMoney(toDisplay(capital))}
                    onChange={(e) => {
                      const v = parseMoney(e.target.value);
                      if (v === '') setCapital('');
                      else setCapital(fromDisplay(v));
                    }}
                    className="w-full pr-12 pl-7 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-gray-400 text-sm">{currCode}</span>
                </>
              )}
            </div>
          </div>
          {strategy === "calls" && (
            <div className="flex flex-col h-full justify-end">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                <span className="inline-flex items-center gap-2 max-w-full"><span className="truncate min-w-0">Value of Shares</span></span>
              </label>
              <div className="relative">
                <span aria-hidden="true" className="pointer-events-none absolute inset-y-0 left-3 flex items-center text-gray-400">$</span>
                <input
                  type="text"
                  inputMode="decimal"
                  value={price != null && shareCount !== "" && Number.isFinite(Number(shareCount)) ? formatMoney(toDisplay(Number(shareCount) * Number(price))) : ""}
                  readOnly
                  disabled
                  className="w-full pr-12 pl-7 py-2 border border-gray-200 bg-gray-50 rounded-md focus:outline-none"
                />
                <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-gray-400 text-sm">{currCode}</span>
              </div>
            </div>
          )}
          <div className="flex flex-col h-full justify-end">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              <span className="inline-flex items-center gap-2 max-w-full"><span className="truncate min-w-0">Target Monthly Income</span></span>
            </label>
            <div className="relative">
              <span aria-hidden="true" className="pointer-events-none absolute inset-y-0 left-3 flex items-center text-gray-400">$</span>
              <input
                type="text"
                inputMode="decimal"
                value={targetIncome === '' ? '' : formatMoney(toDisplay(targetIncome))}
                onChange={(e) => {
                  const v = parseMoney(e.target.value);
                  if (v === '') setTargetIncome('');
                  else setTargetIncome(fromDisplay(v));
                }}
                className="w-full pr-12 pl-7 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-gray-400 text-sm">{currCode}</span>
            </div>
          </div>
        </div>
          </>
        )}
        </div>
      </div>

      {/* Symbol Selection */}
      <div className="bg-white rounded-lg shadow mb-6 p-2 md:p-6">
        <div className="flex items-center justify-between mb-3">
          <button
            type="button"
            className="flex items-center gap-2 select-none"
            onClick={() => setIsSymbolOpen((v) => !v)}
            aria-expanded={isSymbolOpen}
            aria-controls="symbol-section"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              className={`w-5 h-5 transition-transform ${isSymbolOpen ? 'rotate-90' : ''}`}
            >
              <path d="M9 5l7 7-7 7" />
            </svg>
            <h2 className="text-xl font-semibold">
              Select Symbol
              <span className="ml-2 text-gray-600 font-normal">
                {symbol?.toUpperCase()}{iv != null ? ` (IV: ${(iv * 100).toFixed(0)}%)` : ''}
              </span>
            </h2>
          </button>
        </div>
        {isSymbolOpen && (
        <div id="symbol-section" className="mb-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <h3 className="text-lg font-medium mb-2">Tech Stocks</h3>
              <div className="flex flex-wrap gap-2 mb-2">
                {["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "PLTR", "AMD"].map((sym) => (
                  <button
                    key={sym}
                    onClick={() => handleSymbolSelect(sym)}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                      symbol === sym
                        ? "bg-blue-600 text-white"
                        : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                    }`}
                  >
                    {sym}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <h3 className="text-lg font-medium mb-2">Indices</h3>
              <div className="flex flex-wrap gap-2 mb-2">
                {["SPY", "QQQ", "DIA", "IWM", "VTI"].map((sym) => (
                  <button
                    key={sym}
                    onClick={() => handleSymbolSelect(sym)}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                      symbol === sym
                        ? "bg-blue-600 text-white"
                        : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                    }`}
                  >
                    {sym}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <h3 className="text-lg font-medium mb-2">Symbol</h3>
              <input
                id="custom-symbol"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter any symbol..."
              />
            </div>
          </div>
          <div ref={priceAnchorRef} className="h-px"></div>
          <div className="mt-2 text-sm text-gray-600">
            {priceLoading ? (
              <span>Fetching price…</span>
            ) : priceError ? (
              <span className="text-red-600">{priceError}</span>
            ) : price != null ? (
              <div>
                <div>
                  Current price for <span className="font-semibold">{symbol.toUpperCase()}</span>: <span className="font-semibold">{formatCurrency(price)}</span>
                  {rangeChange ? (
                    <>
                      <span className={`ml-2 inline-flex items-center px-1.5 py-0.5 rounded ${rangeChange.delta >= 0 ? 'bg-green-200 text-green-700' : 'bg-red-200 text-red-700'}`}>
                        {`${rangeChange.delta >= 0 ? '+' : '-'}${Math.abs(rangeChange.pct).toFixed(2)}%`}
                      </span>
                      <span className={`ml-1 font-semibold ${rangeChange.delta >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {`${rangeChange.delta >= 0 ? '+' : '-'}${formatCurrency(Math.abs(rangeChange.delta))}`}
                      </span>
                      <span className="ml-1 text-xs text-gray-500">({chartRangeKey})</span>
                    </>
                  ) : null}
                  {priceSource === 'last_close' ? (
                    <span className="ml-2 inline-flex items-center px-1.5 py-0.5 rounded border border-amber-200 bg-amber-50 text-amber-700">
                      Last close
                    </span>
                  ) : null}
                </div>
                {companyName ? <div className="text-gray-500">{companyName}</div> : null}
              </div>
            ) : (
              <span>Enter a symbol to see the current price.</span>
            )}
          </div>
          {/* Mini price chart under current price (full width) */}
          <div className="mt-3 w-full">
            {historyLoading ? (
              <div className="text-xs text-gray-500">Loading chart…</div>
            ) : historyError ? (
              <div className="text-xs text-red-600">{historyError}</div>
            ) : history && history.length > 1 ? (
              <div className="w-full h-40">
                {chartType === 'line' ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis
                        dataKey="dateFull"
                        type="category"
                        allowDuplicatedCategory={false}
                        interval="preserveStartEnd"
                        tick={(props) => (
                          <CustomXAxisTick
                            {...props}
                            first={props?.payload?.value === history?.[0]?.dateFull}
                            last={props?.payload?.value === history?.[history.length - 1]?.dateFull}
                          />
                        )}
                        tickLine={false}
                        axisLine={{ stroke: '#ffffff' }}
                        tickFormatter={(v) => formatDateMMDDYY(v)}
                      />
                      <YAxis tick={{ fontSize: 10 }} width={40} domain={["auto", "auto"]} axisLine={{ stroke: '#ffffff' }} tickLine={false} />
                      <ReTooltip content={<MiniTooltip range={chartRangeKey} currency="USD" rate={1} />} cursor={{ stroke: '#ffffff', strokeWidth: 1 }} />
                      <Line type="monotone" dataKey="close" stroke="#2563eb" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="w-full h-full">
                    <CandleMiniChart data={history} />
                  </div>
                )}
              </div>
            ) : (
              <div className="text-xs text-gray-400">No chart data</div>
            )}
            {/* Timeframe toggle below the chart */}
            <div className="mt-2 flex flex-wrap justify-center gap-1 text-xs">
              {Object.keys(chartRanges).map((k) => (
                <button
                  key={k}
                  onClick={() => applyRange(k)}
                  className={`px-2 py-1 rounded border ${isRangeActive(k) ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'}`}
                  aria-pressed={isRangeActive(k)}
                >
                  {k}
                </button>
              ))}
            </div>
          </div>
        </div>
        )}

      {/* Inline info panels are now rendered within the Capital & Target accordion */}

      {/* Close Symbol Selection container */}
      </div>

      {/* Scan Parameters */}
      <div className="bg-white rounded-lg shadow mb-6 p-4 md:p-6">
        <div className="flex items-center justify-between mb-4">
          <button
            type="button"
            className="flex items-center gap-2 select-none"
            onClick={() => setIsParamsOpen((v) => !v)}
            aria-expanded={isParamsOpen}
            aria-controls="params-section"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              className={`w-5 h-5 transition-transform ${isParamsOpen ? 'rotate-90' : ''}`}
            >
              <path d="M9 5l7 7-7 7" />
            </svg>
            <h2 className="text-xl font-semibold">Scan Parameters</h2>
          </button>
          <button
            onClick={handleScan}
            disabled={loading}
            className="w-auto lg:min-w-[140px] px-3 md:px-4 py-2 bg-blue-600 text-white rounded-md text-sm md:text-base hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap -mt-0.5 md:-mt-1"
          >
            {loading ? "Scanning..." : (strategy === "calls" ? "Scan Calls" : "Scan Puts")}
          </button>
        </div>
        {isParamsOpen && (
        <div id="params-section">
          <div className="flex items-center justify-end mb-2">
            <button
              type="button"
              aria-label="Scan Parameters info"
              className="w-8 h-8 flex items-center justify-center rounded-full border border-gray-300 text-gray-500 hover:text-gray-700 hover:border-gray-400"
              onClick={() => setParamsInfoOpen(true)}
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                <path d="M12 2a10 10 0 100 20 10 10 0 000-20zm.75 14.5h-1.5v-6h1.5v6zm0-8h-1.5V7h1.5v1.5z" />
              </svg>
            </button>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-7 xl:grid-cols-8 gap-3 md:gap-5 lg:gap-6 xl:gap-8 items-end">
          <div className="flex flex-col h-full justify-end col-span-1 lg:col-span-2 min-w-0">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <span className="inline-flex items-center gap-2 max-w-full"><span className="truncate min-w-0">DTE Min</span></span>
            </label>
            <input
              type="text"
              inputMode="numeric"
              value={formatInteger(dteMin)}
              onChange={(e) => setDteMin(parseInteger(e.target.value))}
              className="w-full px-3 py-2.5 lg:px-2 lg:py-2.5 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm lg:text-sm"
            />
          </div>
          <div className="flex flex-col h-full justify-end col-span-1 lg:col-span-2 min-w-0">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <span className="inline-flex items-center gap-2 max-w-full"><span className="truncate min-w-0">DTE Max</span></span>
            </label>
            <input
              type="text"
              inputMode="numeric"
              value={formatInteger(dteMax)}
              onChange={(e) => setDteMax(parseInteger(e.target.value))}
              className="w-full px-3 py-2.5 lg:px-2 lg:py-2.5 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm lg:text-sm"
            />
          </div>
          <div className="flex flex-col h-full justify-end col-span-1 lg:col-span-2 min-w-0">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <span className="inline-flex items-center gap-2 max-w-full"><span className="truncate min-w-0">Min Open Interest</span></span>
            </label>
            <input
              type="text"
              inputMode="numeric"
              value={formatInteger(minOI)}
              onChange={(e) => setMinOI(parseInteger(e.target.value))}
              className="w-full px-3 py-2.5 lg:px-2 lg:py-2.5 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm lg:text-sm"
            />
          </div>
          {/* Force desktop line break so toggles start on a new row */}
          <div className="hidden lg:block col-span-1 lg:col-span-7 xl:col-span-8" aria-hidden="true" />
          <>
            <div className="flex flex-col h-full justify-end col-span-1 lg:col-span-1 min-w-0">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <span className="inline-flex items-center gap-2 max-w-full">
                  <span className="truncate min-w-0">Delta Min</span>
                </span>
              </label>
              <input
                type="text"
                inputMode="decimal"
                value={formatDecimal(deltaMin)}
                onChange={(e) => setDeltaMin(parseDecimal(e.target.value))}
                className="w-full px-3 py-2.5 lg:px-2 lg:py-2.5 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm lg:text-sm"
              />
              <div className="text-[11px] text-gray-500 mt-1">0–1 abs (e.g., 0.15)</div>
            </div>
            <div className="flex flex-col h-full justify-end col-span-1 lg:col-span-1 min-w-0">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <span className="inline-flex items-center gap-2 max-w-full">
                  <span className="truncate min-w-0">Delta Max</span>
                </span>
              </label>
              <input
                type="text"
                inputMode="decimal"
                value={formatDecimal(deltaMax)}
                onChange={(e) => setDeltaMax(parseDecimal(e.target.value))}
                className="w-full px-3 py-2.5 lg:px-2 lg:py-2.5 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm lg:text-sm"
              />
              <div className="text-[11px] text-gray-500 mt-1">0–1 abs (e.g., 0.30)</div>
            </div>
          </>
          {/* Conservative premium toggle */}
          <div className="flex flex-col h-full justify-end col-span-1 lg:col-span-2 xl:col-span-2 min-w-0">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <span className="inline-flex items-center gap-2 max-w-full"><span className="truncate min-w-0">Use Bid (Conservative)</span></span>
            </label>
            <label className="flex items-center h-[40px] cursor-pointer select-none w-full">
              <input
                id="use-bid"
                type="checkbox"
                checked={useBid}
                onChange={(e) => setUseBid(e.target.checked)}
                className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
              <span className="ml-2 lg:ml-3 text-sm text-gray-700 pr-3 sm:pr-6 lg:pr-10 xl:pr-16">Conservative premium uses Bid for income and yields</span>
            </label>
            {/* Invisible helper to match POP OTM helper text height for vertical alignment */}
            <div className="text-[11px] mt-1 invisible select-none">0–1 (default 0.70)</div>
          </div>
          {/* POP fallback controls */}
          <div className="flex flex-col h-full justify-end col-span-1 lg:col-span-1 min-w-0">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <span className="inline-flex items-center gap-2 max-w-full">
                <span className="truncate min-w-0">POP OTM Fallback</span>
              </span>
            </label>
            <input
              type="text"
              inputMode="decimal"
              value={formatDecimal(popOtmFallback)}
              onChange={(e) => setPopOtmFallback(parseDecimal(e.target.value))}
              className="w-full px-3 py-2.5 lg:px-2 lg:py-2.5 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm lg:text-sm"
            />
            <div className="text-[11px] text-gray-500 mt-1">0–1 (default 0.70)</div>
          </div>
          <div className="flex flex-col h-full justify-end col-span-1 lg:col-span-1 min-w-0">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <span className="inline-flex items-center gap-2 max-w-full">
                <span className="truncate min-w-0">POP ITM Fallback</span>
              </span>
            </label>
            <input
              type="text"
              inputMode="decimal"
              value={formatDecimal(popItmFallback)}
              onChange={(e) => setPopItmFallback(parseDecimal(e.target.value))}
              className="w-full px-3 py-2.5 lg:px-2 lg:py-2.5 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm lg:text-sm"
            />
            <div className="text-[11px] text-gray-500 mt-1">0–1 (default 0.30)</div>
          </div>
        </div>
        </div>
        )}

        {paramsInfoOpen && (
          <div className="fixed inset-0 z-50" role="dialog" aria-modal="true">
            <div className="absolute inset-0 bg-black/40" onClick={() => setParamsInfoOpen(false)} />
            <div className="absolute inset-0 flex items-center justify-center p-4" onClick={() => setParamsInfoOpen(false)}>
              <div className="w-full max-w-lg bg-white rounded-lg shadow-xl p-5" onClick={(e) => e.stopPropagation()}>
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-lg font-semibold">Scan Parameters — Definitions</h3>
                  <button
                    type="button"
                    aria-label="Close"
                    className="p-2 text-gray-500 hover:text-gray-700"
                    onClick={() => setParamsInfoOpen(false)}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                      <path d="M6.225 4.811L4.811 6.225 10.586 12l-5.775 5.775 1.414 1.414L12 13.414l5.775 5.775 1.414-1.414L13.414 12l5.775-5.775-1.414-1.414L12 10.586 6.225 4.811z"/>
                    </svg>
                  </button>
                </div>
                <div className="space-y-3 text-sm text-gray-700">
                  <div>
                    <div className="font-semibold">DTE Min</div>
                    <p>Minimum Days-To-Expiration for options to include. Contracts with fewer days than this are excluded. Lower values yield faster premium decay; higher values include more time value.</p>
                  </div>
                  <div>
                    <div className="font-semibold">DTE Max</div>
                    <p>Maximum Days-To-Expiration for options to include. Contracts with more days than this are excluded, focusing the scan on nearer expirations.</p>
                  </div>
                  <div>
                    <div className="font-semibold">Min Open Interest</div>
                    <p>Minimum number of outstanding contracts required to include an option. Higher OI generally implies better liquidity and tighter bid/ask spreads.</p>
                  </div>
                </div>
                <div className="mt-4 text-right">
                  <button className="px-4 py-2 rounded-md bg-blue-600 text-white hover:bg-blue-700" onClick={() => setParamsInfoOpen(false)}>Close</button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {resultsInfoOpen && (
        <div className="fixed inset-0 z-50" role="dialog" aria-modal="true">
          <div className="absolute inset-0 bg-black/40" onClick={() => setResultsInfoOpen(false)} />
          <div className="absolute inset-0 flex items-center justify-center p-4" onClick={() => setResultsInfoOpen(false)}>
            <div className="w-full max-w-lg bg-white rounded-lg shadow-xl p-5" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-semibold">Results — Definitions</h3>
                <button
                  type="button"
                  aria-label="Close"
                  className="p-2 text-gray-500 hover:text-gray-700"
                  onClick={() => setResultsInfoOpen(false)}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                    <path d="M6.225 4.811L4.811 6.225 10.586 12l-5.775 5.775 1.414 1.414L12 13.414l5.775 5.775 1.414-1.414L13.414 12l5.775-5.775-1.414-1.414L12 10.586 6.225 4.811z"/>
                  </svg>
                </button>
              </div>
              <div className="space-y-3 text-sm text-gray-700">
                <div className="text-xs text-gray-600">
                  <span className="font-semibold">Columns:</span> STRIKE · EXPIRY · BID · ASK · OI · DELTA · $/CONTRACT · CONTRACTS · CAPITAL USED · INCOME/CONTRACT · INCOME TOTAL · POP · SCORE
                </div>
                <div className="text-xs text-gray-600">
                  <span className="font-semibold">Scan Parameters:</span> Filters and options that shape the results.
                </div>
                <div>
                  <div className="font-semibold">Delta Min</div>
                  <p>Minimum absolute delta to include (applies to both strategies). Range 0–1 (e.g., 0.15). Lower delta = further OTM.</p>
                </div>
                <div>
                  <div className="font-semibold">Delta Max</div>
                  <p>Maximum absolute delta to include (applies to both strategies). Range 0–1 (e.g., 0.30).</p>
                </div>
                <div>
                  <div className="font-semibold">Use Bid (Conservative)</div>
                  <p>When enabled, uses Bid instead of Mid to compute premium and income. Produces more conservative estimates.</p>
                </div>
                <div>
                  <div className="font-semibold">POP OTM Fallback</div>
                  <p>Fallback Probability of Profit used when the option is OTM and IV is unavailable. Range 0–1 (default 0.70).</p>
                </div>
                <div>
                  <div className="font-semibold">POP ITM Fallback</div>
                  <p>Fallback Probability of Profit used when the option is ITM and IV is unavailable. Range 0–1 (default 0.30).</p>
                </div>
                <div>
                  <div className="font-semibold">Strike</div>
                  <p>The option strike price. For cash-secured puts this is the price at which you may be obligated to buy 100 shares per contract if assigned.</p>
                </div>
                <div>
                  <div className="font-semibold">Expiry</div>
                  <p>Days to expiration (DTE) for the option contract. Shorter DTE typically has faster time decay; longer DTE includes more time value.</p>
                </div>
                <div>
                  <div className="font-semibold">Bid</div>
                  <p>The highest price a buyer is willing to pay for the option (per share). Option premiums are quoted per share; 1 contract = 100 shares.</p>
                </div>
                <div>
                  <div className="font-semibold">Ask</div>
                  <p>The lowest price a seller is willing to accept for the option (per share). The difference between Ask and Bid is the bid/ask spread.</p>
                </div>
                <div>
                  <div className="font-semibold">OI (Open Interest)</div>
                  <p>The number of outstanding contracts that currently exist. Higher OI generally implies better liquidity and tighter spreads.</p>
                </div>
                <div>
                  <div className="font-semibold">POP (Probability of Profit)</div>
                  <p>Estimated probability the position finishes profitable at expiration. Currently a simplified proxy; consider alongside other metrics.</p>
                </div>
                <div>
                  <div className="font-semibold">$/Contract</div>
                  <p>Capital required per contract, typically Strike × 100 for cash-secured puts (may include broker rounding or fees in some cases).</p>
                </div>
                <div>
                  <div className="font-semibold">Contracts</div>
                  <p>Number of contracts sized using your Capital. Computed as floor(Capital ÷ $/Contract). If $/Contract exceeds Capital, this will be 0.</p>
                </div>
                <div>
                  <div className="font-semibold">Capital Used</div>
                  <p>Total capital reserved for the position: $/Contract × Contracts.</p>
                </div>
                <div>
                  <div className="font-semibold">Income/Contract</div>
                  <p>Estimated premium received per contract. If available, uses model mid-price; otherwise derived from Bid/Ask. Quoted in USD per contract (×100 shares).</p>
                </div>
                <div>
                  <div className="font-semibold">Income Total</div>
                  <p>Total estimated premium for the position: Income/Contract × Contracts.</p>
                </div>
                <div>
                  <div className="font-semibold">Score</div>
                  <p>A composite ranking metric to compare candidates (higher is better). May incorporate yield, liquidity, DTE, and other heuristics.</p>
                </div>
              </div>
              <div className="mt-4 text-right">
                <button className="px-4 py-2 rounded-md bg-blue-600 text-white hover:bg-blue-700" onClick={() => setResultsInfoOpen(false)}>Close</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      <div className="-mx-8 md:mx-0 p-0 md:p-6 bg-transparent md:bg-white rounded-none md:rounded-lg shadow-none md:shadow">
        <div className="flex flex-col gap-2 md:gap-3 mb-4 px-[60px] md:px-0">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold">Results — {strategy === "calls" ? "Call" : "Put"}</h2>
            <button
              type="button"
              aria-label="Results info"
              className="w-8 h-8 flex items-center justify-center rounded-full border border-gray-300 text-gray-500 hover:text-gray-700 hover:border-gray-400"
              onClick={() => setResultsInfoOpen(true)}
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                <path d="M12 2a10 10 0 100 20 10 10 0 000-20zm.75 14.5h-1.5v-6h1.5v6zm0-8h-1.5V7h1.5v1.5z" />
              </svg>
            </button>
          </div>
          {/* DTE Preset Toggle */}
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-xs text-gray-600 mr-1">Expiry:</span>
            {Object.keys(dtePresets).map((k) => (
              <button
                key={`dte-${k}`}
                type="button"
                onClick={() => applyDtePreset(k)}
                className={`px-2 py-1 rounded border text-xs ${isDtePresetActive(k) ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'}`}
                aria-pressed={isDtePresetActive(k)}
                title={`Show expiries ${dtePresets[k].min}–${dtePresets[k].max} days`}
              >
                {k}
              </button>
            ))}
          </div>
          {/* Strikes Display Count Toggle */}
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-xs text-gray-600 mr-1">Strikes:</span>
            {['6','12','24','all'].map((k) => {
              const active = String(strikeLimit) === String(k);
              return (
                <button
                  key={`strikes-${k}`}
                  type="button"
                  onClick={() => setStrikeLimit(k)}
                  className={`px-2 py-1 rounded border text-xs ${active ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'}`}
                  aria-pressed={active}
                  title={`Show ${k === 'all' ? 'all' : k} strikes`}
                >
                  {k.toUpperCase()}
                </button>
              );
            })}
            <button
              type="button"
              onClick={handleScan}
              disabled={loading}
              className="ml-2 inline-flex items-center px-2 py-1 rounded border text-xs bg-white text-gray-700 border-gray-300 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              title="Refresh option prices"
              aria-label="Refresh option prices"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                <path fillRule="evenodd" d="M3.172 7a7 7 0 111.664 7.606.75.75 0 10-1.07 1.05A8.5 8.5 0 102.25 6.75V5a.75.75 0 00-1.5 0v3A.75.75 0 001.5 8h3a.75.75 0 000-1.5H3.172z" clipRule="evenodd" />
              </svg>
              <span className="ml-1">Refresh</span>
            </button>
          </div>
        </div>
        {/* Summary */}
        <div className="px-16 md:px-0">
          <ErrorBoundary name="SummaryBar">
            <SummaryBar results={displayResults} targetIncome={targetIncome} selected={
              selectedIndex != null && selectedIndex >= 0 && Array.isArray(displayResults) && selectedIndex < displayResults.length
                ? displayResults[selectedIndex]
                : null
            } currency={currency} rate={fxUsdToCad} />
          </ErrorBoundary>
        </div>
        {loading ? (
          <div className="text-center py-8">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <p className="mt-2 text-gray-600">Scanning for options...</p>
          </div>
        ) : displayResults.length > 0 ? (
          <div className="-mx-1 md:mx-0">
            <div className="overflow-x-auto pl-8 pr-8 md:px-0">
              <table className="min-w-full table-auto border-collapse">
                <thead className="bg-gray-50 border-b border-gray-200">
                  <tr>
                    <th className="px-2 py-2 md:px-4 md:py-3 text-left text-[10px] md:text-xs font-medium text-gray-500 uppercase tracking-wider whitespace-nowrap" title="Strike">
                      <span className="hidden md:inline">Strike</span><span className="md:hidden">STRIKE</span>
                    </th>
                    <th className="px-2 py-2 md:px-4 md:py-3 text-left text-[10px] md:text-xs font-medium text-gray-500 uppercase tracking-wider whitespace-nowrap" title="Expiry">
                      <span className="hidden md:inline">Expiry</span><span className="md:hidden">EXPIRY</span>
                    </th>
                    <th className="px-2 py-2 md:px-4 md:py-3 text-left text-[10px] md:text-xs font-medium text-gray-500 uppercase tracking-wider whitespace-nowrap" title="Bid">
                      <span className="hidden md:inline">Bid</span><span className="md:hidden">BID</span>
                    </th>
                    <th className="px-2 py-2 md:px-4 md:py-3 text-left text-[10px] md:text-xs font-medium text-gray-500 uppercase tracking-wider whitespace-nowrap" title="Ask">
                      <span className="hidden md:inline">Ask</span><span className="md:hidden">ASK</span>
                    </th>
                    <th className="px-2 py-2 md:px-4 md:py-3 text-left text-[10px] md:text-xs font-medium text-gray-500 uppercase tracking-wider whitespace-nowrap" title="Last">
                      <span className="hidden md:inline">Last</span><span className="md:hidden">LAST</span>
                    </th>
                    <th className="px-2 py-2 md:px-4 md:py-3 text-left text-[10px] md:text-xs font-medium text-gray-500 uppercase tracking-wider whitespace-nowrap" title="Open Interest">
                      <span className="hidden md:inline">OI</span><span className="md:hidden">OI</span>
                    </th>
                    <th className="px-2 py-2 md:px-4 md:py-3 text-left text-[10px] md:text-xs font-medium text-gray-500 uppercase tracking-wider whitespace-nowrap" title="Delta (abs)">
                      <span className="hidden md:inline">Delta</span><span className="md:hidden">DELTA</span>
                    </th>
                    <th className="px-2 py-2 md:px-4 md:py-3 text-left text-[10px] md:text-xs font-medium text-gray-500 uppercase tracking-wider whitespace-nowrap" title="Dollars Per Contract">
                      <span className="hidden md:inline">$/Contract</span><span className="md:hidden">$/CONTRACT</span>
                    </th>
                    <th className="px-2 py-2 md:px-4 md:py-3 text-left text-[10px] md:text-xs font-medium text-gray-500 uppercase tracking-wider whitespace-nowrap" title="Contracts">
                      <span className="hidden md:inline">Contracts</span><span className="md:hidden">CONTRACTS</span>
                    </th>
                    <th className="px-2 py-2 md:px-4 md:py-3 text-left text-[10px] md:text-xs font-medium text-gray-500 uppercase tracking-wider whitespace-nowrap" title={strategy === "calls" ? "Share Value" : "Capital Used"}>
                      <span className="hidden md:inline">{strategy === "calls" ? "Share Value" : "Capital Used"}</span><span className="md:hidden">{strategy === "calls" ? "SHARE VALUE" : "CAPITAL USED"}</span>
                    </th>
                    <th className="px-2 py-2 md:px-4 md:py-3 text-left text-[10px] md:text-xs font-medium text-gray-500 uppercase tracking-wider whitespace-nowrap" title="Income Per Contract">
                      <span className="hidden md:inline">Income/Contract</span><span className="md:hidden">INCOME/CONTRACT</span>
                    </th>
                    <th className="px-2 py-2 md:px-4 md:py-3 text-left text-[10px] md:text-xs font-medium text-gray-500 uppercase tracking-wider whitespace-nowrap" title="Income Total">
                      <span className="hidden md:inline">Income Total</span><span className="md:hidden">INCOME TOTAL</span>
                    </th>
                    <th className="px-2 py-2 md:px-4 md:py-3 text-left text-[10px] md:text-xs font-medium text-gray-500 uppercase tracking-wider whitespace-nowrap" title={strategy === "calls" ? "Probability of Keeping Premium" : "Probability of Profit"}>
                      <span className="hidden md:inline">POP</span><span className="md:hidden">POP</span>
                    </th>
                    <th className="px-2 py-2 md:px-4 md:py-3 text-left text-[10px] md:text-xs font-medium text-gray-500 uppercase tracking-wider whitespace-nowrap" title="Score">
                      <span className="hidden md:inline">Score</span><span className="md:hidden">SCORE</span>
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white">
                  {displayResults.map((put, i) => {
                    const isClosest = i === closestDisplayIndex;
                    let isITM = false;
                    if (price != null && Number.isFinite(Number(put?.strike))) {
                      const strikeNum = Number(put.strike);
                      const priceNum = Number(price);
                      isITM = strategy === "calls" ? (strikeNum < priceNum) : (strikeNum > priceNum);
                    }
                    const tdBase = `px-2 py-2 md:px-4 md:py-3 text-xs md:text-sm text-gray-500 group-hover:text-blue-700 border-b border-gray-200`;
                    const tdBaseSelected = `px-2 py-2 md:px-4 md:py-3 text-xs md:text-sm text-gray-500 group-hover:text-blue-700 border-y-2 border-blue-400`;
                    const tdBaseNoBottom = `px-2 py-2 md:px-4 md:py-3 text-xs md:text-sm text-gray-500 group-hover:text-blue-700`;
                    return (
                      <React.Fragment key={`result-frag-${i}`}>
                        <tr
                          onClick={() => setSelectedIndex(i)}
                          aria-selected={selectedIndex === i}
                          className={`group cursor-pointer transition-colors duration-150 hover:bg-blue-50 ${isITM ? 'bg-green-50/60' : ''}`}
                        >
                          {(() => {
                            const selected = selectedIndex === i;
                            const isAboveSelected = selectedIndex === i + 1;
                            const baseClass = selected ? tdBaseSelected : (isAboveSelected ? tdBaseNoBottom : tdBase);
                            return (
                              <>
                                <td className={`${baseClass} text-gray-900 font-semibold`}>{formatCurrency(Number(put?.strike))}</td>
                                <td className={baseClass}>{put.days_to_expiry}d</td>
                                <td className={baseClass}>{formatCurrency(put.bid)}</td>
                                <td className={baseClass}>{formatCurrency(put.ask)}</td>
                                <td className={baseClass}>{formatCurrency(put.last)}</td>
                                <td className={baseClass}>{formatNumber(put.open_interest)}</td>
                                <td className={baseClass}>{put?.delta_abs != null ? Number(put.delta_abs).toFixed(2) : 'N/A'}</td>
                                <td className={baseClass}>{formatCurrencyUI(put.capital_per_contract ?? (put.strike * 100), 0)}</td>
                                <td className={baseClass}>{formatNumber(put.contracts ?? 0)}</td>
                                <td className={baseClass}>{formatCurrencyUI(put.capital_used ?? 0, 0)}</td>
                                <td className={baseClass}>{formatCurrencyUI(put.income_per_contract ?? put.mid)}</td>
                                <td className={baseClass}>{formatCurrencyUI(put.income_total ?? 0)}</td>
                                <td className={baseClass}>{put.pop != null ? `${(Number(put.pop) * 100).toFixed(0)}%` : 'N/A'}</td>
                                <td className={baseClass}>{put.score?.toFixed(2) || 'N/A'}</td>
                              </>
                            );
                          })()}
                        </tr>
                        {isClosest && (
                          <tr aria-hidden="true">
                            <td colSpan={14} className="p-0 border-b-2 border-green-600"></td>
                          </tr>
                        )}
                      </React.Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <p>
              {results.length > 0
                ? "No results match your current delta range. Try widening Delta Min/Max."
                : `Click "${strategy === "calls" ? "Scan Calls" : "Scan Puts"}" to see results here.`}
            </p>
            <p className="text-sm mt-1">Try adjusting your parameters to find more options.</p>
          </div>
        )}
      </div>
    </div>
  );
}

function SummaryBar({ results, targetIncome, selected, currency = 'USD', rate = 1 }) {
  const safeResults = Array.isArray(results) ? results : [];
  let best = null;
  for (const r of safeResults) {
    const income = Number(r?.income_total ?? 0);
    if (!best || income > Number(best?.income_total ?? 0)) best = r;
  }

  const target = Number(targetIncome ?? 0) || 0;
  const useRow = selected || best;
  const rowIncome = Number(useRow?.income_total ?? 0) || 0;
  const rowCapital = Number(useRow?.capital_used ?? 0) || 0;
  const rowContracts = Number(useRow?.contracts ?? 0) || 0;
  const pct = target > 0 ? (rowIncome / target) * 100 : null;

  const currCode = currency === 'CAD' ? 'CAD' : 'USD';
  const toDisplay = (usd) => {
    const n = Number(usd);
    if (!Number.isFinite(n)) return 0;
    return currCode === 'CAD' ? n * Number(rate) : n;
  };
  const fmt = (usd, decimals = 2) => {
    const n = toDisplay(usd);
    try {
      return Number(n).toLocaleString('en-US', {
        style: 'currency',
        currency: currCode,
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      });
    } catch {
      return `${currCode === 'CAD' ? 'CA$' : '$'}${Number(n || 0).toFixed(decimals)}`;
    }
  };

  return (
    <div className="mb-4 p-4 bg-gray-50 border border-gray-200 rounded">
      <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:gap-6 text-sm">
        <div className="inline-flex items-center gap-2 w-full sm:w-full lg:w-auto">
          <span className="text-gray-500">Target Income:</span>{" "}
          <span className="font-semibold">{fmt(target)}</span>
        </div>
        <div className="flex flex-col w-full lg:flex-row lg:items-center lg:gap-2 lg:w-auto">
          <div className="inline-flex items-center gap-2 w-full sm:w-full lg:w-auto">
            <span className="text-gray-500">{selected ? 'Selected Position Income:' : 'Best Single Position Income:'}</span>
            <span className="font-semibold">{fmt(rowIncome)}</span>
          </div>
          {useRow && (
            <span className="text-gray-500 block lg:inline lg:ml-1">
              (Contracts: {rowContracts.toLocaleString()}, Capital Used: {fmt(rowCapital)})
            </span>
          )}
        </div>
        <div className="inline-flex items-center gap-2 w-full sm:w-full lg:w-auto">
          <span className="text-gray-500">Progress to Target:</span>{" "}
          {pct !== null ? (
            <span className="font-semibold">{pct.toFixed(2)}%</span>
          ) : (
            <span className="text-gray-500">Set a target</span>
          )}
        </div>
      </div>
    </div>
  );
}

// Reusable info tooltip with icon; content comes from children
function InfoTooltip({ children, ariaLabel = "More info" }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative inline-block">
      <button
        type="button"
        aria-label={ariaLabel}
        className="w-5 h-5 flex items-center justify-center rounded-full border border-gray-300 text-gray-500 hover:text-gray-700 hover:border-gray-400"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((v) => !v)}
      >
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
          <path d="M12 2a10 10 0 100 20 10 10 0 000-20zm.75 14.5h-1.5v-6h1.5v6zm0-8h-1.5V7h1.5v1.5z" />
        </svg>
      </button>
      {open && (
        <div className="absolute z-20 left-1/2 -translate-x-1/2 mt-2 w-80 p-3 text-xs bg-white border border-gray-200 rounded shadow-lg">
          {children}
        </div>
      )}
    </div>
  );
}

export default App;

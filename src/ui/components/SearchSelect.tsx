// Searchable single-select dropdown (used for model / GPU catalogs).

import { useEffect, useMemo, useRef, useState } from 'react';

export interface SearchOption {
  id: string;
  name: string;
  sub?: string; // secondary line with key specs
  tag?: string; // small badge, e.g. MoE / Dense
}

interface Props {
  options: SearchOption[];
  value: string;
  onChange: (id: string) => void;
  placeholder?: string;
}

const MAX_RENDER = 400;

export function SearchSelect({ options, value, onChange, placeholder }: Props) {
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  const current = options.find((o) => o.id === value);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return options;
    return options.filter((o) => `${o.name} ${o.id}`.toLowerCase().includes(q));
  }, [options, query]);

  useEffect(() => {
    function onDocPointerDown(e: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener('mousedown', onDocPointerDown);
    return () => document.removeEventListener('mousedown', onDocPointerDown);
  }, []);

  return (
    <div className="search-select" ref={rootRef}>
      <input
        className="input"
        value={open ? query : current?.name ?? ''}
        placeholder={placeholder ?? '搜索…'}
        onFocus={() => {
          setOpen(true);
          setQuery('');
        }}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Escape') setOpen(false);
        }}
      />
      {open && (
        <div className="search-list">
          {filtered.slice(0, MAX_RENDER).map((o) => (
            <button
              key={o.id}
              type="button"
              className={`search-item${o.id === value ? ' active' : ''}`}
              onClick={() => {
                onChange(o.id);
                setOpen(false);
              }}
            >
              <span className="search-name">
                {o.name}
                {o.tag ? <em className={`tag${o.tag === 'MoE' ? ' tag-moe' : ''}`}>{o.tag}</em> : null}
              </span>
              {o.sub ? <span className="search-sub">{o.sub}</span> : null}
            </button>
          ))}
          {filtered.length > MAX_RENDER && (
            <div className="search-empty">… 其余 {filtered.length - MAX_RENDER} 项，请输入关键词过滤</div>
          )}
          {filtered.length === 0 && <div className="search-empty">无匹配项</div>}
        </div>
      )}
    </div>
  );
}
